"""End-to-end integration tests for the Home Node ↔ Coordination API flow.

These tests spin up a **real** Coordination API (uvicorn subprocess with SQLite)
and a **real** Home Node TLS server, then exercise the full lifecycle:

  1. Node registers via EIP-191 challenge-verify (on-chain stake is mocked
     via SR_MOCK_STAKE_WEI env var injected into the coordination API)
  2. Coordination API persists the node in SQLite
  3. Home Node proxies HTTP and CONNECT requests through its TLS server
  4. Node deregisters on shutdown

The Coordination API runs as a uvicorn subprocess.
The Home Node TLS server is started via asyncio.start_server (same as production).
No respx mocking — real HTTP calls between the services.

Requirements:
  - space-router-node deps (httpx, eth-account, cryptography, pydantic-settings)
  - coordination-api deps (fastapi, uvicorn, web3, eth-account)
  - Both repos available at ../space-router/coordination-api/

Run:
  cd space-router-node
  .venv/bin/python -m pytest tests/e2e/test_e2e_integration.py -v -s
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import signal
import socket
import ssl
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from eth_account import Account

# Node-side imports
from app.config import Settings as NodeSettings
from app.identity import NodeIdentity
from app.registration import deregister_node, register_node
from app.tls import create_server_ssl_context, ensure_certificates
from app.proxy_handler import handle_client

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_NODE_ROOT = Path(__file__).resolve().parents[2]
_COORD_ROOT = _NODE_ROOT.parent / "space-router" / "coordination-api"
_COORD_PYTHON = _COORD_ROOT / ".venv" / "bin" / "python"

# ---------------------------------------------------------------------------
# Deterministic test wallets
# ---------------------------------------------------------------------------
_TEST_KEY = "0x" + "ab" * 32
_TEST_ACCOUNT = Account.from_key(_TEST_KEY)
_TEST_ADDRESS = _TEST_ACCOUNT.address.lower()
_TEST_IDENTITY = NodeIdentity(_TEST_KEY)

_TEST_KEY_2 = "0x" + "cd" * 32
_TEST_ACCOUNT_2 = Account.from_key(_TEST_KEY_2)
_TEST_ADDRESS_2 = _TEST_ACCOUNT_2.address.lower()
_TEST_IDENTITY_2 = NodeIdentity(_TEST_KEY_2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_node_ssl_context(cert_path: str) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _wait_for_http(url: str, timeout: float = 10.0) -> None:
    """Poll until the URL returns 200."""
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadError, OSError):
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError(f"Service at {url} did not become ready within {timeout}s")


# ---------------------------------------------------------------------------
# Fixture: Coordination API subprocess
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def coordination_api(tmp_path):
    """Start the Coordination API as a uvicorn subprocess with stake mocking."""
    if not _COORD_ROOT.is_dir():
        pytest.skip(f"Coordination API not found at {_COORD_ROOT}")
    if not _COORD_PYTHON.is_file():
        pytest.skip(f"Coordination API venv not found at {_COORD_PYTHON}")

    db_path = str(tmp_path / "test_e2e.db")
    port = _free_port()

    # The coordination API reads SR_ env vars; we also set a mock env var
    # that our patched _get_staked_wei reads (see _coord_startup_patch.py)
    env = {
        **os.environ,
        "SR_USE_SQLITE": "true",
        "SR_SQLITE_DB_PATH": db_path,
        "SR_INTERNAL_API_SECRET": "e2e-test-secret",
        "SR_IPINFO_TOKEN": "",
        "SR_LOG_LEVEL": "WARNING",
        # Mock: bypass real RPC by setting a high stake amount
        "SR_MOCK_STAKE_WEI": str(2000 * 10**18),
    }

    # Write a small wrapper that monkey-patches _get_staked_wei before
    # starting uvicorn so we don't need real Creditcoin RPC
    wrapper = tmp_path / "_coord_e2e_runner.py"
    wrapper.write_text(f"""\
import os, sys
sys.path.insert(0, {str(_COORD_ROOT)!r})

# Monkey-patch the on-chain stake check before the app loads
import app.routers.registration as reg
_mock_wei = int(os.environ.get("SR_MOCK_STAKE_WEI", "0"))
if _mock_wei:
    reg._get_staked_wei = lambda address: _mock_wei

import uvicorn
from app.main import app
uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
""")

    proc = subprocess.Popen(
        [str(_COORD_PYTHON), str(wrapper)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        await _wait_for_http(f"{base_url}/healthz", timeout=15.0)
    except RuntimeError:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(
            f"Coordination API failed to start.\n"
            f"stdout: {stdout.decode()}\n"
            f"stderr: {stderr.decode()}"
        )

    yield {"base_url": base_url, "port": port, "db_path": db_path, "proc": proc}

    # Shutdown
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


# ---------------------------------------------------------------------------
# Fixture: Home Node TLS server
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def home_node(tmp_path, coordination_api):
    """Start a real Home Node TLS server on a random port."""
    cert_path = str(tmp_path / "node.crt")
    key_path = str(tmp_path / "node.key")
    ensure_certificates(cert_path, key_path)

    node_port = _free_port()
    node_settings = NodeSettings(
        NODE_PORT=node_port,
        COORDINATION_API_URL=coordination_api["base_url"],
        NODE_LABEL="e2e-test-node",
        NODE_REGION="us-west",
        NODE_TYPE="residential",
        PUBLIC_IP="127.0.0.1",
        BUFFER_SIZE=65536,
        REQUEST_TIMEOUT=10.0,
        RELAY_TIMEOUT=30.0,
        LOG_LEVEL="DEBUG",
        TLS_CERT_PATH=cert_path,
        TLS_KEY_PATH=key_path,
        UPNP_ENABLED=False,
    )

    ssl_ctx = create_server_ssl_context(cert_path, key_path)
    handler = functools.partial(handle_client, settings=node_settings)
    server = await asyncio.start_server(
        handler, host="127.0.0.1", port=node_port, ssl=ssl_ctx,
    )

    yield {
        "settings": node_settings,
        "port": node_port,
        "cert_path": cert_path,
        "key_path": key_path,
        "server": server,
    }

    server.close()
    await server.wait_closed()


# ===========================================================================
# Registration lifecycle tests
# ===========================================================================

class TestRegistrationE2E:
    """Full registration lifecycle against a real Coordination API."""

    @pytest.mark.asyncio
    async def test_full_registration_lifecycle(self, coordination_api):
        """Challenge → sign → verify → deregister (raw HTTP calls)."""
        base_url = coordination_api["base_url"]

        async with httpx.AsyncClient() as client:
            # Step 1: Challenge
            resp = await client.post(
                f"{base_url}/nodes/register/challenge",
                json={"address": _TEST_ADDRESS},
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert "nonce" in data
            assert data["expires_in"] == 300
            nonce = data["nonce"]

            # Step 2: Sign and verify
            signed_nonce = _TEST_IDENTITY.sign_message(nonce)
            resp = await client.post(
                f"{base_url}/nodes/register/verify",
                json={
                    "address": _TEST_ADDRESS,
                    "endpoint_url": "https://127.0.0.1:9090",
                    "signed_nonce": signed_nonce,
                    "public_ip": "127.0.0.1",
                    "connectivity_type": "direct",
                    "region": "us-west",
                    "label": "e2e-test",
                },
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["status"] == "registered"
            assert data["address"] == _TEST_ADDRESS
            node_id = data["node_id"]

            # Step 3: Deregister
            resp = await client.patch(
                f"{base_url}/nodes/{node_id}/status",
                json={"status": "offline"},
            )
            assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_registration_via_node_functions(self, coordination_api):
        """Use register_node() / deregister_node() from the home-node codebase
        against the real Coordination API — proves cross-project compatibility."""
        base_url = coordination_api["base_url"]

        node_settings = NodeSettings(
            NODE_PORT=9090,
            COORDINATION_API_URL=base_url,
            NODE_LABEL="integration-node",
            NODE_REGION="eu-west",
            NODE_TYPE="residential",
            PUBLIC_IP="",
        )

        async with httpx.AsyncClient() as client:
            node_id = await register_node(
                client, node_settings, "10.0.0.1", _TEST_IDENTITY,
            )
            assert node_id is not None
            assert len(node_id) > 0

            await deregister_node(client, node_settings, node_id)

    @pytest.mark.asyncio
    async def test_wrong_signature_rejected(self, coordination_api):
        """Verify rejects a signature from a different key."""
        base_url = coordination_api["base_url"]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/nodes/register/challenge",
                json={"address": _TEST_ADDRESS},
            )
            assert resp.status_code == 200
            nonce = resp.json()["nonce"]

            # Sign with wallet 2 (wrong key)
            wrong_sig = _TEST_IDENTITY_2.sign_message(nonce)
            resp = await client.post(
                f"{base_url}/nodes/register/verify",
                json={
                    "address": _TEST_ADDRESS,
                    "endpoint_url": "https://1.2.3.4:9090",
                    "signed_nonce": wrong_sig,
                    "public_ip": "1.2.3.4",
                },
            )
            assert resp.status_code == 403
            assert "Signature mismatch" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_nonce_consumed_after_use(self, coordination_api):
        """A nonce can only be used once."""
        base_url = coordination_api["base_url"]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{base_url}/nodes/register/challenge",
                json={"address": _TEST_ADDRESS},
            )
            nonce = resp.json()["nonce"]
            signed = _TEST_IDENTITY.sign_message(nonce)

            payload = {
                "address": _TEST_ADDRESS,
                "endpoint_url": "https://1.2.3.4:9090",
                "signed_nonce": signed,
                "public_ip": "1.2.3.4",
            }

            # First verify succeeds
            resp = await client.post(
                f"{base_url}/nodes/register/verify", json=payload,
            )
            assert resp.status_code == 200

            # Second verify fails — nonce consumed
            resp = await client.post(
                f"{base_url}/nodes/register/verify", json=payload,
            )
            assert resp.status_code == 400
            assert "No pending challenge" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_sybil_prevention_same_ip(self, coordination_api):
        """Two different wallets cannot register with the same public IP."""
        base_url = coordination_api["base_url"]

        async with httpx.AsyncClient() as client:
            # Register wallet 1 with IP 10.0.0.1
            resp = await client.post(
                f"{base_url}/nodes/register/challenge",
                json={"address": _TEST_ADDRESS},
            )
            nonce = resp.json()["nonce"]
            signed = _TEST_IDENTITY.sign_message(nonce)
            resp = await client.post(
                f"{base_url}/nodes/register/verify",
                json={
                    "address": _TEST_ADDRESS,
                    "endpoint_url": "https://10.0.0.1:9090",
                    "signed_nonce": signed,
                    "public_ip": "10.0.0.1",
                },
            )
            assert resp.status_code == 200

            # Try wallet 2 with the SAME IP — should be rejected
            resp = await client.post(
                f"{base_url}/nodes/register/challenge",
                json={"address": _TEST_ADDRESS_2},
            )
            nonce2 = resp.json()["nonce"]
            signed2 = _TEST_IDENTITY_2.sign_message(nonce2)
            resp = await client.post(
                f"{base_url}/nodes/register/verify",
                json={
                    "address": _TEST_ADDRESS_2,
                    "endpoint_url": "https://10.0.0.1:9091",
                    "signed_nonce": signed2,
                    "public_ip": "10.0.0.1",
                },
            )
            assert resp.status_code == 409
            assert "already registered" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_re_registration_updates_node(self, coordination_api):
        """Same wallet re-registering should update (upsert), not duplicate."""
        base_url = coordination_api["base_url"]

        async with httpx.AsyncClient() as client:
            # First registration
            resp = await client.post(
                f"{base_url}/nodes/register/challenge",
                json={"address": _TEST_ADDRESS},
            )
            nonce = resp.json()["nonce"]
            signed = _TEST_IDENTITY.sign_message(nonce)
            resp = await client.post(
                f"{base_url}/nodes/register/verify",
                json={
                    "address": _TEST_ADDRESS,
                    "endpoint_url": "https://1.2.3.4:9090",
                    "signed_nonce": signed,
                    "public_ip": "1.2.3.4",
                    "label": "first-run",
                },
            )
            assert resp.status_code == 200
            first_node_id = resp.json()["node_id"]
            assert resp.json()["status"] == "registered"

            # Second registration (same wallet, new endpoint)
            resp = await client.post(
                f"{base_url}/nodes/register/challenge",
                json={"address": _TEST_ADDRESS},
            )
            nonce2 = resp.json()["nonce"]
            signed2 = _TEST_IDENTITY.sign_message(nonce2)
            resp = await client.post(
                f"{base_url}/nodes/register/verify",
                json={
                    "address": _TEST_ADDRESS,
                    "endpoint_url": "https://5.6.7.8:9090",
                    "signed_nonce": signed2,
                    "public_ip": "5.6.7.8",
                    "label": "second-run",
                },
            )
            assert resp.status_code == 200
            assert resp.json()["node_id"] == first_node_id
            assert resp.json()["status"] == "updated"

    @pytest.mark.asyncio
    async def test_two_nodes_different_ips(self, coordination_api):
        """Two wallets can register with different IPs."""
        base_url = coordination_api["base_url"]

        async with httpx.AsyncClient() as client:
            # Node 1
            settings1 = NodeSettings(
                NODE_PORT=9090,
                COORDINATION_API_URL=base_url,
                NODE_LABEL="node-1",
                NODE_REGION="us-east",
                PUBLIC_IP="",
            )
            node_id_1 = await register_node(
                client, settings1, "10.0.0.1", _TEST_IDENTITY,
            )

            # Node 2
            settings2 = NodeSettings(
                NODE_PORT=9091,
                COORDINATION_API_URL=base_url,
                NODE_LABEL="node-2",
                NODE_REGION="eu-west",
                PUBLIC_IP="",
            )
            node_id_2 = await register_node(
                client, settings2, "10.0.0.2", _TEST_IDENTITY_2,
            )

            assert node_id_1 != node_id_2

            await deregister_node(client, settings1, node_id_1)
            await deregister_node(client, settings2, node_id_2)

    @pytest.mark.asyncio
    async def test_node_re_registers_after_offline(self, coordination_api):
        """A node that went offline can re-register and keeps the same node_id."""
        base_url = coordination_api["base_url"]

        settings = NodeSettings(
            NODE_PORT=9090,
            COORDINATION_API_URL=base_url,
            NODE_LABEL="restart-node",
            PUBLIC_IP="",
        )

        async with httpx.AsyncClient() as client:
            node_id = await register_node(
                client, settings, "10.0.0.1", _TEST_IDENTITY,
            )
            await deregister_node(client, settings, node_id)

            # Re-register (simulates node restart with new IP)
            node_id_2 = await register_node(
                client, settings, "10.0.0.2", _TEST_IDENTITY,
            )
            assert node_id_2 == node_id


# ===========================================================================
# Home Node proxy tests (TLS server + registration + proxying)
# ===========================================================================

class TestHomeNodeProxyE2E:
    """Tests that exercise the Home Node TLS server with real TCP connections."""

    @pytest.mark.asyncio
    async def test_register_and_proxy_http(self, coordination_api, home_node):
        """Full flow: register → proxy HTTP request → get response."""
        node_port = home_node["port"]
        node_settings = home_node["settings"]

        # Register
        async with httpx.AsyncClient() as client:
            node_id = await register_node(
                client, node_settings, "127.0.0.1", _TEST_IDENTITY,
            )
            assert node_id is not None

        # Start a simple HTTP target server
        target_port = _free_port()
        target_body = b"Hello from target!"
        target_response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: " + str(len(target_body)).encode() + b"\r\n"
            b"Connection: close\r\n"
            b"\r\n" + target_body
        )

        async def _target_handler(reader, writer):
            await reader.read(4096)
            writer.write(target_response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        target_server = await asyncio.start_server(
            _target_handler, "127.0.0.1", target_port,
        )

        try:
            # Connect to Home Node via TLS
            ssl_ctx = _make_node_ssl_context(home_node["cert_path"])
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", node_port, ssl=ssl_ctx,
            )

            # Send absolute-URI HTTP proxy request
            request = (
                f"GET http://127.0.0.1:{target_port}/test HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{target_port}\r\n"
                f"\r\n"
            ).encode()
            writer.write(request)
            await writer.drain()

            # Read response
            response = b""
            try:
                while True:
                    chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                    if not chunk:
                        break
                    response += chunk
            except asyncio.TimeoutError:
                pass

            writer.close()
            await writer.wait_closed()

            assert b"200 OK" in response
            assert b"Hello from target!" in response

        finally:
            target_server.close()
            await target_server.wait_closed()

        # Deregister
        async with httpx.AsyncClient() as client:
            await deregister_node(client, node_settings, node_id)

    @pytest.mark.asyncio
    async def test_connect_tunnel(self, coordination_api, home_node):
        """Register → CONNECT tunnel → bidirectional data through node."""
        node_port = home_node["port"]
        node_settings = home_node["settings"]

        # Register
        async with httpx.AsyncClient() as client:
            node_id = await register_node(
                client, node_settings, "127.0.0.1", _TEST_IDENTITY,
            )

        # Echo server as tunnel target
        target_port = _free_port()

        async def _echo_handler(reader, writer):
            data = await reader.read(4096)
            writer.write(b"ECHO:" + data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        target_server = await asyncio.start_server(
            _echo_handler, "127.0.0.1", target_port,
        )

        try:
            ssl_ctx = _make_node_ssl_context(home_node["cert_path"])
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", node_port, ssl=ssl_ctx,
            )

            # CONNECT request
            connect_req = (
                f"CONNECT 127.0.0.1:{target_port} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{target_port}\r\n"
                f"\r\n"
            ).encode()
            writer.write(connect_req)
            await writer.drain()

            # Read 200 Connection Established
            established = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=5.0,
            )
            assert b"200" in established

            # Tunnel is open — send data
            writer.write(b"ping")
            await writer.drain()

            # Read echo back
            response = await asyncio.wait_for(reader.read(4096), timeout=5.0)
            assert response == b"ECHO:ping"

            writer.close()
            await writer.wait_closed()

        finally:
            target_server.close()
            await target_server.wait_closed()

        async with httpx.AsyncClient() as client:
            await deregister_node(client, node_settings, node_id)

    @pytest.mark.asyncio
    async def test_rejects_bad_request(self, home_node):
        """Home Node returns 400 for malformed requests."""
        ssl_ctx = _make_node_ssl_context(home_node["cert_path"])

        reader, writer = await asyncio.open_connection(
            "127.0.0.1", home_node["port"], ssl=ssl_ctx,
        )

        writer.write(b"NOT_A_VALID_REQUEST\r\n\r\n")
        await writer.drain()

        response = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        assert b"400" in response

        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_returns_502_for_unreachable_target(self, home_node):
        """Home Node returns 502 when target is unreachable."""
        ssl_ctx = _make_node_ssl_context(home_node["cert_path"])

        reader, writer = await asyncio.open_connection(
            "127.0.0.1", home_node["port"], ssl=ssl_ctx,
        )

        dead_port = _free_port()
        request = (
            f"GET http://127.0.0.1:{dead_port}/nope HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{dead_port}\r\n"
            f"\r\n"
        ).encode()
        writer.write(request)
        await writer.drain()

        response = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        assert b"502" in response

        writer.close()
        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_strips_spacerouter_headers(self, coordination_api, home_node):
        """Internal X-SpaceRouter-* headers should not be forwarded to target."""
        node_port = home_node["port"]
        node_settings = home_node["settings"]

        # Register
        async with httpx.AsyncClient() as client:
            await register_node(
                client, node_settings, "127.0.0.1", _TEST_IDENTITY,
            )

        # Target server that echoes back received headers
        target_port = _free_port()
        received_headers = {}

        async def _header_echo_handler(reader, writer):
            data = await reader.read(4096)
            # Parse headers from the forwarded request
            lines = data.decode("latin-1", errors="replace").split("\r\n")
            for line in lines[1:]:  # skip request line
                if ":" in line:
                    k, _, v = line.partition(":")
                    received_headers[k.strip().lower()] = v.strip()
                elif line == "":
                    break

            body = json.dumps(received_headers).encode()
            resp = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                b"Connection: close\r\n"
                b"\r\n" + body
            )
            writer.write(resp)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        target_server = await asyncio.start_server(
            _header_echo_handler, "127.0.0.1", target_port,
        )

        try:
            ssl_ctx = _make_node_ssl_context(home_node["cert_path"])
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", node_port, ssl=ssl_ctx,
            )

            # Send request with internal headers that should be stripped
            request = (
                f"GET http://127.0.0.1:{target_port}/test HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{target_port}\r\n"
                f"X-SpaceRouter-Node-Id: secret-node-id\r\n"
                f"X-SpaceRouter-Request-Id: req-123\r\n"
                f"Proxy-Authorization: Basic dGVzdDp0ZXN0\r\n"
                f"X-Custom-Header: keep-this\r\n"
                f"\r\n"
            ).encode()
            writer.write(request)
            await writer.drain()

            response = b""
            try:
                while True:
                    chunk = await asyncio.wait_for(reader.read(4096), timeout=5.0)
                    if not chunk:
                        break
                    response += chunk
            except asyncio.TimeoutError:
                pass

            writer.close()
            await writer.wait_closed()

            assert b"200 OK" in response

            # Verify internal headers were stripped
            assert "x-spacerouter-node-id" not in received_headers
            assert "x-spacerouter-request-id" not in received_headers
            assert "proxy-authorization" not in received_headers
            # Custom header should be preserved
            assert received_headers.get("x-custom-header") == "keep-this"

        finally:
            target_server.close()
            await target_server.wait_closed()
