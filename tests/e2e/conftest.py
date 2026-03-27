"""Pytest configuration for e2e integration tests."""

import pytest


# Use auto mode so async fixtures and tests work without explicit markers
def pytest_collection_modifyitems(items):
    """Auto-add asyncio marker to all async test functions."""
    for item in items:
        if item.get_closest_marker("asyncio") is None:
            if hasattr(item, "function") and hasattr(item.function, "__wrapped__"):
                item.add_marker(pytest.mark.asyncio)
