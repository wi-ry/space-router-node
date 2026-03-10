import os
import questionary

def run_wizard():
    print("Welcome to the Space Router Node Setup Wizard!")
    
    config = {}
    config['REGION'] = questionary.select(
        "Select your deployment region:",
        choices=["US-East", "EU-West", "AP-Northeast", "Global"]
    ).ask()
    
    config['NODE_LABEL'] = questionary.text("Enter a label for this node:", default="my-residential-node").ask()
    config['COORDINATION_API_URL'] = questionary.text(
        "Coordination API URL:", 
        default="https://api.spacerouter.net"
    ).ask()
    
    with open(".env", "w") as f:
        for k, v in config.items():
            f.write(f"{k}={v}\n")
            
    print("Setup complete! Configuration saved to .env")

if __name__ == "__main__":
    run_wizard()
