""" Update version in package.json, Cargo.toml and tauri.conf.json from pyproject.toml """
import json
import toml
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Updating versions...")

# Read master version
with open("pyproject.toml", "r") as f:
    config = toml.load(f)
    version = config["project"]["version"]

logging.info(f"Current version is {version}")

# Update package.json
with open("hal-ui/package.json", "r") as f:
    pkg = json.load(f)
old_version = pkg["version"]
pkg["version"] = version
with open("hal-ui/package.json", "w") as f:
    json.dump(pkg, f, indent=2)

logging.info(f"Updated package.json from {old_version} to {version}")

# Update Cargo.toml
with open("hal-ui/src-tauri/Cargo.toml", "r") as f:
    cargo = toml.load(f)
old_cargo_version = cargo["package"]["version"]
cargo["package"]["version"] = version
with open("hal-ui/src-tauri/Cargo.toml", "w") as f:
    toml.dump(cargo, f)

logging.info(f"Updated Cargo.toml from {old_cargo_version} to {version}")

# Update tauri.conf.json
with open("hal-ui/src-tauri/tauri.conf.json", "r") as f:
    tauri = json.load(f)
old_tauri_version = tauri["version"]
tauri["version"] = version
with open("hal-ui/src-tauri/tauri.conf.json", "w") as f:
    json.dump(tauri, f, indent=2)

logging.info(f"Updated tauri.conf.json from {old_tauri_version} to {version}")

logging.info("All version updates completed successfully.")