import subprocess
import sys

def update_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

packages = [
    "torch",
    "transformers",
    "datasets",
    "tqdm",
    "pyyaml",
    "scikit-learn",
    "tensorboard",
    "safetensors",
]

print("Updating packages...")
for package in packages:
    print(f"Updating {package}...")
    update_package(package)
print("All packages updated successfully!")
