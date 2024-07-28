import wget
import os
import subprocess
import sys

def download_file(url):
    print(f"Downloading {url}...")
    filename = url.split("/")[-1]
    wget.download(url, out=filename)
    print(f"\nDownload complete: {filename}")

def update_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

def main():
    urls = [
        "https://huggingface.co/AI-MO/NuminaMath-7B-CoT/resolve/main/model-00001-of-00003.safetensors",
        "https://huggingface.co/AI-MO/NuminaMath-7B-CoT/resolve/main/model-00002-of-00003.safetensors",
        "https://huggingface.co/AI-MO/NuminaMath-7B-CoT/resolve/main/model-00003-of-00003.safetensors"
    ]

    for url in urls:
        download_file(url)

    print("All downloads completed.")
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

if __name__ == "__main__":
    main()




