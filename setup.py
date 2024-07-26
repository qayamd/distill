import wget
import os

def download_file(url):
    print(f"Downloading {url}...")
    filename = url.split("/")[-1]
    wget.download(url, out=filename)
    print(f"\nDownload complete: {filename}")

def main():
    urls = [
        "https://huggingface.co/AI-MO/NuminaMath-7B-CoT/resolve/main/model-00001-of-00003.safetensors",
        "https://huggingface.co/AI-MO/NuminaMath-7B-CoT/resolve/main/model-00002-of-00003.safetensors",
        "https://huggingface.co/AI-MO/NuminaMath-7B-CoT/resolve/main/model-00003-of-00003.safetensors"
    ]

    for url in urls:
        download_file(url)

    print("All downloads completed.")

if __name__ == "__main__":
    main()
