from huggingface_hub import snapshot_download
import os

# Function to download a model
def download_model(model_name, local_dir, ignore_patterns=None, hf_token=None):
    # Ensure the directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Download the model
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Do not use symlinks
        token=hf_token,  # Use the provided Hugging Face token
        ignore_patterns=ignore_patterns,  # Ignore unnecessary files
        resume_download=True  # Support resuming the download if interrupted
    )

    print(f"Model has been downloaded to: {local_dir}")

# List of model names and corresponding local storage paths
models = [
    {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "local_dir": "/data/models/Llama-2-7b-chat-hf"
    },
    {
        "model_name": "meta-llama/Llama-2-13b-chat-hf",
        "local_dir": "/data/models/Llama-2-13b-chat-hf"
    },
    {
        "model_name":"meta-llama/Llama-2-70b-chat-hf",
        "local_dir": "/data/models/Llama-2-70b-chat-hf"
    },
    {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": "/data/models/Llama-3.1-8B-Instruct"
    },
    {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "local_dir": "/data/models/Llama-3.2-3B-Instruct"
    },
    {
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "local_dir": "/data/models/Llama-3.3-70B-Instruct"
    },
    {
        "model_name": "microsoft/deberta-v2-xlarge-mnli",
        "local_dir": "/models/deberta-v2-xlarge-mnli"
    },
    {
        "model_name": "lucadiliello/BLEURT-20",
        "local_dir": "/models/BLEURT-20",
    }
]

# Hugging Face token (replace with your own token)
hf_token = "your_hugging_face_token"  # Replace with your actual token

# Download all models
for model in models:
    download_model(model["model_name"], model["local_dir"], model.get("ignore_patterns"), hf_token)

print("All models have been downloaded!")
