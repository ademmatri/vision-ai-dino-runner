from huggingface_hub import HfApi

api = HfApi()

print("Uploading 372 MB model folder to Hugging Face...")

# Creates the repo on Hugging Face (if it doesn't exist) and uploads model files
api.create_repo(repo_id="ademmatri/vit-hand-gesture-dino", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="./my_finetuned_vit",
    repo_id="ademmatri/vit-hand-gesture-dino",
    repo_type="model"
)

print("? Upload Complete! Model is now hosted online.")
