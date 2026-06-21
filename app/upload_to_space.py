
from huggingface_hub import upload_folder

HF_SPACE_REPO = "deepakpandit08/visit-with-us-wellness-app"

upload_folder(
    folder_path="tourism_project/app",
    repo_id=HF_SPACE_REPO,
    repo_type="space"
)

print("Deployment files uploaded to Hugging Face Space successfully.")
