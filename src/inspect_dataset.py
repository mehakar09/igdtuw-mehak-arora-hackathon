from huggingface_hub import HfApi

api = HfApi()

repo = "Zihan1004/FNSPID"

files = api.list_repo_files(
    repo_id=repo,
    repo_type="dataset"
)

print("Files available in FNSPID:\n")

for file in files:
    print(file)