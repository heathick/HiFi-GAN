import os

DATA_DIR = "data/ruslan"
FOLDER_URL = "https://drive.google.com/drive/folders/1QjaIKtPHmj-baiUMjjQqe8XjZ5XpiNoC"

os.makedirs(DATA_DIR, exist_ok=True)

os.system(f"gdown --folder -O {DATA_DIR} {FOLDER_URL}")

tar_path = os.path.join(DATA_DIR, "RUSLAN.tar.gz")
os.system(f"tar -xzf {tar_path} -C {DATA_DIR}")

print("DONE!")