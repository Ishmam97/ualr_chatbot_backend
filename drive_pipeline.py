import os
import io
import subprocess
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# === CONFIG ===
SERVICE_ACCOUNT_FILE = './service_account.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = '1wN3PrGWBN6b0HvP7d4CERxemNYhd7RQM'
DEST_DIR = 'data'

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build('drive', 'v3', credentials=creds)

def download_folder_files(service):
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and trashed = false",
        fields="files(id, name)",
    ).execute()
    files = results.get('files', [])

    for file in files:
        file_id = file['id']
        file_name = file['name']
        dest_path = os.path.join(DEST_DIR, file_name)

        if os.path.exists(dest_path):
            print(f"Skipping (already exists): {file_name}")
            continue

        print(f"Downloading: {file_name}")
        request = service.files().get_media(fileId=file_id)
        with open(dest_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

def main():
    service = get_drive_service()
    download_folder_files(service)

    print("Files synced. Running pipeline.py...")
    subprocess.run(["python", "pipeline.py"], check=True)

if __name__ == '__main__':
    main()
