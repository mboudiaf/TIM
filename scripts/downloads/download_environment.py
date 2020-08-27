import os
from scripts.downloads.utils import download_file_from_google_drive


def download_env():
    id = '1-4WhU-bM3wJFcYV0kIyokShddmQFNXTM'
    name = f"env.zip"
    print('Start Download (may take a few minutes)')
    download_file_from_google_drive(id, name)
    print('Finish Download')
    os.system(f'unzip {name}')
    os.system(f'rm -r {name}')


if __name__ == '__main__':
    download_env()