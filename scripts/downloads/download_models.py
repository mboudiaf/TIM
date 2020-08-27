import os
from scripts.downloads.utils import download_file_from_google_drive


ids = {'checkpoints': '15MFsig6pjXO7vZdo-1znJoXtHv4NY-AF'}


def download_models(name):
    id = ids[name]
    name = f"{name}.zip"
    print('Start Download (may take a few minutes)')
    download_file_from_google_drive(id, name)
    print('Finish Download')
    os.system(f'unzip {name}')
    os.system(f'rm -r {name}')


if __name__ == '__main__':
    download_models('checkpoints')
