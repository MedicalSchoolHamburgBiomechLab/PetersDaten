from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_data_file_path():
    import os
    one_drive_root = Path(os.getenv('ONE_DRIVE_ROOT'))
    path = one_drive_root.joinpath('Dokumente')
    path = path.joinpath('projects')
    path = path.joinpath('PetersDaten')
    path = path.joinpath('continuousstatistics')
    path = path.joinpath('DatenPeter.xlsx')
    assert path.exists()
    return path
