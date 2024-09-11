from pathlib import Path
import pandas as pd
import shutil

if __name__ == '__main__':
    dir = Path('dataset')
    metadata = dir.joinpath('tpsm_metadata.csv')

    train_dir = dir.joinpath('tpsm_train')
    train_dir.mkdir(exist_ok=True)

    test_dir = dir.joinpath('tpsm_test')
    test_dir.mkdir(exist_ok=True)

    train_dir.joinpath('audio').mkdir(exist_ok=True)
    test_dir.joinpath('audio').mkdir(exist_ok=True)

    train_dir.joinpath('keypoints').mkdir(exist_ok=True)
    test_dir.joinpath('keypoints').mkdir(exist_ok=True)

    df = pd.read_csv(str(metadata))

    for d in df.itertuples():
        file_name = d.video_id + '#' + str(d.start).zfill(6) + '#' + str(d.end).zfill(6)
        audio_file = dir.joinpath('audio', file_name + '.mp3')
        keypoints_file = dir.joinpath('keypoints', file_name + '.pt')

        if not audio_file.exists() or not keypoints_file.exists():
            continue
        shutil.copy2(audio_file, dir.joinpath(f'tpsm_{d.partition}', 'audio', file_name + '.mp3'))
        shutil.copy2(keypoints_file, dir.joinpath(f'tpsm_{d.partition}', 'keypoints', file_name + '.pt'))
