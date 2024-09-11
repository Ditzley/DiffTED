from pathlib import Path
import pandas as pd
import shutil
from skimage import io, img_as_float32, transform
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
import imageio
import numpy as np
from tqdm import tqdm

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(imageio.mimread(name, memtest=False))

        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = video #img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

if __name__ == '__main__':
    dir = Path('dataset')
    metadata = dir.joinpath('tpsm_metadata.csv')

    train_dir = dir.joinpath('tpsm_train')
    train_dir.mkdir(exist_ok=True)

    test_dir = dir.joinpath('tpsm_test')
    test_dir.mkdir(exist_ok=True)

    train_dir.joinpath('images').mkdir(exist_ok=True)
    test_dir.joinpath('images').mkdir(exist_ok=True)

    df = pd.read_csv(str(metadata))

    for d in tqdm(df.itertuples()):
        file_name = d.video_id + '#' + str(d.start).zfill(6) + '#' + str(d.end).zfill(6)
        video_file = dir.joinpath('video', f'{file_name}.mp4')
        if video_file.exists():
            video = read_video(str(video_file), (384, 384, 3))
            save_dir = dir.joinpath(f'tpsm_{d.partition}', 'images', file_name)
            save_dir.mkdir(exist_ok=True)
            for i, f in enumerate(video):
                imageio.imwrite(save_dir.joinpath(f'{i:0>4}.png'), f)
