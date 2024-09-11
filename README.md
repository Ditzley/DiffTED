# DiffTED: One-shot Audio-driven TED Talk Video Generation with Diffusion-based Co-speech Gestures

This is the official code for **DiffTED: One-shot Audio-driven TED Talk Video Generation with
Diffusion-based Co-speech Gestures**

## Installation
To install dependencies run:
1. Install packages
    ```bash
    pip install -r requirements.txt
    ```
2. Install Thin-Plate-Spline-Motion-Model
   
    Follow directions at install https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model/
       
    Place repository at ```scripts/tps/```
## Dataset

Follow instructions from [MRAA](https://github.com/snap-research/articulated-animation) to download TED-talks dataset.

## Training
To train model run:
```bash
python scripts/train_tpsm.py --config config/pose_diffusion.yml
```

## Inference
To run generate videos run:
```bash
python scripts/test_tpsm.py long <checkpoint_path> <test_data_path>
```

## Citation
If you find our work useful, please kindly cite as:
```bib
@InProceedings{Hogue2024,
    author    = {Hogue, Steven and Zhang, Chenxu and Daruger, Hamza and Tian, Yapeng and Guo, Xiaohu},
    title     = {DiffTED: One-shot Audio-driven TED Talk Video Generation with Diffusion-based Co-speech Gestures},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {1922-1931}
}
```

## Acknowledgement
- The codebase is developed based on [DiffGesture](https://github.com/Advocate99/DiffGesture) of Zhu et al.
