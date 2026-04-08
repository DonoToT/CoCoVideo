# CoCoVideo

**CoCoVideo: The High-Quality Commercial-Model-Based Contrastive Benchmark for AI-Generated Video Detection**

<p align="center">
  <img src="figs/teaser.png" width="90%">
</p>



## Abstract

With the rapid advancement of artificial intelligence generated content (AIGC) technologies, video forgery has become increasingly prevalent, posing new challenges to public discourse and societal security. Despite remarkable progress in existing deepfake detection methods, AIGC forgery detection remains challenging, as existing datasets mainly rely on open-source video generation models with quality far below that of commercial AIGC systems.
Even datasets containing a few commercial samples often retain visible watermarks, compromising authenticity and hindering model generalization to high-fidelity AIGC videos.
To address these issues, we introduce **CoCoVideo-26K**, a contrastive, commercial-model-based AIGC video dataset covering 13 mainstream commercial generators and providing semantically aligned real–fake video pairs.
This dataset enables deeper exploration of the differences between authentic and high-quality synthetic videos and establishes a new benchmark for highly realistic video forgery detection.
Building on this dataset, we propose **CoCoDetect**, a detection framework integrating contrastive learning with confidence-gated multimodal large language model (MLLM) inference.
An R3D-18 backbone extracts spatio-temporal representations, while a confidence gate routes uncertain cases to an MLLM for reasoning about physical plausibility and scene consistency.
Extensive experiments on CoCoVideo-26K and public benchmarks demonstrate state-of-the-art performance, validating the framework’s robustness and generalizability.
Our code and dataset are available at https://github.com/DonoToT/CoCoVideo.



## Quick Start

### 1. Environment Setup

Clone the repository:
```bash
git clone https://github.com/DonoToT/CoCoVideo.git
cd CoCoVideo
```

Python Version: 
We use Python 3.11 for this project.

Create and activate Conda environment:
```python
conda create -n cocovideo
conda activate cocovideo
pip install -r requirements.txt
```

### 2. Download Pretrained Model
Download pretrained model from: [[Google Drive]](https://drive.google.com/file/d/1RZ7EYWrdtWQp5YJzEXTgIkNm6f8hSre5/view).

### 3. Download Dataset
*(Coming soon)*



## Training

### 1. Prepare Dataset Directory
Create a `dataset` folder in your project root and place the downloaded video dataset inside it.

### 2. Extract Video Frames
Extract a continuous sequence of video frames starting from the **30th frame** of each video, depending on your training requirements. Save these extracted frames into a `dataset_frames` directory matching the following structure:

```text
dataset_frames/
├── generated_videos/
│   ├── 1_jimeng/
│   ├── ...
│   ├── 10_vidu/
│   │   └── 0gqxDPO3I9g_3_0to156/
│   │       ├── 0000.jpg
│   │       ├── 0001.jpg
│   │       ├── 0002.jpg
│   │       ├── 0003.jpg
│   │       └── 0004.jpg
│   └── ...
└── original_videos/
    ├── 1/
    ├── ...
    ├── 10/
    │   └── 0gqxDPO3I9g_3_0to156/
    │       ├── 0000.jpg
    │       ├── 0001.jpg
    │       ├── 0002.jpg
    │       ├── 0003.jpg
    │       └── 0004.jpg
    └── ...
```
*(`0000.jpg` here corresponds to the 30th frame of the original video in our method. Name the extracted frame sequence consecutively.)*

### 3. Run Training
Once your data is successfully prepared, you can start the training process by running:
```bash
python train.py
```



## Inference

*(Coming soon)*


## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{feng2026cocovideo,
  title={{CoCoVideo: The High-Quality Commercial-Model-Based Contrastive Benchmark for AI-Generated Video Detection}},
  author={Feng, Huidong and Chen, Wentao and Chen, Jie and Cai, Xinqi and Ma, Ruolong and Zheng, Yinglin and Lin, Yuxin and Zeng, Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```