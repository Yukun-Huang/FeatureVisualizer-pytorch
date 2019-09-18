# FeatureVisualizer-pytorch
An implementation of feature visualizer for pytorch tensor, based on Matplotlib.

## Prerequisites

- NumPy

- Matplotlib

- PyTorch 0.4.1+

## Usage

### Preparation
```
from feature_visualizer import FeatureVisualizer

V = FeatureVisualizer()

img = torch.load('data/image.pt')   # torch.Size([4, 3,   256, 128])
f = torch.load('data/feature.pt')   # torch.Size([4, 2048,  8,   4])
```

### Visualize Feature
```
V.save_feature(f, save_path='demo/feature.jpg')
```
<img src="https://github.com/hyk1996/FeatureVisualizer-pytorch/blob/master/demo/feature.jpg" height="200" width="100">

### Visualize Image
```
V.save_image(img, save_path='demo/image.jpg')
```
<img src="https://github.com/hyk1996/FeatureVisualizer-pytorch/blob/master/demo/image.jpg" height="200" width="400">

### Visualize both image and feature
```
V.save_both(img, f, 'demo/demo.jpg')
```
<img src="https://github.com/hyk1996/FeatureVisualizer-pytorch/blob/master/demo/demo.jpg" height="200" width="300">

Check **demo.py** for more details.

