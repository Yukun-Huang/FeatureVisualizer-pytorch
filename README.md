# FeatureVisualizer-pytorch
An implementation of feature visualizer for pytorch tensor.

## Prerequisites

- python >= 3.6

- pytorch

- matplotlib

- numpy

## Usage
```
# Initialization
FV = FeatureVisualizer()

# Load data
image = load_image(image_path)
feat = net(image)

# Visualize features
FV.viz(feat, save_path=feat_path)

# Mix images and features
FV.mix(image_path, feat_path, save_path=mix_path)
```

## Demo

```
python demo.py
```
<img src="https://github.com/hyk1996/FeatureVisualizer-pytorch/blob/master/demo/demo.jpg" height="210" width="750">
