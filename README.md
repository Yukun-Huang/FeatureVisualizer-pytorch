# FeatureVisualizer-pytorch
An implementation of feature visualizer for pytorch tensor, based on Matplotlib.

## Usage

```
from feature_visualizer import FeatureVisualizer

V = FeatureVisualizer()
```

Visualize feature:
```
# f is torch.tensor with shape=(8, 2048, 9, 5)
# img is torch.tensor with shape=(8, 3, 288, 144)

V.show(f)
V.save(f, 'feature.jpg')
```

Visualize both image and feature:
```
V.show_both(img, f)
V.save_both(img, f, 'demo.jpg')
```

Check **demo.py** for more details.

## Visualization
![demo](https://github.com/hyk1996/FeatureVisualizer-pytorch/blob/master/demo/demo.jpg)
