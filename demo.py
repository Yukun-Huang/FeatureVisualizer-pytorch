import torch
from feature_visualizer import FeatureVisualizer

## Load Data
img = torch.load('data/image.pt')   # torch.Size([4, 3,   256, 128])
f = torch.load('data/feature.pt')   # torch.Size([4, 2048,  8,   4])

## Feature visualizer
V = FeatureVisualizer(
    cmap_type='jet',
    reduce_type='mean',
    upsample_size=(256, 128),
    upsample_type='bilinear',
)

## Visualize Feature
V.save_feature(f, save_path='demo/feature.jpg')

## Visualize Image
""" If image data are processed by transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
you should set recover=True. """
V.save_image(img, save_path='demo/image.jpg', recover=False)

## Visualize both Image and Feature
V.save_both(img, f, 'demo/demo.jpg', recover=False)

