import torch
from feature_visualizer import FeatureVisualizer
from torchvision.utils import save_image

## Load Data
img = torch.load('data/image.pt')   # torch.Size([8, 3, 288, 144])
f = torch.load('data/feature.pt')   # torch.Size([8, 2048, 9, 5])

## Feature visualizer
V = FeatureVisualizer(
    cmap_type='jet',
    reduce_type='mean',
    upsample_size=(288, 144),
    upsample_type='bilinear',
)

'''
## Visualize Feature
FeatureVisualizer.show() and FeatureVisualizer.save() are not support to visualize RGB images.
If you want to show rgb images, use 'save_image()' in torchvision.
'''
V.show(f)
V.save(f, 'demo/feature.jpg')

save_image(img[0,:,:,:], 'demo/image.jpg')


'''
## Visualize both Image and Feature
If image data are processed by transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
you should set norm=True.
'''
V.show_both(img, f, norm=False)
V.save_both(img, f, 'demo/demo.jpg', norm=False)

