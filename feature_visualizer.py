import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np


class FeatureVisualizer:

    def __init__(self, upsample_size=(256, 256), cmap_type='jet', reduce_type='mean', upsample_type='bilinear'):
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.color_map = plt.get_cmap(cmap_type)
        self.reduce_type = reduce_type
        self.upsample_size = upsample_size
        self.upsample_type = upsample_type
        self.height = upsample_size[0]
        self.width = upsample_size[1]

    def _check(self, x):
        ## Assert
        assert len(x.shape) == 3 or len(x.shape) == 4, \
            'Input should be 3D or 4D tensor.'
        ## Reduce by batch, choose the first sample
        if len(x.shape) == 4:
            x = x[0,:,:,:]
        return x.unsqueeze(dim=0)  # shape=[1, C, H, W]

    def _norm(self, x):
        max_val, min_val = torch.max(x), torch.min(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def _recover(self, inp):
        inp = self.std * inp + self.mean
        inp = inp * 255.0
        inp = np.clip(inp, 0, 255)
        inp = inp.astype(np.uint8)
        return inp

    def _transform(self, f):
        f = self._check(f)
        f = self._norm(f)  # normalize to [0, 1]
        if self.reduce_type == 'mean':
            f = torch.mean(f, dim=1, keepdim=True)  # reduce by channel
            f = F.interpolate(f, size=self.upsample_size, mode=self.upsample_type, align_corners=False)
        else:
            raise NotImplementedError
        return f.squeeze().detach().cpu().numpy()

    def _transform_rgb(self, x, norm):
        x = self._check(x)
        x = F.interpolate(x, size=self.upsample_size, mode=self.upsample_type, align_corners=False)
        x = x.squeeze(dim=0).detach().cpu().numpy().transpose((1, 2, 0))
        if norm:
            x = self._recover(x)
        return x

    def _draw(self, x, ca):
        # set x and y axis invisible
        ca.axes.get_xaxis().set_visible(False)
        ca.axes.get_yaxis().set_visible(False)
        # draw
        ca.imshow(x, cmap=self.color_map)

    def show(self, f):
        f = self._transform(f)
        self._draw(f, plt.gca())
        plt.show()
        plt.close()

    def save(self, f, save_path):
        f = self._transform(f)
        self._draw(f, plt.gca())
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()

    def show_both(self, img, f, norm=False):
        fig = plt.figure()
        ax0 = fig.add_subplot(121, title="image")
        ax1 = fig.add_subplot(122, title="feature")
        self._draw(self._transform_rgb(img, norm), ax0)
        self._draw(self._transform(f), ax1)
        plt.show()
        plt.close()

    def save_both(self, img, f, save_path, norm=False):
        fig = plt.figure()
        ax0 = fig.add_subplot(121, title="image")
        ax1 = fig.add_subplot(122, title="feature")
        self._draw(self._transform_rgb(img, norm), ax0)
        self._draw(self._transform(f), ax1)
        plt.savefig(save_path)
        plt.close()

