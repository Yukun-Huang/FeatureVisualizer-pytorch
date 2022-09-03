import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


class FeatureVisualizer:
    def __init__(self, norm='spatial', reduce='mean', interpolate='bilinear'):
        self.norm = norm
        self.reduce = reduce
        self.interpolate = interpolate

    def _reduce(self, x):
        x = x.squeeze()
        if self.reduce == 'mean':
            # reduce by channel, [C, H, W] -> [H, W]
            x = x.mean(dim=0, keepdim=False)
        else:
            raise NotImplementedError
        return x

    def _interpolate(self, x, output_size):
        assert x.dim() == 2, f'Input shape should be [H, W]!'
        x = x.view(1, 1, *x.shape)
        x = F.interpolate(x, size=output_size, mode=self.interpolate, align_corners=False)
        return x.squeeze()

    def _normalize(self, x):
        assert x.dim() == 2, f'Input shape should be [H, W]!'
        if self.norm == 'minmax':
            max_val, min_val = torch.max(x), torch.min(x)
            x = (x - min_val) / (max_val - min_val)
        elif self.norm == 'spatial':
            x = x.div(torch.norm(x.flatten(), p=2, dim=0))
        elif self.norm == 'abs':
            x = torch.abs(x)
        return x

    def viz(self, f, size=None, save_path=None, cmap=None, show=False):
        # Check dimension
        assert f.dim() == 3 or (f.dim() == 4 and f.size(0) == 1),\
            f'Input shape should be [C, H, W] or [1, C, H, W], not {f.shape}!'
        # Process
        with torch.no_grad():
            f = self._reduce(f)
            if size is not None:
                f = self._interpolate(f, size)
            f = self._normalize(f).detach().cpu().numpy()
        # Draw
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.gca().imshow(f, cmap=plt.get_cmap(cmap))
        # Save
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        if show:
            plt.show()
        else:
            plt.close()
        return f

    @staticmethod
    def mix(image_path, feat_path, save_path, alpha=0.7, beta=None):
        if beta is None:
            beta = 1 - alpha
        img1 = Image.open(image_path).convert('RGB')
        img2 = Image.open(feat_path).convert('RGB').resize(img1.size)
        img_mix = np.array(img1) * alpha + np.array(img2) * beta
        Image.fromarray(img_mix.clip(0.0, 255.0).astype(np.uint8)).save(save_path)
