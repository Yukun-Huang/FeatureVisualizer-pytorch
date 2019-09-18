import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
from torchvision.utils import save_image


class FeatureVisualizer:

    def __init__(self, upsample_size=(256, 256), cmap_type='jet', reduce_type='mean', upsample_type='bilinear'):

        self.color_map = plt.get_cmap(cmap_type)
        self.reduce_type = reduce_type
        self.upsample_size = upsample_size
        self.upsample_type = upsample_type
        self.height = upsample_size[0]
        self.width = upsample_size[1]

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.t_mean = torch.Tensor([[[[0.485]], [[0.456]], [[0.406]]]])
        self.t_std = torch.Tensor([[[[0.229]], [[0.224]], [[0.225]]]])

    def __check_dimention(self, x):
        assert x.dim() == 3 or x.dim() == 4, 'Input should be 3D or 4D tensor.'

    def __recover_numpy(self, x):
        x = self.std * x + self.mean
        x = x * 255.0
        x = np.clip(x, 0, 255)
        x = x.astype(np.uint8)
        return x

    def __recover_torch(self, x):
        if x.is_cuda:
            x = x.detach().cpu()
        return x * self.t_std + self.t_mean

    def __normalize(self, x):
        max_val, min_val = torch.max(x), torch.min(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def __transform_image(self, x, recover):
        """
        Transform image from torch.tensor to numpy.array.
        """
        # reduce by batch, choose the first sample
        x = x[0].unsqueeze(dim=0) if x.dim() == 4 else x.unsqueeze(dim=0)
        x = F.interpolate(x, size=self.upsample_size, mode=self.upsample_type, align_corners=False)
        x = x.squeeze(dim=0).detach().cpu().numpy().transpose((1, 2, 0))
        return self.__recover_numpy(x) if recover else x

    def __transform_feature(self, f):
        """
        Transform feature from torch.tensor to numpy.array.
        """
        # reduce by batch, choose the first sample
        f = f[0].unsqueeze(dim=0) if f.dim() == 4 else f.unsqueeze(dim=0)
        # normalize to [0, 1]
        f = self.__normalize(f)
        if self.reduce_type == 'mean':
            f = f.mean(dim=1, keepdim=True)  # reduce by channel
            f = F.interpolate(f, size=self.upsample_size, mode=self.upsample_type, align_corners=False)
        else:
            raise NotImplementedError
        return f.squeeze().detach().cpu().numpy()

    def __draw(self, x, ca):
        # set x and y axis invisible
        ca.axes.get_xaxis().set_visible(False)
        ca.axes.get_yaxis().set_visible(False)
        # draw
        ca.imshow(x, cmap=self.color_map)

    def save_feature(self, f, save_path, show=False):
        self.__check_dimention(f)
        f = self.__transform_feature(f.cpu())
        self.__draw(f, plt.gca())
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        if show:
            plt.show()
        plt.close()

    def save_image(self, imgs, save_path, nrow=8, recover=False):
        self.__check_dimention(imgs)
        if recover:
            imgs = self.__recover_torch(imgs)
        save_image(imgs, save_path, nrow=nrow)

    def save_both(self, img, f, save_path, recover=False, show=False):
        fig = plt.figure()
        ax0 = fig.add_subplot(121, title="image")
        ax1 = fig.add_subplot(122, title="feature")
        self.__draw(self.__transform_image(img, recover), ax0)
        self.__draw(self.__transform_feature(f), ax1)
        plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()


