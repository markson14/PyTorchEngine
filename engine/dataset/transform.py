from PIL import Image, ImageOps
import numpy as np
import torch
import torchvision.transforms.functional as F
import random

__all__ = ['ZeroPadding', 'VerticalFlip', 'RandomRotate90']


class ZeroPadding(object):

    def __call__(self, force_apply=False, **kwargs):
        pil_img = Image.fromarray(kwargs['image'])
        w, h = pil_img.size
        if w == h:
            return {'image': np.array(pil_img), 'label': kwargs['label']}
        max_edge = max(pil_img.size)
        pil_img = ImageOps.pad(pil_img, (max_edge, max_edge))
        return {'image': np.array(pil_img), 'label': kwargs['label']}


class VerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, force_apply=False, **kwargs):
        pil_img = Image.fromarray(kwargs['image'])
        label = kwargs['label']

        if torch.rand(1) < self.p:
            # for HGR only
            if label == 'like':
                label = 'dislike'
            elif label == 'dislike':
                label = 'like'
            else:
                pass
            return {"image": np.array(F.vflip(pil_img)), "label": label}

        return {"image": np.array(pil_img), "label": label}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotate90(object):
    """Randomly rotate the input by 90 degrees 1 or 3 times.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, force_apply=False, **kwargs):
        cv_img = kwargs['image']
        label = kwargs['label']

        if torch.rand(1) < self.p:
            # for HGR only
            if label == 'like':
                label = 'others'
            elif label == 'dislike':
                label = 'others'
            else:
                pass
            return {"image": np.ascontiguousarray(np.rot90(cv_img, random.choice((1, 3)))), "label": label}

        return {"image": cv_img, "label": label}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
