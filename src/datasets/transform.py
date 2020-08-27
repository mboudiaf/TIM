import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


import torch
from PIL import ImageEnhance

transformtypedict = dict(Brightness=ImageEnhance.Brightness,
                         Contrast=ImageEnhance.Contrast,
                         Sharpness=ImageEnhance.Sharpness,
                         Color=ImageEnhance.Color)


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


def without_augment(size=84, enlarge=False):
    if enlarge:
        resize = int(size*256./224.)
    else:
        resize = size
    return transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ])


def with_augment(size=84, disable_random_resize=False, jitter=False):
    if disable_random_resize:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        if jitter:
            return transforms.Compose([
                transforms.RandomResizedCrop(size),
                ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])