import PIL
import torch
import numpy as np
from PIL import Image

class Transformer(object):
    """"Transform"""
    def __init__(self,):
        pass
    def __call__(self, imgA, imgB=None):
        pass

class Compose(Transformer):
    """Compose transforms"""
    def __init__(self, transforms=[]):
        super().__init__()
        self.transforms=transforms
        
    def __call__(self, imgA, imgB=None):
        if imgB is None:
            for transform in self.transforms:
                imgA = transform(imgA, imgB)
            return imgA
        for transform in self.transforms:
            imgA, imgB = transform(imgA, imgB)
        return imgA, imgB
    
class Resize(Transformer):
    """Resize imageA and imageB"""
    def __init__(self, size=(256, 256)):
        """
        :param: size (default: tuple=(256, 256)) - target size
        """
        super().__init__()
        self.size=size
        
    def __call__(self, imgA, imgB=None):
        imgA = imgA.resize(self.size)
        if imgB is None:
            return imgA
        imgB = imgB.resize(self.size)
        return imgA, imgB
    
class CenterCrop(Transformer):
    """CenterCrop imageA and imageB"""
    def __init__(self, size=(256, 256), p=0.5):
        """
        :param: size (default: tuple=(256, 256)) - target size
        """
        super().__init__()
        self.size=size
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        W, H = imgA.size
        cW, cH = W//2, H//2
        fW, fH = self.size[0]//2, self.size[1]//2
        if np.random.uniform() < self.p:
            imgA = imgA.crop((cW-fW, cH-fH, cW+fW, cH+fH))
            if imgB is not None:
                imgB = imgB.crop((cW-fW, cH-fH, cW+fW, cH+fH))
        else:
            imgA = imgA.resize(self.size)
            if imgB is not None:
                imgB = imgB.resize(self.size)
        if imgB is None:
            return imgA
        return imgA, imgB
    
class Rotate(Transformer):
    """Rotate imageA and imageB"""
    def __init__(self, p=0.5):
        """
        :param: p (default: float=0.5) - probability of rotation
        """
        super().__init__()
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        if np.random.uniform() < self.p:
            imgA = imgA.rotate(180)
            if imgB is not None:
                imgB = imgB.rotate(180)
        if imgB is None:
            return imgA
        return imgA, imgB
    
class HorizontalFlip(Transformer):
    """Horizontal flip of imageA and imageB"""
    def __init__(self, p=0.5):
        """
        :param: p (default: float=0.5) - probability of horizontal flip
        """
        super().__init__()
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        if np.random.uniform() < self.p:
            imgA = imgA.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            if imgB is not None:
                imgB = imgB.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        if imgB is None:
            return imgA
        return imgA, imgB
    
class VerticalFlip(Transformer):
    """Vertical flip of imageA and imageB"""
    def __init__(self, p=0.5):
        """
        :param: p (default: float=0.5) - probability of vertical flip
        """
        super().__init__()
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        if np.random.uniform() < self.p:
            imgA = imgA.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            if imgB is not None:
                imgB = imgB.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        if imgB is None:
            return imgA
        return imgA, imgB
    
class ToTensor(Transformer):
    """Convert imageA and imageB to torch.tensor"""
    def __init__(self,):
        super().__init__()
    
    def __call__(self, imgA, imgB=None):
        imgA = np.array(imgA)/255.
        imgA = torch.from_numpy(imgA).float().permute(2, 0, 1)
        if imgB is None:
            return imgA
        imgB = np.array(imgB)/255.
        imgB = torch.from_numpy(imgB).float().permute(2, 0, 1)
        return imgA, imgB
    
class ToImage(Transformer):
    """Convert imageA and imageB tensors to PIL.Image"""
    def __init__(self,):
        super().__init__()
    
    def __call__(self, imgA, imgB=None):
        imgA = imgA.permute(1,2,0).numpy()
        imgA = Image.fromarray(np.uint8(imgA*255))
        if imgB is None:
            return imgA
        imgB = imgB.permute(1,2,0).numpy()
        imgB = Image.fromarray(np.uint8(imgB*255))
        return imgA, imgB
    
class Normalize(Transformer):
    """Normalize imageA and imageB"""
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        """
        :param: mean (default: list=[0.485, 0.456, 0.406]) - list of means for each image channel 
        :param: std (default: list=[0.229, 0.224, 0.225]) - list of stds for each image channel
        """
        super().__init__()
        self.mean=mean
        self.std=std
        
    def __call__(self, imgA, imgB=None):
        
        if (self.mean is not None) and (self.std is not None):
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            imgA = (imgA - mean)/std
            if imgB is not None:
                imgB = (imgB - mean)/std
            
        if imgB is None:
            return imgA
        return imgA, imgB
    
class DeNormalize(Transformer):
    """DeNormalize imageA and imageB"""
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        """
        :param: mean (default: list=[0.485, 0.456, 0.406]) - list of means for each image channel 
        :param: std (default: list=[0.229, 0.224, 0.225]) - list of stds for each image channel
        """
        super().__init__()
        self.mean=mean
        self.std=std
        
    def __call__(self, imgA, imgB=None):
        
        if (self.mean is not None) and (self.std is not None):
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            imgA = imgA*std + mean
            imgA = torch.clip(imgA, 0., 1.)
            if imgB is not None:
                imgB = imgB*std + mean
                imgB = torch.clip(imgB, 0., 1.)
            
        if imgB is None:
            return imgA
        return imgA, imgB

    
__all__ = ['Transformer', 'Compose', 'Resize', 'CenterCrop', 'Rotate', 'HorizontalFlip',
           'VerticalFlip', 'ToTensor', 'ToImage', 'Normalize', 'DeNormalize',]