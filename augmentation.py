import torch, torchvision
import torchvision.transforms.functional as F

from PIL import Image, ImageOps


class Compose:
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxs):
        for t in self.transforms:
            image, bboxs = t(image, bboxs)
        return image, bboxs


class ToTensor:
    """
    Converts a PIL Image or numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    Only applied to image, not bboxes.
    """
    def __call__(self, image, bboxs):
        return F.to_tensor(image), bboxs
    
    
class Normalize(torch.nn.Module):
    """
    Normalize a tensor image with mean and standard deviation.
    Only applied to image, not bboxes.
    """
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, image, bboxs):
        return F.normalize(image, self.mean, self.std, self.inplace), bboxs
    
    
class Resize(torch.nn.Module):
    """
    Resize the short side of image to given size.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Both applied to image and bboxes.
    """
    def __init__(self, min_size, max_size):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size

    def forward(self, image, bboxs):
        return resize(image, bboxs, self.min_size, self.max_size)
    
    
class Flip(torch.nn.Module):
    """
    Apply horizontal flip on image and bboxes.
    Assume the coords are given min_x, min_y, max_x, max_y.
    Both applied to image and bboxes.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, image, bboxs):
        if torch.rand(1) < self.p:
            flip_image = ImageOps.mirror(image)
            if bboxs == None:
                return flip_image, bboxs
            else:
                flip_bbox = flip(image, bboxs)
                return flip_image, flip_bbox
        else:
            return image, bboxs
            
            
def resize(img, bboxs, min_size, max_size):
    w, h = img.size
    min_side, max_side = min(w, h), max(w, h)
    
    ratio = min(min_size / min_side, max_size / max_side)
    resize_w, resize_h = int(ratio * w), int(ratio * h)
    ratio_w, ratio_h = resize_w / w, resize_h / h
    
    resize_img = img.resize((resize_w, resize_h), resample=Image.BILINEAR)
    resize_bboxs = bboxs.clone()
    if bboxs != None:
        resize_bboxs[:, 0::2] = bboxs[:, 0::2] * ratio_w
        resize_bboxs[:, 1::2] = bboxs[:, 1::2] * ratio_h
    return resize_img, resize_bboxs
            
            
def flip(img, bboxs):
    img = F.pil_to_tensor(img)
    _, h, w = img.shape

    flip_bboxs = []
    for bbox in bboxs:
        min_x, min_y, max_x, max_y = bbox
        flip_min_x, flip_max_x = w-min_x, w-max_x
        flip_bboxs.append(torch.FloatTensor([flip_max_x, min_y, flip_min_x, max_y]))
    return torch.stack(flip_bboxs)