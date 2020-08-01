import os
import torch
import numbers
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class EyeGlasses(Dataset):

    def __init__(self, root_dir, trans=None):
        super(EyeGlasses, self).__init__()
        imgs = os.listdir(root_dir)
        self.imgs = [os.path.join(root_dir, img) for img in imgs]
        self.trans = trans

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.trans:
            return self.trans(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            return torch.from_numpy(pil_img)

    def __len__(self):
        return len(self.imgs)


class Crop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, left, top, width, height):
        """Crop the given PIL Image.

        Args:
            top (int): Vertical component of the top left corner of the crop box.
            left (int): Horizontal component of the top left corner of the crop box.
            height (int): Height of the crop box.
            width (int): Width of the crop box.

        Returns:
            PIL Image: Cropped image.
        """
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return img.crop((self.left, self.top, self.left + self.width, self.top + self.height))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


if __name__ == "__main__":

    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader

    trans = transforms.Compose([
        Crop(25, 53, 176, 64),
        transforms.ToTensor()
    ])

    dataset = EyeGlasses('../../data/eyeglasses', trans)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in loader:
        print(batch.shape)
        exit(0)
