import os
import torch
import numbers
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat


class EyeGlasses(Dataset):
    ''' Loading EyeGlasses
    Test code:
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
    '''

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


class PhysicalDataset(Dataset):
    ''' The dataset is used to attack physically, including eyeglasses and attacker's images
    '''
    def __init__(self, dataset_dirs, trans=[None, None]):
        '''
        Args:
            dataset_dirs: eyeglasses, and attacker's images
            trans: eyeglasses transformations and attacker's images transformations
        Test Code:
            from torchvision import transforms
            from torch.utils.data import Dataset, DataLoader

            eyeglasses_trans = transforms.Compose([
                Image.open,
                Crop(25, 53, 176, 64),
                transforms.ToTensor()
            ])
            attacker_trans = transforms.Compose([
                Image.open,
                transforms.ToTensor()
            ])

            dataset = PhysicalDataset(['../../data/eyeglasses', '../../data/physical'], [eyeglasses_trans, attacker_trans])
            loader = DataLoader(dataset, batch_size=2, shuffle=True)
            for batch in loader:
                print(batch[0].shape, batch[1].shape, batch[2].shape)
                exit(0)
        '''
        assert len(dataset_dirs) == 2 and len(trans) == 2

        super(PhysicalDataset, self).__init__()
        eyeglasses_imgs, attacker_imgs = None, None
        dirs = []
        for dirname in dataset_dirs:
            dirs += [os.listdir(dirname)]
        eyeglasses_imgs, attacker_imgs = dirs

        # dataset
        self.eyeglasses_imgs = [os.path.join(dataset_dirs[0], img) for img in eyeglasses_imgs]
        self.attacker_imgs = [os.path.join(dataset_dirs[1], img) for img in attacker_imgs if img.endswith('.jpg') or img.endswith('.png')]

        # transform
        self.eyeglasses_trans = trans[0]
        self.attacker_trans = trans[1]

    def __getitem__(self, index):
        short_len = len(self.attacker_imgs)
        eyeglasses, attacker = None, None
        idx = index % short_len

        # Eyeglasses Images
        eyeglasses = self.eyeglasses_imgs[index]
        if self.eyeglasses_trans:
            eyeglasses = self.eyeglasses_trans(eyeglasses)

        # Attackers' Images
        attacker = self.attacker_imgs[idx]
        attacker_path = attacker
        if self.attacker_trans:
            attacker = self.attacker_trans(attacker)

        # Transformer Parameters
        attacker_param_path = attacker_path.replace('.jpg', '.mat').replace('.png', '.mat')
        attacker_param = loadmat(attacker_param_path)['matrix'].astype(np.float32)
        return eyeglasses, attacker, attacker_param

    def __len__(self):
        return len(self.eyeglasses_imgs)
