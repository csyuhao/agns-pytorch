'''
Auxillary functions
'''
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms


def load_img(file_path, img_size):
    ''' loading images
    Args:
        file_path (str): the path of loadding image
    Returns:
        Tensor: size (1, C, H, W)
    '''
    img = Image.open(file_path)
    trans = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    return trans(img).unsqueeze(0)


def wear_eyeglasses(eyeglasses, attacker_img, mask):
    ''' wearing glasses to attacker's face
    Args:
        eyeglasses (FloatTensor, B x 3 x 64 x 176): the eyeglasses
        attacker_img (FloatTensor, 1 x 3 x 224 x 224): the attacker images
        mask (FloatTensor, 1 x 3 x 224 x 224)： eyeglasses mask
    Returns:
        (FloatTensor)
    Testing code
        from module.utils.dataset import Crop

        eyeglasses_path = r'data/eyeglasses/glasses000002-1.png'
        eyeglasses = Image.open(eyeglasses_path)
        trans = transforms.Compose([
            Crop(25, 53, 176, 64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        eyeglasses = trans(eyeglasses).unsqueeze(0)

        attacker_img = load_img(r'data/digital/19.jpg', 224)
        mask = load_img(r'data/eyeglasses_mask_6percent.png', 224)

        worn_img = wear_eyeglasses(eyeglasses, attacker_img, mask)

        trans = transforms.Compose([
            transforms.ToPILImage()
        ])
        img = trans(worn_img[0])
        img.show()
    '''

    batch_size, _, h, w = eyeglasses.shape
    target_h, target_w = attacker_img.shape[2], attacker_img.shape[3]
    pad_diff_h = (target_h - h) // 2
    pad_diff_w = (target_w - w) // 2
    theta = torch.eye(3, dtype=torch.float32, device=eyeglasses.device).unsqueeze(0).repeat(batch_size, 1, 1)[:, :2, :]
    theta[:, 0, 2] = (pad_diff_w - 25) / 112.0
    theta[:, 1, 2] = (pad_diff_h - 53) / 112.0

    # expand attacker_img and mask
    normal_attacker_img = attacker_img.repeat(batch_size, 1, 1, 1)
    normal_mask = mask.repeat(batch_size, 1, 1, 1)

    # convert color bound
    converted_eyeglasses = (eyeglasses + 1.0) / 2.0
    padded_eyeglasses = F.pad(converted_eyeglasses, [pad_diff_w, pad_diff_w, pad_diff_h, pad_diff_h])
    grid = F.affine_grid(theta, size=normal_attacker_img.size(), align_corners=True)
    normal_eyeglasses = F.grid_sample(padded_eyeglasses, grid, mode='bilinear', align_corners=True)

    worn_img = normal_attacker_img.masked_fill(normal_mask != 0, 0.0) + normal_eyeglasses

    return worn_img


def calc_loss(model, input, target_class, img_size, mode):

    batch_size = input.shape[0]

    # resizing images
    resized_img = F.interpolate(input, img_size, mode='bilinear', align_corners=True)
    labels = torch.LongTensor([target_class] * batch_size).to(input.device)

    # forward
    logits = model.forward(resized_img)
    prob = F.softmax(logits, dim=1)
    if mode == 'dodge':
        c_loss = 1.0 - 2.0 * prob.gather(1, labels.view(batch_size, -1)).mean(0)
    elif mode == 'impersonate':
        c_loss = 2.0 * prob.gather(1, labels.view(batch_size, -1)).mean(0) - 1.0
    return c_loss, prob.gather(1, labels.view(batch_size, -1)).mean(0)
