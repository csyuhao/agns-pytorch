'''
    Physical attack by using accessarios
'''
import os
import sys
import torch
import argparse
from PIL import Image
from module.discriminator import Discriminator
from module.generator import Generator
from module.utils.dataset import PhysicalDataset, PhysicalTestDataset, Crop
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import mean, optim
import torch.autograd as autograd
from torchvision.utils import save_image
from util import load_img, wear_eyeglasses_physical, calc_loss
from module.target import FaceNet, ArcFace, CosFace, VggFace


def main(args):

    # ===========================
    # Hyper parameters settings #
    # ===========================
    batch_size = args.batch_size
    epochs = args.epochs
    sample_interval = args.interval
    device = 'cpu' if args.no_cuda else 'cuda:0'
    lr = args.lr
    pretrained_epochs = args.pretrained_epochs
    classnum = 153
    target_model = None
    img_size = None
    mode = args.mode
    target = args.target
    kappa = args.kappa
    save_path = args.save_path
    dirname = '{}-{}-{}-{}'.format(args.attacker, args.target, args.target_model, args.mode)
    log_dir = os.path.join(r'logs', dirname)
    save_dir = os.path.join(save_path, dirname)
    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    # ======================
    # Summary Informations #
    # ======================
    print('# ===========================')
    print('# Summary Informations')
    print('# The attacked model [{}]'.format(args.target_model))
    print('# The target class [{}]'.format(args.target))
    print('# The attack mode [{}]'.format(args.mode))
    print('# ===========================')

    # ===========================
    # Preparing datasets        #
    # ===========================
    # loading eyeglasses dataset
    eyeglasses = args.eyeglasses
    eyeglasses_trans = transforms.Compose([
        Image.open,
        Crop(25, 53, 176, 64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # loading attacker's images dataset
    attacker = os.path.join(r'data\physical', args.attacker)
    attacker_trans = transforms.Compose([
        Image.open,
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # loading eyeglasses mask
    mask_img = load_img(args.mask, 224).to(device)

    # ===========================
    # Loading pretrained models #
    # ===========================
    # generator
    gen = Generator().to(device)
    pretrained_model = r'model\gen_{}.pt'.format(pretrained_epochs)
    gen.load_state_dict(torch.load(pretrained_model))
    optimizer_g = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))

    # discriminator
    disc = Discriminator().to(device)
    pretrained_model = r'model\disc_{}.pt'.format(pretrained_epochs)
    disc.load_state_dict(torch.load(pretrained_model))
    optimizer_d = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    # target model
    if args.target_model == 'FaceNet':
        target_model = FaceNet(device, classnum, r'model\finetuned_facenet.pt')
        img_size = (160, 160)
    elif args.target_model == 'CosFace':
        target_model = CosFace(device, classnum, r'model\finetuned_cosface.pt')
        img_size = (112, 96)
    elif args.target_model == 'ArcFace':
        target_model = ArcFace(device, classnum, r'model\finetuned_arcface.pt')
        img_size = (112, 112)
    elif args.target_model == 'VggFace':
        target_model = VggFace(device, classnum, r'model\finetuned_vggface.pt')
        img_size = (224, 224)
    else:
        raise Exception(
            'The target model [{}] is not defined!'.format(args.target_model))

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    dataset = PhysicalDataset([eyeglasses, attacker], [eyeglasses_trans, attacker_trans])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):

        for idx, batch in enumerate(loader):
            eyeglasses_img, attacker_img, matrix = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            batch_size = eyeglasses_img.shape[0]

            # adversarial ground truths
            valid = torch.ones((batch_size, 1), dtype=torch.float32).to(device) * 0.9
            fake = torch.zeros((batch_size, 1), dtype=torch.float32).to(device)

            # ==========================
            # train generator          #
            # ==========================
            for p in disc.parameters():
                p.requires_grad = False
            optimizer_g.zero_grad()
            noise = torch.FloatTensor(
                batch_size, 25).uniform_(-1.0, 1.0).to(device)
            z = autograd.Variable(noise.data, requires_grad=True)

            # discriminative loss
            fake_images = gen(z)
            g_loss = adversarial_loss(disc(fake_images), valid)
            grads_disc_loss = autograd.grad(g_loss, gen.parameters(), retain_graph=True)
            # attack loss
            worn_imgs = wear_eyeglasses_physical(fake_images, attacker_img, mask_img, matrix)
            clf_loss, prob, _ = calc_loss(
                target_model, worn_imgs, target, img_size, mode)
            grads_clf_loss = autograd.grad(-1.0 * clf_loss, gen.parameters(), retain_graph=False)
            # update generator parameters gradients
            for i, p in enumerate(gen.parameters()):
                grad_1 = grads_disc_loss[i]
                grad_2 = grads_clf_loss[i]
                if torch.norm(grad_1, p=2) > torch.norm(grad_2, p=2):
                    grad_1 = grad_1 * torch.norm(grad_2, p=2) / torch.norm(grad_1, p=2)
                else:
                    grad_2 = grad_2 * torch.norm(grad_1, p=2) / torch.norm(grad_2, p=2)
                p.grad = (kappa * grad_1 + (1.0 - kappa) * grad_2).clone()
            optimizer_g.step()

            # ==========================
            # train discriminator      #
            # ==========================
            for p in disc.parameters():
                p.requires_grad = True
            optimizer_d.zero_grad()
            fake_images = autograd.Variable(
                fake_images.data, requires_grad=True)
            real_loss = adversarial_loss(disc(eyeglasses_img), valid)
            fake_loss = adversarial_loss(disc(fake_images), fake)
            d_loss = (fake_loss + real_loss) / 2.0
            d_loss.backward()
            optimizer_d.step()
            if idx % 50 == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [Prob: %f] [Disc: %f]"
                      % (epoch, epochs, idx, len(loader), d_loss.item(), prob.item(), g_loss.item()))

            batches_done = epoch * len(loader) + idx
            if batches_done % sample_interval == 0:
                save_image(worn_imgs.data[:25], "%s/%d.png" %
                           (log_dir, batches_done), nrow=5, normalize=False)

    torch.save(gen.state_dict(), r'{}\gen_{}.pt'.format(save_dir, epochs))
    torch.save(disc.state_dict(), r'{}\disc_{}.pt'.format(save_dir, epochs))

    gen.load_state_dict(torch.load(r'{}\gen_{}.pt'.format(save_dir, epochs)))
    disc.load_state_dict(torch.load(r'{}\disc_{}.pt'.format(save_dir, epochs)))

    cropped_eyeglasses_mask = r'data/cropped_eyeglasses_mask.png'
    cropped_mask = transforms.Compose([
        Image.open,
        transforms.ToTensor(),
    ])
    eyeglasses_mask = cropped_mask(cropped_eyeglasses_mask).unsqueeze(0).to(device)

    dataset = PhysicalTestDataset(attacker, attacker_trans)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    loss, prob, success_rate = 0.0, 0.0, 0.0
    if args.mode == 'dodge':
        prob = 1.0
    else:
        prob = 0.0
    save_noise = None
    with torch.no_grad():
        for idx in range(32):
            for batch in loader:
                attacker_img, matrix = batch[0].to(device), batch[1].to(device)
                attacker_img, matrix = attacker_img[:20], matrix[:20]
                noise = torch.FloatTensor(1, 25).uniform_(-1.0, 1.0).to(device)
                noise = noise.repeat(matrix.shape[0], 1)
                z = autograd.Variable(noise.data, requires_grad=True)

                # discriminative loss
                fake_images = gen(z)
                # attack loss
                worn_imgs = wear_eyeglasses_physical(fake_images, attacker_img, mask_img, matrix)
                clf_loss, p, rate = calc_loss(
                    target_model, worn_imgs, target, img_size, mode)

                mean_loss = clf_loss.detach().cpu().mean()
                mean_prob = p.detach().cpu().mean()
                if args.mode == 'dodge':
                    success_rate = 1 - rate
                    if mean_prob < prob:
                        prob = mean_prob
                        loss = mean_loss
                        save_noise = noise
                else:
                    success_rate = rate
                    if mean_prob > prob:
                        prob = mean_prob
                        loss = mean_loss
                        save_noise = noise

    trans = transforms.Compose([
        transforms.ToPILImage()
    ])
    fake_images = gen(save_noise)

    eyeglasses_mask = eyeglasses_mask.repeat(fake_images.shape[0], 1, 1, 1)
    fake_images = fake_images.masked_fill(eyeglasses_mask.mean(dim=1, keepdims=True) == 0.0, 1.0)
    fake_images = (fake_images + 1.0) / 2.0
    fake_image = trans(fake_images[0].cpu()).resize((530, 193), resample=0)
    fake_image.save(os.path.join(save_dir, 'fake_image.png'), 'PNG')
    return loss, prob, success_rate


def parse(argv):
    parser = argparse.ArgumentParser('Digital attack by using accessarios.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--no_cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--interval', type=int, default=50,
                        help='how many batches to wait before printing')
    parser.add_argument('--mode', type=str, default='dodge',
                        help='mode (dodge or impersonate) of attacking face recognition system (default: dodge)')
    parser.add_argument('--target', type=int, default=0,
                        help='face-id of choosen victim (default: 0)')
    parser.add_argument('--target_model', type=str, default='FaceNet',
                        help='attacked face recognition model (FaceNet, CosFace, ArcFace, default: FaceNet)')

    # dataset setting
    parser.add_argument('--eyeglasses', type=str,
                        default=r'data\eyeglasses', help='training eyeglasses dataset')
    parser.add_argument(
        '--mask', type=str, default=r'data\eyeglasses_mask_6percent.png', help='path of eyeglasses mask')
    parser.add_argument('--attacker', type=str,
                        default=r'data\physical', help='the picture of attacker')

    # params setting
    parser.add_argument('--kappa', type=float, default=0.25,
                        help='weight of generator\'s loss function')

    # pretrained model
    parser.add_argument('--pretrained_epochs', type=int,
                        default=200, help='number of epochs of trained model')
    parser.add_argument('--save_path', type=str, default='save', help='path to save trained model')

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse(sys.argv[1:])

    '''
    Attacker Victim1 Victim2 # imperosonating
    Attacker # dodging
    '''
    attacker_info = {}
    with open('pair.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            names = line.split(' ')

            # storing attackers' information
            attacker = names[0]
            if attacker in attacker_info:
                raise Exception('The attacker is repetitive.')
            attacker_info[attacker] = []

            # storing victims' information
            for name in names[1:]:
                attacker_info[attacker].append(int(name))

    for attacker in attacker_info:
        args.attacker = attacker
        for victim_id in attacker_info[attacker]:
            args.target = victim_id
            if len(attacker_info[attacker]) == 1:
                args.mode = 'dodge'
            else:
                args.mode = 'impersonate'
            for target_model in ['VggFace']:
                args.target_model = target_model
                args.batch_size = 32

                loss, prob, success_rate = main(args)

                with open('result.txt', 'a+') as f:
                    f.write('Attacker: {}, Target: {}, Attacked Model: {}, Attack Mode: {}, Prob: {}, Loss: {}, Success Rate: {}\n'.format(args.attacker, args.target, args.target_model, args.mode, prob, loss, success_rate))
