'''
    Using this script to pre-train generator and discriminator
'''
from module.discriminator import Discriminator
from module.generator import Generator
from module.utils.dataset import EyeGlasses, Crop
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import torch
import torch.autograd as autograd
from torchvision.utils import save_image


if __name__ == '__main__':

    # params
    root_dir = r'data\eyeglasses'
    trans = transforms.Compose([
        Crop(25, 53, 176, 64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    batch_size = 32
    epochs = 30
    sample_interval = 50

    # generator
    gen = Generator().cuda()
    optimizer_g = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # discriminator
    disc = Discriminator().cuda()
    optimizer_d = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    dataset = EyeGlasses(root_dir, trans)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):

        for idx, batch in enumerate(loader):
            batch_size = batch.shape[0]
            batch = batch.cuda()

            # adversarial ground truths
            valid = torch.ones((batch_size, 1), dtype=torch.float32).cuda()
            valid_label = autograd.Variable(valid.data, requires_grad=False)
            fake = torch.zeros((batch_size, 1), dtype=torch.float32).cuda()
            fake_label = autograd.Variable(fake.data, requires_grad=False)

            # ==========================
            # train generator          #
            # ==========================
            for p in disc.parameters():
                p.requires_grad = False
            optimizer_g.zero_grad()
            noise = torch.FloatTensor(batch_size, 25).uniform_(-1.0, 1.0).cuda()
            z = autograd.Variable(noise.data, requires_grad=True)

            fake_images = gen(z)
            g_loss = adversarial_loss(disc(fake_images), valid)
            g_loss.backward()
            optimizer_g.step()

            # ==========================
            # train discriminator      #
            # ==========================
            for p in disc.parameters():
                p.requires_grad = True
            optimizer_d.zero_grad()
            fake_images = autograd.Variable(fake_images.data, requires_grad=True)
            real_loss = adversarial_loss(disc(batch), valid)
            fake_loss = adversarial_loss(disc(fake_images), fake)
            d_loss = (fake_loss + real_loss) / 2.0
            d_loss.backward()
            optimizer_d.step()
            if idx % 50 == 0:
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 30, idx, len(loader), d_loss.item(), g_loss.item()))

            batches_done = epoch * len(loader) + idx
            if batches_done % sample_interval == 0:
                save_image(fake_images.data[:25], "logs/%d.png" % batches_done, nrow=5, normalize=True)

    torch.save(gen.state_dict(), r'model\gen_{}.pt'.format(epochs))
    torch.save(disc.state_dict(), r'model\disc_{}.pt'.format(epochs))
