import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
2022-10-29 20:33:16,437 [_load_config()] [DEBUG]  User refresh rate: 2.0
2022-10-29 20:33:16,437 [_load_config()] [DEBUG]  UTF8 selected as False
2022-10-29 20:33:16,437 [_load_config()] [DEBUG]  No user config for temp threshold
2022-10-29 20:33:16,439 [__init__()] [DEBUG]  Power reading is not available
2022-10-29 20:33:16,439 [__init__()] [INFO ]  num cpus 8
2022-10-29 20:33:16,448 [on_unicode_checkbox()] [DEBUG]  unicode State is False
2022-10-29 20:33:16,450 [main_window()] [DEBUG]  Pile index: 15
2022-10-29 20:33:16,452 [update()] [INFO ]  Core id util 16.7
2022-10-29 20:33:16,452 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:16,452 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:16,452 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:16,452 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:16,452 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:16,452 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:16,452 [update()] [INFO ]  Core id util 100.0
2022-10-29 20:33:16,452 [update()] [INFO ]  Utilization recorded [12.8, 16.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
2022-10-29 20:33:16,452 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:16,452 [get_top()] [DEBUG]  Returning top 0.0
2022-10-29 20:33:16,453 [update()] [INFO ]  Reading [12.8, 16.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
2022-10-29 20:33:18,454 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:18,455 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:18,455 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:18,455 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:18,455 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:18,455 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:18,455 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:18,455 [update()] [INFO ]  Core id util 1.5
2022-10-29 20:33:18,455 [update()] [INFO ]  Utilization recorded [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.5]
2022-10-29 20:33:18,455 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:18,455 [update()] [INFO ]  Reading [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.5]
2022-10-29 20:33:20,459 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:20,459 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:20,459 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:20,459 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:20,459 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:20,459 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:20,459 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:20,459 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:20,459 [update()] [INFO ]  Utilization recorded [0.2, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
2022-10-29 20:33:20,459 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:20,460 [update()] [INFO ]  Reading [0.2, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
2022-10-29 20:33:22,463 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:22,463 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:22,463 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:22,463 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:22,463 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:22,463 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:22,463 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:22,463 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:22,463 [update()] [INFO ]  Utilization recorded [0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
2022-10-29 20:33:22,463 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:22,464 [update()] [INFO ]  Reading [0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
2022-10-29 20:33:24,467 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:24,467 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:24,467 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:24,468 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:24,468 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:24,468 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:24,468 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:24,468 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:24,468 [update()] [INFO ]  Utilization recorded [0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 1.0]
2022-10-29 20:33:24,468 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:24,468 [update()] [INFO ]  Reading [0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 1.0]
2022-10-29 20:33:26,472 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:26,472 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:26,472 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:26,472 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:26,472 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:26,472 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:26,472 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:26,472 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:26,472 [update()] [INFO ]  Utilization recorded [0.4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5]
2022-10-29 20:33:26,472 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:26,473 [update()] [INFO ]  Reading [0.4, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5]
2022-10-29 20:33:28,474 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:28,474 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:28,474 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:28,474 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:28,474 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:28,474 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:28,474 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:28,475 [update()] [INFO ]  Core id util 6.5
2022-10-29 20:33:28,475 [update()] [INFO ]  Utilization recorded [0.9, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 6.5]
2022-10-29 20:33:28,475 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:28,475 [update()] [INFO ]  Reading [0.9, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 6.5]
2022-10-29 20:33:30,478 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:30,479 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:30,479 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:30,479 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:30,479 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:30,479 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:30,479 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:30,479 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:30,479 [update()] [INFO ]  Utilization recorded [0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]
2022-10-29 20:33:30,479 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:30,479 [update()] [INFO ]  Reading [0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]
2022-10-29 20:33:32,483 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:32,483 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:32,483 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:32,483 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:32,483 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:32,483 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:32,483 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:32,483 [update()] [INFO ]  Core id util 4.5
2022-10-29 20:33:32,483 [update()] [INFO ]  Utilization recorded [0.7, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 4.5]
2022-10-29 20:33:32,483 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:32,484 [update()] [INFO ]  Reading [0.7, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 4.5]
2022-10-29 20:33:34,487 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:34,487 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:34,487 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:34,487 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:34,488 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:34,488 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:34,488 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:34,488 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:34,488 [update()] [INFO ]  Utilization recorded [0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5]
2022-10-29 20:33:34,488 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:34,488 [update()] [INFO ]  Reading [0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5]
2022-10-29 20:33:36,492 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:36,492 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:36,492 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:36,492 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:36,492 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:36,492 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:36,492 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:36,492 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:36,492 [update()] [INFO ]  Utilization recorded [0.2, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0]
2022-10-29 20:33:36,492 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:36,493 [update()] [INFO ]  Reading [0.2, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0]
2022-10-29 20:33:38,494 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:38,495 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:38,495 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:38,495 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:38,495 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:38,495 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:38,495 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:38,495 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:38,495 [update()] [INFO ]  Utilization recorded [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
2022-10-29 20:33:38,495 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:38,495 [update()] [INFO ]  Reading [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
2022-10-29 20:33:40,499 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:40,499 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:40,499 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:40,499 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:40,499 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:40,499 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:40,499 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:40,499 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:40,499 [update()] [INFO ]  Utilization recorded [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]
2022-10-29 20:33:40,499 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:40,499 [update()] [INFO ]  Reading [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]
2022-10-29 20:33:42,503 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:42,503 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:42,503 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:42,503 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:42,503 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:42,503 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:42,503 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:42,503 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:42,503 [update()] [INFO ]  Utilization recorded [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]
2022-10-29 20:33:42,503 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:42,504 [update()] [INFO ]  Reading [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0]
2022-10-29 20:33:44,507 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:44,507 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:44,507 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:44,507 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:44,508 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:44,508 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:44,508 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:44,508 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:44,508 [update()] [INFO ]  Utilization recorded [0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
2022-10-29 20:33:44,508 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:44,508 [update()] [INFO ]  Reading [0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
2022-10-29 20:33:46,511 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:46,511 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:46,511 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:46,511 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:46,511 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:46,511 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:46,511 [update()] [INFO ]  Core id util 3.5
2022-10-29 20:33:46,511 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:46,511 [update()] [INFO ]  Utilization recorded [0.6, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 3.5, 1.0]
2022-10-29 20:33:46,511 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:46,512 [update()] [INFO ]  Reading [0.6, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 3.5, 1.0]
2022-10-29 20:33:48,515 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:48,515 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:48,516 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:48,516 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:48,516 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:48,516 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:48,516 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:48,516 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:48,516 [update()] [INFO ]  Utilization recorded [0.2, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
2022-10-29 20:33:48,516 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:48,516 [update()] [INFO ]  Reading [0.2, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
2022-10-29 20:33:50,519 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:50,519 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:50,519 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:50,519 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:50,519 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:50,519 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:50,519 [update()] [INFO ]  Core id util 0.5
2022-10-29 20:33:50,519 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:50,519 [update()] [INFO ]  Utilization recorded [0.2, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0]
2022-10-29 20:33:50,519 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:50,520 [update()] [INFO ]  Reading [0.2, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0]
2022-10-29 20:33:52,523 [update()] [INFO ]  Core id util 1.0
2022-10-29 20:33:52,523 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:52,523 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:52,523 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:52,523 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:52,523 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:52,523 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:52,523 [update()] [INFO ]  Core id util 0.0
2022-10-29 20:33:52,523 [update()] [INFO ]  Utilization recorded [0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
2022-10-29 20:33:52,523 [update()] [INFO ]  Reading [2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006, 2500.006]
2022-10-29 20:33:52,524 [update()] [INFO ]  Reading [0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
2022-10-29 20:33:52,679 [kill_child_processes()] [DEBUG]  Killing stress process
2022-10-29 20:33:52,679 [kill_child_processes()] [DEBUG]  No such process
2022-10-29 20:33:52,679 [kill_child_processes()] [DEBUG]  Could not kill process
