import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageOps, ImageDraw
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import random

WIDTH = 28
font = ImageFont.truetype("SIMSUN.ttf", WIDTH)  # Assumes chinese chars have width == font size
CHARS = [chr(i) for i in range(19968, 40890)]

# Ensure chars fit in WIDTH x WIDTH square
assert (max([font.getmask(char).size[0] for char in CHARS]) <= WIDTH)
assert (max([font.getmask(char).size[1] for char in CHARS]) <= WIDTH)

# Turn CHARS into N x 1 x WIDTH x WIDTH Pytorch Tensor  (batch, channels, width, height)
char_arrays = []
for char in CHARS:
    mask = font.getmask(char)
    im = Image.new('L', (WIDTH, WIDTH), 0)
    ImageDraw.Draw(im).text((0,0), char, 255, font=font)
    char_arrays.append(np.array(im))
chars_tensor = torch.from_numpy(np.array(char_arrays)).unsqueeze(1).float() / 255  # 1.0 is max value

def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1) # channels as last dimension
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())



ZDIM = 100

# (N, 1, WIDTH, WIDTH) -> (N, 1)
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Convolutional steps to map (N, input_channels, input_width, input_height)
        #                         -> (N, output_channels, output_width, output_height)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU()
        )
        # Linear steps to map (N, length) -> (N, 1)
        self.linear = nn.Sequential(
            nn.Linear(18432, 1),
        )
    def forward(self, x):
        N = x.shape[0] # Batch Size
        x = self.conv(x)
        x = x.view(N, -1) # Make linear
        x = self.linear(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        feats = 64
        self.main = nn.Sequential(
            nn.ConvTranspose2d(ZDIM, feats * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feats * 8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(feats * 8, feats * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feats * 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(feats * 4, feats * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feats * 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(feats * 2, 1, 2, 2, 2, bias=False),
        )
    def forward(self, x):
        N = x.shape[0] # Batch Size
        x = x.unsqueeze(2).unsqueeze(3) # Artificial dimensions in preparation for CNN
        x = self.main(x)
        return torch.sigmoid(x)

train_data = torch.utils.data.TensorDataset(chars_tensor)
train_loader = torch.utils.data.DataLoader(train_data, 64, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', device)
D = Discriminator().to(device)
G = Generator().to(device)
optimizerD = torch.optim.RMSprop(D.parameters(), lr=0.002)
optimizerG = torch.optim.RMSprop(G.parameters(), lr=0.002)
criterion = nn.BCELoss()


for epoch in range(128):
    sum_lossD = 0
    sum_lossG = 0
    print(f'Epoch {epoch}')
    for i, batch in enumerate(train_loader):
        # STEP 1: Discriminator optimization step
        x_real = batch[0]
        N = x_real.shape[0]
        x_real = x_real.to(device)
        lab_real = torch.ones(N, 1, device=device)
        lab_fake = torch.zeros(N, 1, device=device)
        # reset accumulated gradients from previous iteration
        optimizerD.zero_grad()

        D_x = D(x_real)

        z = torch.randn(N, ZDIM, device=device)
        x_gen = G(z).detach()
        D_G_z = D(x_gen)

        lossD = -1 * sum(D_x - D_G_z) # Wasserstein
        lossD.backward()
        optimizerD.step()
        sum_lossD += lossD.item()

        # STEP 2: Generator optimization step
        # reset accumulated gradients from previous iteration
        optimizerG.zero_grad()

        z = torch.randn(N, ZDIM, device=device)
        x_gen = G(z)
        D_G_z = D(x_gen)
        lossG = -1 * sum(D_G_z) # Wasserstein
        lossG.backward()
        optimizerG.step()
        sum_lossG += lossG.item()

        if i % 100 == 0:
            print(i)
    print(f'LossD = {sum_lossD}, LossG = {sum_lossG}')


print(D(iter(train_loader).next()[0].to(device)))
z = torch.randn(10, ZDIM, device=device)
print(D(G(z)))

# Generate some samples
z = torch.randn(64, ZDIM, device=device)
x_gen = G(z)
show_imgs(1-x_gen)


a = torch.randn(1, ZDIM, device=device)
b = torch.randn(1, ZDIM, device=device)
z = torch.cat([t * a + (1-t) * b for t in np.arange(0, 1, 1/64)])
show_imgs(1-G(z))


pt_list = list(torch.randn(9, 1, ZDIM, device=device))
z = torch.cat([(1-t) * a + t * b for a, b in zip(pt_list, pt_list[1:]) for t in np.arange(0, 1, 1/8)])
show_imgs(1-G(z))



# Make a GIF
imgs = [Image.fromarray(arr) for arr in np.uint8((1-G(z)).cpu().detach().numpy() * 255).squeeze(1)]
imgs[0].save('results/out.gif', save_all=True, append_images=imgs[1:], loop=0)


# Fancier GIF
pt_list = list(2 * torch.randn(16, 1, ZDIM, device=device))
z = torch.cat([(1-t) * a + t * b for a, b in zip(pt_list, pt_list[1:] + [pt_list[0]]) for t in np.arange(0, 1, 1/10)])
imgs = [Image.fromarray(arr) for arr in np.uint8((1-G(z)).cpu().detach().numpy() * 255).squeeze(1)]
imgs[0].save('results/out.gif', save_all=True, append_images=imgs[1:], loop=0, duration = 200)
