import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from Network.Generator import get_noise
from utils.viz import show_tensor_images, show_real_fake
from Network import Generator, Discriminator
from tqdm.auto import tqdm

torch.manual_seed(0)  # Set for testing purposes, please do not change!criterion = nn.BCEWithLogitsLoss()
criterion = torch.nn.BCEWithLogitsLoss()
z_dim = 64
display_step = 500
batch_size = 128
# A learning rate of 0.0002 works well on DCGAN
lr = 0.0002

# These parameters control the optimizer's momentum, which you can read more about here:
# https://distill.pub/2017/momentum/ but you don’t need to worry about it for this course!
beta_1 = 0.5
beta_2 = 0.999
device = 'cuda:1'

# You can tranform the image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=False, transform=transform),
    batch_size=batch_size,
    shuffle=True)

gen = Generator.Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator.Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))


# You initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
n_epochs = 50
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
min_generator_loss = float('inf')
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        # == Update discriminator ==
        disc_opt.zero_grad()
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        fake = gen(fake_noise)
        disc_fake_pred = disc(fake.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()

        # == Update generator ==
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        disc_fake_pred = disc(fake_2)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        # == Visualization code ==
        if cur_step % display_step == 0 and cur_step > 0:
            print(
                f"/n Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss},"
                f" discriminator loss: {mean_discriminator_loss}")
            show_real_fake(real, fake)
            if mean_generator_loss < min_generator_loss:
                print(f'current loss {mean_generator_loss:.4f} is lower than min loss '
                      f'{min_generator_loss:.4f}, saving generator model.. ')
                torch.save(gen.state_dict(), './prediction/generator.pt')
                min_generator_loss = mean_generator_loss

            mean_generator_loss = 0
            mean_discriminator_loss = 0

        cur_step += 1
