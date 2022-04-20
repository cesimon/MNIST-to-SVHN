import argparse
import datetime
import time
import sys
import itertools

import warnings
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.optim as optim
from torchsummary import summary
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

# Numpy imports
import numpy as np

# Local imports
from data_loader import get_mnist_loaders, get_svhn_loaders
from models import Discriminator, GeneratorResNet, weights_init_normal


SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(G_MNIST_SVHN, G_SVHN_MNIST, D_SVHN, D_MNIST):
    """Prints model information for the generators and discriminators.
    """
    print("                 G_MNIST_SVHN                ")
    print("---------------------------------------")
    summary(G_MNIST_SVHN, (1, 32, 32))
    print("---------------------------------------")

    print("                 G_SVHN_MNIST                ")
    print("---------------------------------------")
    summary(G_SVHN_MNIST, (3, 32, 32))
    print("---------------------------------------")

    print("                  D_SVHN                  ")
    print("---------------------------------------")
    summary(D_SVHN, (3, 32, 32))
    print("---------------------------------------")

    print("                  D_MNIST                  ")
    print("---------------------------------------")
    summary(D_MNIST, (1, 32, 32))
    print("---------------------------------------")

def create_model(opts):
    """Builds the generators and discriminators.
    """

    # We keep the original number of channels for MNIST and SVHN.
    # Therefore, the two generators take a different number of channels as input.
    # Likewise for discriminators.
    input_shape_svhn = (3, 32, 32)
    input_shape_mnist = (1, 32, 32)

    # Generator from MNIST to SVHN. Input: 1 channel. Output: 3 channels.
    G_MNIST_SVHN = GeneratorResNet(input_shape_mnist, opts.n_residual_blocks, 3)
    # Generator from SVHN to MNIST. Input: 3 channels. Output: 1 channel.
    G_SVHN_MNIST = GeneratorResNet(input_shape_svhn, opts.n_residual_blocks, 1)

    # Discriminator for SVHN
    D_SVHN = Discriminator(input_shape_svhn)
    # Discriminator for MNIST
    D_MNIST = Discriminator(input_shape_mnist)

    # Print architecture of created models
    #print_models(G_MNIST_SVHN, G_SVHN_MNIST, D_SVHN, D_MNIST)

    if torch.cuda.is_available():
        G_MNIST_SVHN.cuda()
        G_SVHN_MNIST.cuda()
        D_SVHN.cuda()
        D_MNIST.cuda()
        print('Models moved to GPU.')

    return G_MNIST_SVHN, G_SVHN_MNIST, D_SVHN, D_MNIST


def checkpoint(G_MNIST_SVHN, G_SVHN_MNIST, D_SVHN, D_MNIST, epoch):
    """Saves the parameters of both generators and discriminators.
    """
    torch.save(G_MNIST_SVHN.state_dict(), "saved_models/G_MNIST_SVHN_epoch_%d.pth" % epoch)
    torch.save(G_SVHN_MNIST.state_dict(), "saved_models/G_SVHN_MNIST_epoch_%d.pth" % epoch)
    torch.save(D_SVHN.state_dict(), "saved_models/D_SVHN_epoch_%d.pth" % epoch)
    torch.save(D_MNIST.state_dict(), "saved_models/D_MNIST_epoch_%d.pth" % epoch)


def sample_images(G_MNIST_SVHN, G_SVHN_MNIST, fixed_MNIST, fixed_SVHN, epoch):
    """Saves a generated sample from the test set"""
    G_MNIST_SVHN.eval()
    G_SVHN_MNIST.eval()

    # Generate images (translation and cycle)
    fake_MNIST = G_SVHN_MNIST(fixed_SVHN)
    fake_SVHN = G_MNIST_SVHN(fixed_MNIST)

    recov_MNIST = G_SVHN_MNIST(fake_SVHN)
    recov_SVHN = G_MNIST_SVHN(fake_MNIST)

    # Arange images along x-axis
    real_MNIST_grid = make_grid(fixed_MNIST, nrow=20, normalize=True)
    real_SVHN_grid = make_grid(fixed_SVHN, nrow=20, normalize=True)
    fake_SVHN_grid = make_grid(fake_SVHN, nrow=20, normalize=True)
    fake_MNIST_grid = make_grid(fake_MNIST, nrow=20, normalize=True)
    recov_MNIST_grid = make_grid(recov_MNIST, nrow=20, normalize=True)
    recov_SVHN_grid = make_grid(recov_SVHN, nrow=20, normalize=True)

    # Arange images along y-axis
    image_grid = torch.cat((real_MNIST_grid,
                            fake_SVHN_grid,
                            recov_MNIST_grid,
                            real_SVHN_grid,
                            fake_MNIST_grid,
                            recov_SVHN_grid),
                           1)

    save_image(image_grid, "images/epoch_%s.png" % epoch, normalize=False)


def training_loop(MNIST_dataloader,
                  SVHN_dataloader,
                  MNIST_test_dataloader,
                  SVHN_test_dataloader,
                  opts):
    """Runs the training loop.
        * Saves checkpoint/models every epoch
        * Saves generated samples every epoch
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Losses
    # We use MSE as loss for standard GAN training as in LSGAN
    criterion_GAN = torch.nn.MSELoss()
    # L1Loss for cycle-consistency loss as reminded in CyCADA paper
    criterion_cycle = torch.nn.L1Loss()


    # Models
    G_MNIST_SVHN, G_SVHN_MNIST, D_SVHN, D_MNIST = create_model(opts)

    # CycleGAN paper: weights are initialized from a Gaussian distribution N(0, 0.02)
    G_MNIST_SVHN.apply(weights_init_normal)
    G_SVHN_MNIST.apply(weights_init_normal)
    D_SVHN.apply(weights_init_normal)
    D_MNIST.apply(weights_init_normal)


    # Optimizers
    # Create optimizers for the generators and discriminators
    optimizer_G = optim.Adam(
        itertools.chain(G_MNIST_SVHN.parameters(), G_SVHN_MNIST.parameters()), lr=opts.lr, betas=[opts.b1, opts.b2]
    )
    optimizer_D_SVHN = optim.Adam(D_SVHN.parameters(), lr=opts.lr, betas=[opts.b1, opts.b2])
    optimizer_D_MNIST = optim.Adam(D_MNIST.parameters(), lr=opts.lr, betas=[opts.b1, opts.b2])


    # Iterators for test data
    test_iter_MNIST = iter(MNIST_test_dataloader)
    test_iter_SVHN = iter(SVHN_test_dataloader)

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_MNIST, _ = next(test_iter_MNIST)
    fixed_SVHN, _ = next(test_iter_SVHN)
    while fixed_MNIST.size(0) < 20:
        fixed_MNIST = torch.cat((fixed_MNIST, next(test_iter_MNIST)[0]), 0)
        fixed_SVHN = torch.cat((fixed_SVHN, next(test_iter_SVHN)[0]), 0)

    # Take first 20 pictures
    fixed_MNIST = fixed_MNIST[0:20]
    fixed_SVHN = fixed_SVHN[0:20]
    fixed_MNIST = fixed_MNIST.to(device)
    fixed_SVHN = fixed_SVHN.to(device)


    # Writer to log events with Tensorboard
    writer = SummaryWriter(opts.tensorboard_path)

    # Used to evaluate time needed to process 1 batch
    prev_time = time.time()

    # Main training loop
    for epoch in range(1, opts.n_epochs+1):
        # Reset for each epoch
        iter_MNIST = iter(MNIST_dataloader)
        iter_SVHN = iter(SVHN_dataloader)

        iter_per_epoch = min(len(iter_MNIST), len(iter_SVHN))

        for i in range(iter_per_epoch):
            # Current batch
            MNIST_imgs, _ = next(iter_MNIST)
            SVHN_imgs, _ = next(iter_SVHN)

            MNIST_imgs = MNIST_imgs.to(device)
            SVHN_imgs = SVHN_imgs.to(device)

            # Adversarial ground truths
            valid = torch.tensor(np.ones((MNIST_imgs.size(0), *D_MNIST.output_shape)),
                                 requires_grad=False, dtype=torch.float32).to(device)
            fake = torch.tensor(np.zeros((MNIST_imgs.size(0), *D_MNIST.output_shape)),
                                requires_grad=False, dtype=torch.float32).to(device)

            G_MNIST_SVHN.train()
            G_SVHN_MNIST.train()

            # -----------------------
            #  Train Discriminator MNIST
            # -----------------------

            optimizer_D_MNIST.zero_grad()

            D_MNIST_loss_real = criterion_GAN(D_MNIST(MNIST_imgs), valid)
            fake_MNIST = G_SVHN_MNIST(SVHN_imgs)
            D_MNIST_loss_fake = criterion_GAN(D_MNIST(fake_MNIST), fake)
            D_MNIST_loss = (D_MNIST_loss_real + D_MNIST_loss_fake) / 2

            D_MNIST_loss.backward()
            optimizer_D_MNIST.step()

            # -----------------------
            #  Train Discriminator SVHN
            # -----------------------

            optimizer_D_SVHN.zero_grad()

            D_SVHN_loss_real = criterion_GAN(D_SVHN(SVHN_imgs), valid)
            fake_SVHN = G_MNIST_SVHN(MNIST_imgs)
            D_SVHN_loss_fake = criterion_GAN(D_SVHN(fake_SVHN), fake)
            D_SVHN_loss = (D_SVHN_loss_real + D_SVHN_loss_fake) / 2

            D_SVHN_loss.backward()
            optimizer_D_SVHN.step()

            # Total Discriminator loss
            D_loss = (D_MNIST_loss + D_SVHN_loss) / 2

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Regenerate fake images for generator training
            fake_MNIST = G_SVHN_MNIST(SVHN_imgs)
            fake_SVHN = G_MNIST_SVHN(MNIST_imgs)

            # GAN loss
            SVHN_MNIST_GAN_loss = criterion_GAN(D_MNIST(fake_MNIST), valid)
            MNIST_SVHN_GAN_loss = criterion_GAN(D_SVHN(fake_SVHN), valid)

            G_GAN_loss = (SVHN_MNIST_GAN_loss + MNIST_SVHN_GAN_loss) / 2

            # Cycle loss
            recov_MNIST = G_SVHN_MNIST(fake_SVHN)
            MNIST_cycle_loss = criterion_cycle(recov_MNIST, MNIST_imgs)

            recov_SVHN = G_MNIST_SVHN(fake_MNIST)
            SVHN_cycle_loss = criterion_cycle(recov_SVHN, SVHN_imgs)

            G_cycle_loss = (MNIST_cycle_loss + SVHN_cycle_loss) / 2

            # Total generator loss
            G_loss = G_GAN_loss + opts.lambda_cyc * G_cycle_loss

            G_loss.backward()
            optimizer_G.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = (epoch - 1) * iter_per_epoch + (i + 1)
            batches_left = opts.n_epochs * iter_per_epoch - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            if i % 50 == 0:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f] ETA: %s"
                    % (
                        epoch,
                        opts.n_epochs,
                        i,
                        iter_per_epoch,
                        D_loss.item(),
                        G_loss.item(),
                        G_GAN_loss.item(),
                        G_cycle_loss.item(),
                        time_left
                    )
                )

                writer.add_scalar('Loss/Discriminator', D_loss, (epoch-1)*iter_per_epoch + i)
                writer.add_scalar('Loss/Generator', G_loss, (epoch-1)*iter_per_epoch + i)
                writer.add_scalar('Loss/Generator_GAN', G_GAN_loss, (epoch-1)*iter_per_epoch + i)
                writer.add_scalar('Loss/Generator_Cycle', G_cycle_loss, (epoch-1)*iter_per_epoch + i)

        # Save the generated samples
        sample_images(G_MNIST_SVHN, G_SVHN_MNIST, fixed_MNIST, fixed_SVHN, epoch)

        checkpoint(G_MNIST_SVHN, G_SVHN_MNIST, D_SVHN, D_MNIST, epoch)


def main(opts):
    """Loads the data, starts the training loop.
    """
    mnist_train_loader, mnist_test_loader = get_mnist_loaders(opts.batch_size)
    svhn_train_loader, svhn_test_loader = get_svhn_loaders(opts.batch_size)

    # Start training
    training_loop(mnist_train_loader, svhn_train_loader, mnist_test_loader, svhn_test_loader, opts)


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--tensorboard_path", default='./runs/default', help="tensorboard path")

    return parser

if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()

    print_opts(opts)
    main(opts)
