import argparse

import numpy as np

import torch
import torch.optim as optim

from models import LeNet, GeneratorResNet

from data_loader import get_mnist_loaders, get_svhn_loaders


def train_epoch(model, opt, criterion, dataloader, device, generator=None):
    # Train classifier for one epoch

    model.train()
    losses = []

    for i, (imgs, labels) in enumerate(dataloader):
        opt.zero_grad()

        # If a generator was provided, images are translated
        # Concretely, this means we train the classifier using translated MNIST images (to SVHN) and MNIST labels
        # Otherwise, the classifier is trained on SVHN, using labels from SVHN, in a classic way
        imgs_for_train = imgs.to(device)
        if generator is not None:
            imgs_for_train = generator(imgs_for_train)

        out = model(imgs_for_train)
        # Our classifier's last layer is Linear (no activation)
        # The softmax is integrated into the criterion (CrossEntropyLoss)
        # out has shape torch.Size([BS, 10]) where BS is Batch size
        # Reminder: Here, SVHN labels are in [|0,9|], just like MNIST labels
        loss = criterion(out, labels.to(device))

        loss.backward()

        opt.step()
        losses.append(loss.item())

    return sum(losses) / (i+1)


def eval_model(model, criterion, dataloader, device):
    # We compute the average loss for one batch on the test set,
    # and the accuracy on the test set as metrics

    model.eval()
    total_epoch_loss = 0
    all_preds = np.array([])
    all_ys = np.array([])

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            out = model(imgs.to(device))
            loss = criterion(out, labels.to(device))

            # Predictions
            # Our classifier's last layer is Linear (no activation)
            # We need to use argmax to obtain the predicted label
            all_preds = np.append(all_preds, torch.argmax(out, dim=1).cpu().detach().numpy())
            # Ground truth
            all_ys = np.append(all_ys, labels.cpu().detach().numpy())
            total_epoch_loss += loss.item()

    return total_epoch_loss / (i + 1), np.mean(all_preds == all_ys)


def train_SVHN_classifier(train_dataloader, test_dataloader, device, generator=None):
    model = LeNet(3).to(device)
    opt = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    criterion = torch.nn.CrossEntropyLoss()

    best_test_loss = 1000

    train_losses = []
    # Model is trained on 10 epochs, with early stopping
    for epoch in range(10):
        epoch_number = epoch + 1
        print('Epoch: ', epoch_number)

        # Train model
        train_losses.append(train_epoch(model, opt, criterion, train_dataloader, device, generator=generator))
        # Evaluate model on test set (test set is always SVHN test set)
        test_loss, test_accuracy = eval_model(model, criterion, test_dataloader, device)

        print("Epoch " + str(epoch_number) + " : Test loss = " + str(test_loss))
        print("Epoch " + str(epoch_number) + " : Test accuracy = " + str(test_accuracy))

        # Rudimentary early stopping mechanism
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        else:
            print('Early Stopping')
            break

    return train_losses


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_translation", action='store_true', help="whether to use translation from MNIST to SVHN")
    parser.add_argument("--generator_path", help="path to file for generator from MNIST to SVHN")

    return parser


if __name__ == '__main__':
    # The aim is to train a classifier on SVHN
    # If use_translation is True, the classifier is trained on translated images from MNIST to SVHN,
    # using the source labels from MNIST
    # Otherwise, the classifier is trained on SVHN using the target labels from SVHN

    # Get options
    parser = create_parser()
    opts = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opts.use_translation:
        # Initialize generator from MNIST to SVHN from saved weights
        input_shape_mnist = (1, 32, 32)
        G_MNIST_SVHN = GeneratorResNet(input_shape_mnist, 2, 3)

        path = opts.generator_path
        G_MNIST_SVHN.load_state_dict(torch.load(path, map_location=device))
        G_MNIST_SVHN.to(device)
        G_MNIST_SVHN.eval()

    # Get dataloaders
    mnist_train_loader, _ = get_mnist_loaders(128)
    svhn_train_loader, svhn_test_loader = get_svhn_loaders(128)

    # Start training
    if opts.use_translation:
        train_SVHN_classifier(mnist_train_loader, svhn_test_loader, device, generator=G_MNIST_SVHN)
    else:
        train_SVHN_classifier(svhn_train_loader, svhn_test_loader, device)
