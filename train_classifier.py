from models import LeNet
import torch
import torch.optim as optim
import numpy as np
from data_loader import get_mnist_loaders

def train_epoch(model, opt, criterion, dataloader, device):
    model.train()
    losses = []

    for i, (imgs, labels) in enumerate(dataloader):
        opt.zero_grad()
        out = model(imgs.to(device))
        loss = criterion(out, labels.to(device))

        loss.backward()

        opt.step()
        losses.append(loss.item())

    return sum(losses) / (i+1)


def eval_model(model, criterion, dataloader, device):
    model.eval()
    total_epoch_loss = 0
    all_preds = np.array([])
    all_ys = np.array([])

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            out = model(imgs.to(device))
            loss = criterion(out, labels.to(device))

            all_preds = np.append(all_preds, torch.argmax(out, dim=1).cpu().detach().numpy())
            all_ys = np.append(all_ys, labels.cpu().detach().numpy())
            total_epoch_loss += loss.item()

    return total_epoch_loss / (i + 1), np.mean(all_preds == all_ys)

def train_classifier(train_dataloader, test_dataloader, input_channels, device):
    model = LeNet(input_channels).to(device)
    opt = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = torch.nn.CrossEntropyLoss()

    best_test_loss = 100

    train_losses = []
    for epoch in range(10):
        epoch_number = epoch + 1
        print('Epoch: ', epoch_number)

        train_losses.append(train_epoch(model, opt, criterion, train_dataloader, device))
        test_loss, test_accuracy = eval_model(model, criterion, test_dataloader, device)

        torch.save(model.state_dict(), "saved_models/classifier/MNIST_classifier_epoch_%d.pth" % epoch_number)

        print("Epoch " + str(epoch_number) + " : Test loss = " + str(test_loss))
        print("Epoch " + str(epoch_number) + " : Test accuracy = " + str(test_accuracy))
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        else:
            print('Early Stopping')
            break

    return train_losses

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mnist_train_loader, mnist_test_loader = get_mnist_loaders(100)

    train_classifier(mnist_train_loader, mnist_test_loader, 1, device)



