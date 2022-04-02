from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

def get_mnist_loaders(batch_size=1):
    transform_mnist = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))])

    mnist_train = datasets.MNIST(root='./mnist', train=True, download=True, transform=transform_mnist)
    mnist_test = datasets.MNIST(root='./mnist', train=False, download=True, transform=transform_mnist)

    mnist_train_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)

    return mnist_train_loader, mnist_test_loader


def get_svhn_loaders(batch_size=1):
    transform_svhn = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    svhn_train = datasets.SVHN(root='./svhn', split='train', download=True, transform=transform_svhn)
    svhn_test = datasets.SVHN(root='./svhn', split='test', download=True, transform=transform_svhn)

    svhn_train_loader = DataLoader(dataset=svhn_train, batch_size=batch_size, shuffle=True)
    svhn_test_loader = DataLoader(dataset=svhn_test, batch_size=batch_size, shuffle=False)

    return svhn_train_loader, svhn_test_loader
