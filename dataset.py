import torchvision

train_dataset = torchvision.datasets.CIFAR10(root='./CIFAR-10', train=True, download=True)