from torchvision import datasets, transforms
from torch import tensor, long

def FashionMNIST(args):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.2861]
    std = [0.3530]
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(size=28, padding=4),transforms.ToTensor(),transforms.Normalize(mean=mean, std=std),])
    test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean, std=std),])

    root = args.data_path + '/fashionmnist'
    dst_train = datasets.FashionMNIST(root, train=True, download=True, transform=train_transform)
    dst_unlabeled = datasets.FashionMNIST(root, train=True, download=True, transform=test_transform)
    dst_test = datasets.FashionMNIST(root, train=False, download=True, transform=test_transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_unlabeled, dst_test