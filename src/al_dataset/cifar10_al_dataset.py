import src.support as support
from src.al_dataset.abstract_al_dataset import AbstractALDataset
from torchvision import datasets, transforms


class Cifar10ALDataset(AbstractALDataset):

    def __init__(self, quantity_samples, rs2_enabled):
        self.input_shape = (3, 32, 32)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_dataset = datasets.CIFAR10(root=support.dataset_path.format("cifar10"), train=False, transform=transform_train, download=True)
        train_dataset = datasets.CIFAR10(root=support.dataset_path.format("cifar10"), train=True, transform=transform_test, download=True)
        super(Cifar10ALDataset, self).__init__(quantity_samples, test_dataset, train_dataset, self.input_shape, rs2_enabled)
