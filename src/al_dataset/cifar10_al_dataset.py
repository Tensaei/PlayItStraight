import src.support as support
from src.al_dataset.abstract_al_dataset import AbstractALDataset
from torchvision import datasets, transforms


class Cifar10ALDataset(AbstractALDataset):

    def __init__(self, quantity_samples, rs2_enabled):
        self.input_shape = (3, 32, 32)
        test_dataset = datasets.CIFAR10(root=support.dataset_path.format("cifar10"), train=False, transform=transforms.ToTensor(), download=True)
        train_dataset = datasets.CIFAR10(root=support.dataset_path.format("cifar10"), train=True, transform=transforms.ToTensor(), download=True)
        super(Cifar10ALDataset, self).__init__(quantity_samples, test_dataset, train_dataset, self.input_shape, rs2_enabled)
