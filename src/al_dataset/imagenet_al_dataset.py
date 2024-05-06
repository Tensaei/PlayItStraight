import src.support as support
from src.al_dataset.abstract_al_dataset import AbstractALDataset
from torchvision import datasets, transforms


class ImageNetALDataset(AbstractALDataset):

    def __init__(self, quantity_samples, rs2_enabled):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.ImageFolder(support.dataset_path.format("imagenet") + "/train", transform=transform)
        test_dataset = datasets.ImageFolder(support.dataset_path.format("imagenet") + "/val", transform=transform)
        super(ImageNetALDataset, self).__init__(quantity_samples, test_dataset, train_dataset, (3, 256, 256), rs2_enabled)
