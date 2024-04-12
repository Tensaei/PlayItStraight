import numpy
import torch
import random

import src.support as support
from src.al_dataset.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class AbstractALDataset:

    def __init__(self, quantity_samples, test_dataset, train_dataset, shape_data):
        self.quantity_samples = quantity_samples
        self.shape_data = shape_data
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=support.model_batch_size, shuffle=False)
        self.x_labeled = []
        self.y_labeled = []
        self.x_labeled_last_batch = None
        self.y_labeled_last_batch = None
        self.unlabeled_dict = {}
        self.test_dict = {}
        x_dataset = []
        y_dataset = []
        for i, data in enumerate(train_dataset):
            image, label = data
            x_dataset.append(image)
            y_dataset.append(label)

        map_training_dataset = list(zip(x_dataset, y_dataset))
        random.shuffle(map_training_dataset)
        x_dataset, y_dataset = zip(*map_training_dataset)
        for i in range(len(train_dataset)):
            if quantity_samples == -1 or i < quantity_samples:
                self.x_labeled.append(x_dataset[i])
                self.y_labeled.append(y_dataset[i])

            else:
                self.unlabeled_dict[x_dataset[i]] = y_dataset[i]

        size = 1
        for dim in self.shape_data:
            size *= dim

        self.test_x = numpy.zeros((len(test_dataset), size))
        self.test_y = numpy.zeros(len(test_dataset))
        i = 0
        for _, data in enumerate(test_dataset):
            image, label = data
            self.test_dict[image] = label
            self.test_x[i] = image.cpu().detach().numpy().reshape(1, -1)
            self.test_y[i] = label
            i += 1

        self.quantity_classes = int(max(self.test_y) + 1)

    def __len__(self):
        return len(self.x_labeled)

    def get_unselected_data(self, pool_size=-1):
        if pool_size == -1:
            x = list(self.unlabeled_dict.keys())
            return x, [self.unlabeled_dict[i] for i in x]

        else:
            quantity_per_class = int(pool_size/self.quantity_classes)
            x = list(self.unlabeled_dict.keys())
            random.shuffle(x)
            counters_for_classes = [0] * self.quantity_classes
            balanced_x = []
            balanced_y = []
            for i in x:
                current_class = self.unlabeled_dict[i]
                if counters_for_classes[current_class] < quantity_per_class:
                    counters_for_classes[current_class] += 1
                    balanced_x.append(i)
                    balanced_y.append(current_class)

            return balanced_x, balanced_y

    def annotate(self, x_to_label):
        # adding old batch to labeled set
        if self.x_labeled_last_batch is not None:
            for i in range(len(self.x_labeled_last_batch)):
                self.x_labeled.append(self.x_labeled_last_batch[i])
                self.y_labeled.append(self.y_labeled_last_batch[i])

        # building new batch
        self.x_labeled_last_batch = []
        self.y_labeled_last_batch = []
        for value in x_to_label:
            self.x_labeled_last_batch.append(value)
            self.y_labeled_last_batch.append(self.unlabeled_dict.pop(value))

    def get_train_loader(self):
        if self.x_labeled_last_batch is None:
            return DataLoader(Dataset(self.shape_data, self.x_labeled, self.y_labeled), batch_size=support.model_batch_size)

        else:
            # building train loader
            train_x = []
            train_y = []
            # taking the same quantity of last batch from the labeled data pool
            map_training_dataset = list(zip(self.x_labeled, self.y_labeled))
            random.shuffle(map_training_dataset)
            x_dataset, y_dataset = zip(*map_training_dataset)
            counters_for_class = [0] * self.quantity_classes
            for i in range(len(x_dataset)):
                if counters_for_class[y_dataset[i]] < int(len(self.x_labeled_last_batch)/self.quantity_classes):
                    counters_for_class[y_dataset[i]] += 1
                    train_x.append(x_dataset[i])
                    train_y.append(y_dataset[i])

            train_x.extend(self.x_labeled_last_batch)
            train_y.extend(self.y_labeled_last_batch)
            return DataLoader(Dataset(self.shape_data, train_x, train_y), batch_size=support.model_batch_size)

    def get_test_loader(self):
        return self.test_loader

    def get_train_numpy(self):
        size = 1
        for dim in self.shape_data:
            size *= dim

        x = numpy.zeros((len(self.x_labeled), size))
        y = numpy.zeros(len(self.x_labeled))
        for i in range(len(self.x_labeled)):
            x[i] = self.x_labeled[i].cpu().detach().numpy().reshape(1, -1)
            y[i] = self.y_labeled[i]

        return x, y

    def get_test_numpy(self):
        return self.test_x, self.test_y
