import numpy
import torch

from torch import nn
from enum import Enum
from src import support
from src.support import clprint, Reason, get_time_in_millis


class SelectionPolicy(Enum):
    L2 = 1
    KL = 3


class SimpleActiveLearner:

    def __init__(self, dataset, al_technique, target_accuracy, selection_policy, pool_size):
        self.dataset = dataset
        self.al_technique = al_technique
        self.n_samples_to_select = -1
        self.target_accuracy = target_accuracy
        self.selection_policy = selection_policy
        self.pool_size = pool_size

    def elaborate(self, model, training_epochs, n_samples_to_select, criterion, optimizer):
        self.n_samples_to_select = n_samples_to_select
        clprint("Starting Active Learning process...", Reason.INFO_TRAINING)
        self._train_model(criterion, model, optimizer, training_epochs)
        accuracy = 0
        i = 1
        total_epochs = 0
        while accuracy < self.target_accuracy:
            clprint("Making n.{} AL epochs...".format(i), Reason.INFO_TRAINING, loggable=True)
            clprint("Selecting {} new samples...".format(self.n_samples_to_select), Reason.INFO_TRAINING)
            start_time = get_time_in_millis()
            self._select_next_samples()
            end_time = get_time_in_millis()
            clprint("Elapsed time: {} seconds".format(int((end_time - start_time) / 1000)), Reason.LIGHT_INFO_TRAINING, loggable=True)
            accuracy = self._train_model(criterion, model, optimizer, training_epochs)
            i += 1
            total_epochs += training_epochs

        return total_epochs

    def _train_model(self, criterion, model, optimizer, training_epochs):
        clprint("Training model...", Reason.INFO_TRAINING)
        model.fit(training_epochs, criterion, optimizer, self.dataset.get_train_loader())
        clprint("Evaluating model...", Reason.INFO_TRAINING)
        loss, accuracy = model.evaluate(criterion, self.dataset.get_test_loader())
        clprint("Loss: {}\nAccuracy: {}".format(loss, accuracy), Reason.LIGHT_INFO_TRAINING, loggable=True)
        return accuracy

    def _select_next_samples(self):
        x, y = self.dataset.get_unselected_data(self.pool_size)
        selected_x, model_y, selected_y, t_scores = self.al_technique.select_samples(x, y, self.n_samples_to_select * 2)

        if t_scores is None:
            self.dataset.annotate(selected_x[:self.n_samples_to_select])

        else:
            quantity_classes = max(y) + 1
            scores = []
            for i in range(len(selected_x)):
                if self.selection_policy == SelectionPolicy.L2:
                    diff = torch.linalg.norm(torch.tensor(numpy.eye(quantity_classes)[y[i]]).to(support.device) - model_y[i].to(support.device))
                    scores.append(t_scores[i].item() + diff.item())

                elif self.selection_policy == SelectionPolicy.KL:
                    diff = torch.nn.functional.kl_div(torch.tensor(numpy.eye(quantity_classes)[y[i]]).to(support.device), model_y[i].to(support.device), reduction="mean")
                    scores.append(t_scores[i].item() + diff.item())

            def sort_by_float(current_tuple):
                return current_tuple[0]

            combined_list = list(zip(scores, selected_x))
            combined_list = sorted(combined_list, key=sort_by_float, reverse=True)
            _, selected_x = zip(*combined_list)
            self.dataset.annotate(selected_x[:self.n_samples_to_select])

        clprint("Updating AL technique...".format(self.n_samples_to_select), Reason.LIGHT_INFO_TRAINING)
        self.al_technique.update(self.dataset)
