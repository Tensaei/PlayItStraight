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

    def __init__(self, dataset, al_technique, selection_policy):
        self.dataset = dataset
        self.al_technique = al_technique
        self.n_samples_to_select = -1
        self.selection_policy = selection_policy

    def elaborate(self, model, target_epochs, step_training_epochs, n_samples_to_select, criterion, optimizer):
        self.n_samples_to_select = n_samples_to_select
        start_epochs = int(target_epochs / 10)
        completion_epochs = int(start_epochs * 9)
        clprint("Starting Active Learning process...", Reason.INFO_TRAINING)
        self._train_model(criterion, model, optimizer, start_epochs, step_training_epochs)
        clprint("Making one-shot AL process...", Reason.INFO_TRAINING, loggable=True)
        clprint("Selecting {} new samples...".format(self.n_samples_to_select), Reason.INFO_TRAINING)
        start_time = get_time_in_millis()
        self._select_next_samples()
        end_time = get_time_in_millis()
        clprint("Elapsed time: {} seconds".format(int((end_time - start_time) / 1000)), Reason.LIGHT_INFO_TRAINING, loggable=True)
        self._train_model(criterion, model, optimizer, completion_epochs, step_training_epochs)

    def _train_model(self, criterion, model, optimizer, target_epochs, step_training_epochs):
        clprint("Training model...", Reason.INFO_TRAINING)
        for epoch in range(0, target_epochs, step_training_epochs):
            start_time = support.get_time_in_millis()
            model.fit(step_training_epochs, criterion, optimizer, self.dataset.get_train_loader())
            elapsed_time = support.get_time_in_millis() - start_time
            clprint("Evaluating model...", Reason.INFO_TRAINING)
            loss, accuracy = model.evaluate(criterion, self.dataset.get_test_loader())
            clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time / 1000)), Reason.LIGHT_INFO_TRAINING, loggable=True)

    def _select_next_samples(self):
        x, y = self.dataset.get_unselected_data()
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
