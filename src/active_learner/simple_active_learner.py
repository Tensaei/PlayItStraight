import numpy
import torch

from random import random
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

    def elaborate(self, model, target_epochs_phase_1, target_epochs_phase_2, step_training_epochs, n_samples_to_select, criterion, optimizer, scheduler, rs2_enabled):
        self.n_samples_to_select = n_samples_to_select
        start_epochs = target_epochs_phase_1
        completion_epochs = target_epochs_phase_2
        clprint("Starting Training process...", Reason.INFO_TRAINING)
        if rs2_enabled:
            self._rs2_train_model(criterion, model, optimizer, start_epochs, step_training_epochs, scheduler)

        else:
            self._train_model(criterion, model, optimizer, start_epochs, step_training_epochs, scheduler)

        clprint("Making one-shot AL process...", Reason.INFO_TRAINING, loggable=True)
        clprint("Selecting {} new samples...".format(self.n_samples_to_select), Reason.INFO_TRAINING)
        start_time = get_time_in_millis()
        self._select_next_samples()
        end_time = get_time_in_millis()
        clprint("Elapsed time: {} seconds".format(int((end_time - start_time) / 1000)), Reason.LIGHT_INFO_TRAINING, loggable=True)
        self._train_model(criterion, model, optimizer, completion_epochs, step_training_epochs, scheduler)

    def _train_model(self, criterion, model, optimizer, target_epochs, step_training_epochs, scheduler):
        clprint("Training model...", Reason.INFO_TRAINING)
        for epoch in range(0, target_epochs, step_training_epochs):
            start_time = support.get_time_in_millis()
            model.fit(step_training_epochs, criterion, optimizer, self.dataset.get_train_loader(), scheduler)
            elapsed_time = support.get_time_in_millis() - start_time
            #clprint("Evaluating model...", Reason.INFO_TRAINING)
            #loss, accuracy = model.evaluate(criterion, self.dataset.get_test_loader())

        loss, accuracy = model.evaluate(criterion, self.dataset.get_test_loader())
        clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time / 1000)), Reason.LIGHT_INFO_TRAINING, loggable=True)

    def _rs2_train_model(self, criterion, model, optimizer, target_epochs, step_training_epochs, scheduler):
        clprint("Training model with rs2...", Reason.INFO_TRAINING)
        dataset_batches = self.dataset.get_dataset_in_batches_rs2(target_epochs)
        for current_batch in dataset_batches:
            start_time = support.get_time_in_millis()
            model.fit(step_training_epochs, criterion, optimizer, current_batch, scheduler)
            elapsed_time = support.get_time_in_millis() - start_time
            #clprint("Evaluating model...", Reason.INFO_TRAINING)
            #loss, accuracy = model.evaluate(criterion, self.dataset.get_test_loader())

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
                    diff = torch.linalg.norm(torch.tensor(numpy.eye(quantity_classes)[selected_y[i]]).to(support.device) - model_y[i].to(support.device))
                    scores.append(t_scores[i] + diff.item())

                elif self.selection_policy == SelectionPolicy.KL:
                    diff = torch.nn.functional.kl_div(torch.tensor(numpy.eye(quantity_classes)[selected_y[i]]).to(support.device), model_y[i].to(support.device), reduction="mean")
                    scores.append(t_scores[i] + diff.item())

            self.dataset.annotate(self._random_distributed_selection(selected_x, scores, self.n_samples_to_select))

        clprint("Updating AL technique...".format(self.n_samples_to_select), Reason.LIGHT_INFO_TRAINING)
        self.al_technique.update(self.dataset)

    def _random_distributed_selection(self, x, scores, n_to_keep):
        total_score = sum(scores)
        normalized_scores = [val / total_score for val in scores]
        thresholds = [sum(normalized_scores[:i + 1]) for i in range(len(normalized_scores))]
        selected = []
        selected_indexes = []
        for i in range(n_to_keep):
            hook = random()
            for j in range(len(thresholds)):
                if hook < thresholds[j]:
                    if j in selected_indexes:
                        i -= 1

                    else:
                        selected.append(x[j])
                        selected_indexes.append(j)
                        break

        return selected
