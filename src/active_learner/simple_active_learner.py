import numpy
import torch
import src.support as support

from src.support import clprint, Reason, get_time_in_millis, device


class SimpleActiveLearner:

    def __init__(self, dataset, al_technique):
        self.dataset = dataset
        self.al_technique = al_technique
        self.n_samples_to_select = -1

    def elaborate(self, model, al_epochs, training_epochs, n_samples_to_select, criterion, optimizer):
        self.n_samples_to_select = n_samples_to_select
        clprint("Starting Active Learning process...", Reason.INFO_TRAINING)
        self._train_model(criterion, model, optimizer, training_epochs)
        for i in range(al_epochs):
            clprint("Making n.{}/{} AL epochs...".format(i + 1, al_epochs), Reason.INFO_TRAINING, loggable=True)
            clprint("Selecting {} new samples...".format(self.n_samples_to_select), Reason.INFO_TRAINING)
            start_time = get_time_in_millis()
            self._select_next_samples(self.n_samples_to_select)
            end_time = get_time_in_millis()
            clprint("Elapsed time: {}".format(end_time - start_time), Reason.LIGHT_INFO_TRAINING, loggable=True)
            self._train_model(criterion, model, optimizer, training_epochs)

    def _train_model(self, criterion, model, optimizer, training_epochs):
        clprint("Training model...", Reason.INFO_TRAINING)
        model.fit(training_epochs, criterion, optimizer, self.dataset.get_train_loader())
        clprint("Evaluating model...", Reason.INFO_TRAINING)
        loss, accuracy = model.evaluate(criterion, self.dataset.get_test_loader())
        clprint("Loss: {}\nAccuracy: {}".format(loss, accuracy), Reason.LIGHT_INFO_TRAINING, loggable=True)

    def _select_next_samples(self, n_samples_to_select):
        x, y = self.dataset.get_unselected_data()
        outputs, t_scores = self.al_technique.evaluate_samples(x)
        # combining AL score with comparison of real output with calculated output
        quantity_classes = max(y) + 1
        scores = []
        for i in range(len(x)):
            #diff = torch.linalg.norm(torch.tensor(numpy.eye(quantity_classes)[y[i]]).to(support.device) - outputs[i])
            #scores.append(t_scores[i].item() + diff.item())
            scores.append(t_scores[i].item())

        # balancing the data by taking the same amount from each class
        def sort_by_float(current_tuple):
            return current_tuple[0]

        combined_list = list(zip(scores, x, y))
        combined_list = sorted(combined_list, key=sort_by_float, reverse=True)
        scores, x, y = zip(*combined_list)

        n_samples_to_select_for_each_class = int(n_samples_to_select/quantity_classes)
        counters_for_classes = [0] * quantity_classes
        selected_x = []
        for i in range(len(x)):
            current_class = y[i]
            if counters_for_classes[current_class] < n_samples_to_select_for_each_class:
                counters_for_classes[current_class] += 1
                selected_x.append(x[i])

        self.dataset.annotate(selected_x)
        clprint("Updating AL technique...".format(self.n_samples_to_select), Reason.LIGHT_INFO_TRAINING)
        self.al_technique.update(self.dataset)
