import numpy
import torch
import src.support as support

from torch import nn
from src.active_learning_technique.abstract_al_technique import AbstractALTechnique
from src.al_dataset.dataset import Dataset


class LCSALTechnique(AbstractALTechnique):

    def __init__(self, neural_network, shape_data):
        self.neural_network = neural_network
        self.shape_data = shape_data

    def select_samples(self, x, y, n_samples_to_select):
        fun_soft = nn.Softmax(dim=1)
        sub_train_loader = torch.utils.data.DataLoader(Dataset(self.shape_data, x, y), batch_size=support.model_batch_size)
        scores = []
        outputs = []
        self.neural_network.eval()
        with torch.no_grad():
            for x_batch, _, _ in sub_train_loader:
                x_batch = x_batch.to(support.device)
                output = self.neural_network(x_batch)
                output_copy = fun_soft(torch.Tensor.cpu(output)).detach().numpy()
                lc_value = numpy.max(output_copy, axis=1)
                scores = numpy.concatenate((scores, lc_value))
                outputs.extend(output)

        self.neural_network.train()
        scores_sort = numpy.argsort(scores)
        scores_lcs = [int(item) for item in scores_sort]
        select_loc = scores_lcs[-n_samples_to_select * 2:]
        selected_samples = [x[index] for index in select_loc]
        real_y = [y[index] for index in select_loc]
        model_y = [outputs[index] for index in select_loc]
        selected_scores = [scores_lcs[index] for index in select_loc]
        return selected_samples, model_y, real_y, selected_scores
