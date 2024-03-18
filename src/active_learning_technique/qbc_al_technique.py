import numpy
import torch

from scipy.spatial.distance import pdist
from torch import nn

from src import support
from src.active_learning_technique.abstract_al_technique import AbstractALTechnique


class QueryByCommiteeALTechnique(AbstractALTechnique):

    def __init__(self, models, n_classes):
        self.models = models
        self.n_classes = n_classes

    def evaluate_samples(self, x):
        disagreements = numpy.zeros(len(x))
        outputs = []
        for sample_index in range(len(x)):
            predictions = numpy.zeros((len(self.models), self.n_classes))
            for model_index in range(len(self.models)):
                predictions[model_index] = self.models[model_index].predict(x[sample_index].squeeze())
                if isinstance(self.models[model_index], nn.Module):
                    outputs.append(self.neural_network(torch.unsqueeze(x[sample_index], 0).to(support.device))[0])

            # calculating disagreement
            disagreements[sample_index] = numpy.mean(pdist(predictions))

        normalized_disagreements = (disagreements - numpy.min(disagreements)) / (numpy.max(disagreements) - numpy.min(disagreements))

        return outputs, normalized_disagreements.tolist()

    def update(self, dataset):
        for model in self.models:
            model.update(dataset)
