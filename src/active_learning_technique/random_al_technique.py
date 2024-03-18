import random
import torch

from src import support
from src.active_learning_technique.abstract_al_technique import AbstractALTechnique


class RandomALTechnique(AbstractALTechnique):

    def __init__(self, neural_network):
        self.neural_network = neural_network

    def evaluate_samples(self, x):
        outputs = []
        for sample in x:
            out_model = self.neural_network(torch.unsqueeze(sample, 0).to(support.device))[0]
            # out_model = self.neural_network(sample.to(support.device))[0]
            outputs.append(out_model)

        return outputs, [random.random() for _ in range(x)]
