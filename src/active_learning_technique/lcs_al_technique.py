import numpy
import torch
import src.support as support

from src.active_learning_technique.abstract_al_technique import AbstractALTechnique


class LCSALTechnique(AbstractALTechnique):

    def __init__(self, neural_network):
        self.neural_network = neural_network

    def evaluate_samples(self, x):
        outputs = []
        confidences = []
        for sample in x:
            # calculating least confidence
            out_model = self.neural_network(torch.unsqueeze(sample, 0).to(support.device))[0]
            #out_model = self.neural_network(sample.to(support.device))[0]
            outputs.append(out_model)
            simple_least_confidence = numpy.nanmax(out_model.cpu().detach().numpy())
            normalized_least_confidence = (1 - simple_least_confidence) * (len(out_model) / (len(out_model) - 1))
            # appending
            confidences.append(normalized_least_confidence)

        return outputs, confidences
