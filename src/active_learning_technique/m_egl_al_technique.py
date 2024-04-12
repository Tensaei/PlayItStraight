import numpy
import torch
import src.support as support

from src.active_learning_technique.abstract_al_technique import AbstractALTechnique
from torch import nn

from src.al_dataset.dataset import Dataset


class ModifiedEGLALTechnique(AbstractALTechnique):

    def __init__(self, neural_network):
        self.neural_network = neural_network

    def select_samples(self, x, y, n_samples_to_select):
        criterion = torch.nn.CrossEntropyLoss()   #TODO

        sub_train_loader = torch.utils.data.DataLoader(Dataset((3, 32, 32), x, y), batch_size=support.model_batch_size)
        scores = []
        self.neural_network.eval()
        for x_batch, y_batch, _ in sub_train_loader:
            x_batch, y_batch = x_batch.to(support.device), y_batch.to(support.device)
            x_batch.requires_grad = True
            output = self.neural_network(x_batch)
            loss = criterion(output, y_batch)
            x_grad = torch.autograd.grad(loss, x_batch, retain_graph=True, create_graph=False)[0]
            grad_sum = numpy.linalg.norm(torch.Tensor.cpu(x_grad).detach().numpy().squeeze().reshape(len(x_grad), -1))
            scores.append(grad_sum)

        scores_sort = numpy.argsort(scores)
        scores_egl = [int(item) for item in scores_sort]
        select_loc = scores_egl[-n_samples_to_select:]
        selected_samples = [x[index] for index in select_loc]
        return selected_samples, None, None, None
