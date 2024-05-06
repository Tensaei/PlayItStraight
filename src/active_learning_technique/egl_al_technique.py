import numpy
import torch
import src.support as support

from src.active_learning_technique.abstract_al_technique import AbstractALTechnique
from torch import nn

from src.al_dataset.dataset import Dataset


class EGLALTechnique(AbstractALTechnique):

    def __init__(self, neural_network, criterion, shape_data):
        self.neural_network = neural_network
        self.criterion = criterion
        self.shape_data = shape_data

    def select_samples(self, x, y, n_samples_to_select):
        fun_soft = nn.Softmax(dim=1)
        sub_train_loader = torch.utils.data.DataLoader(Dataset(self.shape_data, x, y), batch_size=support.model_batch_size)
        scores = []
        outputs = []
        self.neural_network.eval()
        for x_batch, y_batch, _ in sub_train_loader:
            x_batch, y_batch = x_batch.to(support.device), y_batch.to(support.device)
            x_batch.requires_grad = True
            output = self.neural_network(x_batch)
            output_copy = fun_soft(torch.Tensor.cpu(output))
            grad_sum = numpy.zeros(len(x_batch))
            for label_index in range(10):
                y_likely = torch.zeros_like(y_batch) + label_index
                loss = self.criterion(output, y_likely)
                x_grad = torch.autograd.grad(loss, x_batch, retain_graph=True, create_graph=False)[0]
                grad_sum += numpy.linalg.norm(torch.Tensor.cpu(x_grad).detach().numpy().squeeze().reshape(len(x_grad), -1), axis=1) * torch.Tensor.cpu(output_copy[:, label_index]).detach().numpy()
            scores.extend(grad_sum)
            outputs.extend(output)

        self.neural_network.train()
        scores_sort = numpy.argsort(scores)
        scores_egl = [int(item) for item in scores_sort]
        select_loc = scores_egl[-n_samples_to_select*2:]
        selected_samples = [x[index] for index in select_loc]
        real_y = [y[index] for index in select_loc]
        model_y = [outputs[index] for index in select_loc]
        selected_scores = [scores_egl[index] for index in select_loc]
        return selected_samples, model_y, real_y, selected_scores
