from src.active_learning_technique.abstract_al_technique import AbstractALTechnique


class RandomALTechnique(AbstractALTechnique):

    def select_samples(self, x, n_samples_to_select):
        if n_samples_to_select == -1:
            return x, [0] * len(x)

        return x[:n_samples_to_select], [0] * len(x)
