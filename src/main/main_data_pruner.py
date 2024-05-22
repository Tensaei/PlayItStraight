import sys
import src.support as support
import torch

from codecarbon import EmissionsTracker
from torchvision.models.resnet import resnet50
from torch import optim
from src.active_learner.simple_active_learner import SimpleActiveLearner, SelectionPolicy
from src.active_learning_technique.entropy_al_technique import EntropyALTechnique
from src.active_learning_technique.lcs_al_technique import LCSALTechnique
from src.active_learning_technique.egl_al_technique import EGLALTechnique
from src.al_dataset.cifar100_al_dataset import Cifar100ALDataset
from src.al_dataset.imagenet_al_dataset import ImageNetALDataset
from src.al_dataset.tiny_imagenet_al_dataset import TinyImageNetALDataset
from src.neural_networks.cifar100_nn import Cifar100_nn
from src.support import Reason, clprint
from src.active_learning_technique.random_al_technique import RandomALTechnique
from src.neural_networks.mnist_nn import MNIST_nn
from src.neural_networks.fashion_mnist_nn import Fashion_MNIST_nn
from src.neural_networks.cifar10_nn import Cifar10_nn
from src.al_dataset.mnist_al_dataset import MNISTALDataset
from src.al_dataset.fashion_mnist_al_dataset import FashionMNISTALDataset
from src.al_dataset.cifar10_al_dataset import Cifar10ALDataset


def plain_training(model, target_epochs, step_training_epochs, criterion, optimizer, dataset, scheduler=None):
    clprint("Loading dataset...", Reason.INFO_TRAINING)
    clprint("Starting training with all samples!", Reason.INFO_TRAINING, loggable=True)
    clprint("Plain training model...", Reason.INFO_TRAINING)
    accuracy = 0
    for epoch in range(0, target_epochs, step_training_epochs):
        start_time = support.get_time_in_millis()
        model.fit(step_training_epochs, criterion, optimizer, dataset.get_train_loader(), scheduler)
        elapsed_time = support.get_time_in_millis() - start_time
        clprint("Evaluating model...", Reason.INFO_TRAINING)
        loss, accuracy = model.evaluate(criterion, dataset.get_test_loader())
        clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time / 1000)), Reason.LIGHT_INFO_TRAINING, loggable=True)

    loss, accuracy = model.evaluate(criterion, dataset.get_test_loader())
    clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time / 1000)), Reason.OTHER, loggable=True)


def cutted_training(model, criterion, optimizer, n_samples_to_select, al_dataset, incremental_technique, target_epochs_phase_1, target_epochs_phase_2, step_training_epochs, selection_policy, scheduler):
    clprint("Loading dataset...", Reason.INFO_TRAINING)
    clprint("Starting incremental training...", Reason.INFO_TRAINING)
    active_learner = SimpleActiveLearner(al_dataset, incremental_technique, selection_policy)
    start_time = support.get_time_in_millis()
    active_learner.elaborate(model, target_epochs_phase_1, target_epochs_phase_2, step_training_epochs, n_samples_to_select, criterion, optimizer, scheduler, True)
    elapsed_time = support.get_time_in_millis() - start_time
    clprint("Evaluating model...", Reason.INFO_TRAINING)
    loss, accuracy = model.evaluate(criterion, al_dataset.get_test_loader())
    clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time / 1000)), Reason.OTHER, loggable=True)


def load_data_and_model(dataset_name, n_samples_to_select):
    if n_samples_to_select == -1:
        dataset_quantity = -1

    else:
        dataset_quantity = 0

    if dataset_name == "mnist":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = MNISTALDataset(dataset_quantity, True)
        clprint("Loading model...", Reason.INFO_TRAINING)
        model = MNIST_nn(support.device)
        criterion = None
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=support.model_momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_samples_to_select * 2 * 100, eta_min=1e-4)
        scheduler.last_epoch = n_samples_to_select * 2 * 99

    elif dataset_name == "fmnist":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = FashionMNISTALDataset(dataset_quantity, True)
        clprint("Loading model...", Reason.INFO_TRAINING)
        model = Fashion_MNIST_nn(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate)

    elif dataset_name == "cifar10":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = Cifar10ALDataset(dataset_quantity, True)
        model = Cifar10_nn(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        #optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate, betas=(0.9, 0.995), weight_decay=5e-4)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_samples_to_select * 2 * 100, eta_min=1e-4)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50000 * 100 / support.model_batch_size,  eta_min=1e-4)
        #scheduler.last_epoch = n_samples_to_select * 2 * 99
        #scheduler = None

    elif dataset_name == "cifar100":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = Cifar100ALDataset(dataset_quantity, True)
        model = Cifar100_nn(3, 100, support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate)

    elif dataset_name == "imagenet":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = ImageNetALDataset(dataset_quantity, True)
        model = resnet50(weights=None)
        model.to(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate)

    elif dataset_name == "tinyimagenet":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = TinyImageNetALDataset(dataset_quantity, True)
        model = resnet50(weights=None)
        model.to(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate)

    else:
        raise Exception("Unknown dataset!")

    return dataset, model, criterion, optimizer, scheduler


if __name__ == "__main__":
    if len(sys.argv) < 9:
        clprint("You should put these params:"
                "\n\t - dataset to test (mnist, fminst, cifar10, cifar100, imagenet, tinyimagenet);"
                "\n\t - al technique (rnd, lcs, egl, entropy);"
                "\n\t - device (cpu, gpu [if available]);"
                "\n\t - phase 1 target epochs (value between 0 and infinity);"
                "\n\t - phase 2 target epochs (value between 0 and infinity);"
                "\n\t - selection policy (l2, kl);"
                "\n\t - random seed (an integer value);"
                "\n\t - skip classic training (True, False);"
                "\n\t - n samples to select with AL(an integer value).", Reason.WARNING)
        sys.exit()

    dataset_selected = sys.argv[1]
    al_technique_selected = sys.argv[2]
    device = sys.argv[3]
    target_epochs_phase_1 = int(sys.argv[4])
    target_epochs_phase_2 = int(sys.argv[5])
    selection_policy = sys.argv[6]
    random_seed = int(sys.argv[7])
    skip_classic_training = support.str2bool(sys.argv[8])
    n_samples_to_select = int(sys.argv[9])
    step_training_epochs = 10

    clprint("Dataset selected: {}".format(dataset_selected), Reason.INFO_TRAINING, loggable=True)

    if device == "cpu":
        support.device = "cpu"

    support.random_seed = random_seed
    support.warm_up()

    # Plain training
    counts = 0
    sum_cpu_percent = 0
    sum_memory_usage = 0
    start_time = support.get_time_in_millis()

    if not skip_classic_training:
        total_epochs = target_epochs_phase_1 + target_epochs_phase_2
        tracker = EmissionsTracker()
        tracker.start()
        clprint("-"*100, Reason.INFO_TRAINING, loggable=True)
        clprint("Plain training configuration -> target epochs: {}, n_samples: all".format(total_epochs), Reason.INFO_TRAINING, loggable=True)
        clprint("Samples used: all", Reason.INFO_TRAINING, loggable=True)
        dataset, model, criterion, optimizer, scheduler = load_data_and_model(dataset_selected, -1)
        plain_training(model, (total_epochs), step_training_epochs, criterion, optimizer, dataset, scheduler)
        tracker.stop()

    else:
        clprint("Skipping plain training!", Reason.WARNING, loggable=True)

    # Incremental training
    counts = 0
    sum_cpu_percent = 0
    sum_memory_usage = 0
    start_time = support.get_time_in_millis()

    clprint("-" * 100, Reason.INFO_TRAINING, loggable=True)
    clprint("AL one-shot training configuration -> epochs: {}+{}, n samples to select: {}".format(target_epochs_phase_1, target_epochs_phase_2, n_samples_to_select), Reason.INFO_TRAINING, loggable=True)
    dataset, model, criterion, optimizer, scheduler = load_data_and_model(dataset_selected, n_samples_to_select)

    if selection_policy == "l2":
        selection_policy = SelectionPolicy.L2

    elif selection_policy == "kl":
        selection_policy = SelectionPolicy.KL

    else:
        raise Exception("Unknown selection policy: {}!".format(selection_policy))

    if al_technique_selected == "rnd":
        incremental_technique = RandomALTechnique()

    elif al_technique_selected == "lcs":
        incremental_technique = LCSALTechnique(model, dataset.input_shape)

    elif al_technique_selected == "egl":
        incremental_technique = EGLALTechnique(model, torch.nn.CrossEntropyLoss(), dataset.input_shape)

    elif al_technique_selected == "entropy":
        incremental_technique = EntropyALTechnique(model, torch.nn.CrossEntropyLoss(), dataset.input_shape)

    tracker = EmissionsTracker()
    tracker.start()
    clprint("AL technique selected: {}".format(sys.argv[2]), Reason.INFO_TRAINING, loggable=True)
    clprint("Incremental technique selected is {}!".format(incremental_technique.__class__.__name__), Reason.INFO_TRAINING, loggable=True)
    cutted_training(model, criterion, optimizer, n_samples_to_select, dataset, incremental_technique, target_epochs_phase_1, target_epochs_phase_2, step_training_epochs, selection_policy, scheduler)
    clprint("Samples used in second phase: {}".format(n_samples_to_select), Reason.INFO_TRAINING, loggable=True)
    clprint("CPU usage: {}\nMemory usage: {}\nElapsed time: {} seconds\nEpochs elapsed: {}+{}".format((sum_cpu_percent / counts), (sum_memory_usage / counts), int((support.get_time_in_millis() - start_time) / 1000), target_epochs_phase_1, target_epochs_phase_2), Reason.OTHER, loggable=True)
    tracker.stop()
