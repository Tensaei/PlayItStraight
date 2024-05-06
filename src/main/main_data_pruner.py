import sys
import os
import time
import psutil
import src.support as support
import torch
import threading

from codecarbon import EmissionsTracker
from torchvision.models.resnet import resnet50
from torch import optim
from src.active_learner.simple_active_learner import SimpleActiveLearner, SelectionPolicy
from src.active_learning_technique.badge_al_technique import BadgeALTechnique
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
from src.neural_networks.cifar10_nn import Cifar10_nn, BasicBlock
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
        #clprint("Evaluating model...", Reason.INFO_TRAINING)
        #loss, accuracy = model.evaluate(criterion, dataset.get_test_loader())
        #clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time / 1000)), Reason.LIGHT_INFO_TRAINING, loggable=True)

    loss, accuracy = model.evaluate(criterion, dataset.get_test_loader())
    clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time / 1000)), Reason.OTHER, loggable=True)


def cutted_training(model, criterion, optimizer, n_samples_to_select, al_dataset, incremental_technique, target_epochs, step_training_epochs, selection_policy, scheduler, rs2_enabled):
    clprint("Loading dataset...", Reason.INFO_TRAINING)
    clprint("Starting incremental training...", Reason.INFO_TRAINING)
    active_learner = SimpleActiveLearner(al_dataset, incremental_technique, selection_policy)
    start_time = support.get_time_in_millis()
    active_learner.elaborate(model, target_epochs, step_training_epochs, n_samples_to_select, criterion, optimizer, scheduler, rs2_enabled)
    elapsed_time = support.get_time_in_millis() - start_time
    clprint("Evaluating model...", Reason.INFO_TRAINING)
    loss, accuracy = model.evaluate(criterion, al_dataset.get_test_loader())
    clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time / 1000)), Reason.OTHER, loggable=True)


def load_data_and_model(dataset_name, n_samples_at_start=0, rs2_enabled=False):
    if dataset_name == "mnist":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = MNISTALDataset(n_samples_at_start, rs2_enabled)
        clprint("Loading model...", Reason.INFO_TRAINING)
        model = MNIST_nn(support.device)
        criterion = None
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=support.model_momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_samples_at_start * 2 * 100, eta_min=1e-4)
        scheduler.last_epoch = n_samples_at_start * 2 * 99

    elif dataset_name == "fmnist":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = FashionMNISTALDataset(n_samples_at_start, rs2_enabled)
        clprint("Loading model...", Reason.INFO_TRAINING)
        model = Fashion_MNIST_nn(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate)

    elif dataset_name == "cifar10":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = Cifar10ALDataset(n_samples_at_start, rs2_enabled)
        model = Cifar10_nn(img_channels=3, block=BasicBlock, num_classes=10, device=support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate, betas=(0.9,0.995), weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_samples_at_start * 2 * 100, eta_min=1e-4)
        scheduler.last_epoch = n_samples_at_start * 2 * 99

    elif dataset_name == "cifar100":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = Cifar100ALDataset(n_samples_at_start, rs2_enabled)
        model = Cifar100_nn(3, 100, support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate)

    elif dataset_name == "imagenet":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = ImageNetALDataset(n_samples_at_start, rs2_enabled)
        model = resnet50(weights=None)
        model.to(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate)

    elif dataset_name == "tinyimagenet":
        clprint("Loading dataset...", Reason.INFO_TRAINING)
        dataset = TinyImageNetALDataset(n_samples_at_start, rs2_enabled)
        model = resnet50(weights=None)
        model.to(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate)

    else:
        raise Exception("Unknown dataset!")

    return dataset, model, criterion, optimizer, scheduler


counts = 0
sum_cpu_percent = 0
sum_memory_usage = 0
completed = False


def monitor_current_process(interval=1):
    pid = os.getpid()
    process = psutil.Process(pid)

    while True:
        with process.oneshot():
            cpu_percent = process.cpu_percent()
            memory_usage = process.memory_info().rss / (1024 * 1024)
            global counts
            counts += 1
            global sum_cpu_percent
            sum_cpu_percent += cpu_percent
            global sum_memory_usage
            sum_memory_usage += memory_usage
            global completed
            if completed:
                break

        time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) < 9:
        clprint("You  should put these params:"
                "\n\t - dataset to test (mnist, fminst, cifar10, cifar100, imagenet, tinyimagenet);"
                "\n\t - al technique (rnd, lcs, egl, entropy, badge);"
                "\n\t - device (cpu, gpu [if available]);"
                "\n\t - target epochs (value between 0 and infinity);"
                "\n\t - selection policy (l2, kl);"
                "\n\t - random seed (an integer value);"
                "\n\t - skip classic training (True, False)."
                "\n\t - rs2 enabled (to apply rs2 to the first phase of training)."
                "\n\t - n samples to select with AL.", Reason.WARNING)
        sys.exit()

    dataset_selected = sys.argv[1]
    al_technique_selected = sys.argv[2]
    device = sys.argv[3]
    target_epochs = int(sys.argv[4])
    selection_policy = sys.argv[5]
    random_seed = int(sys.argv[6])
    skip_classic_training = support.str2bool(sys.argv[7])
    rs2_enabled = support.str2bool(sys.argv[8])
    n_samples_to_select = int(sys.argv[9])
    step_training_epochs = 10

    clprint("Dataset selected: {}".format(dataset_selected), Reason.INFO_TRAINING, loggable=True)

    if device == "cpu":
        support.device = "cpu"

    support.random_seed = random_seed
    support.warm_up()

    monitoring_thread = threading.Thread(target=monitor_current_process)
    monitoring_thread.start()

    # Plain training
    counts = 0
    sum_cpu_percent = 0
    sum_memory_usage = 0
    start_time = support.get_time_in_millis()

    if not skip_classic_training:
        tracker = EmissionsTracker()
        tracker.start()
        clprint("-"*100, Reason.INFO_TRAINING, loggable=True)
        clprint("Plain training configuration -> target epochs: {}, n_samples: all".format(target_epochs), Reason.INFO_TRAINING, loggable=True)
        clprint("Samples used: all", Reason.INFO_TRAINING, loggable=True)
        dataset, model, criterion, optimizer, scheduler = load_data_and_model(dataset_selected, -1)
        plain_training(model, target_epochs, step_training_epochs, criterion, optimizer, dataset, scheduler)
        clprint("CPU usage: {}\nMemory usage: {}\nElapsed time: {} seconds".format((sum_cpu_percent/counts), (sum_memory_usage/counts), int((support.get_time_in_millis() - start_time) / 1000)), Reason.OTHER, loggable=True)
        tracker.stop()

    else:
        clprint("Skipping plain training!", Reason.WARNING, loggable=True)

    # Incremental training
    counts = 0
    sum_cpu_percent = 0
    sum_memory_usage = 0
    start_time = support.get_time_in_millis()

    clprint("-" * 100, Reason.INFO_TRAINING, loggable=True)
    clprint("AL one-shot training configuration -> epochs: {}, n_select_samples: {}.".format(target_epochs, n_samples_to_select), Reason.INFO_TRAINING, loggable=True)
    dataset, model, criterion, optimizer, scheduler = load_data_and_model(dataset_selected, rs2_enabled=rs2_enabled)

    if selection_policy == "l2":
        selection_policy = SelectionPolicy.L2

    elif selection_policy == "kl":
        selection_policy = SelectionPolicy.KL

    else:
        raise Exception("Unknown selection policy: {}!".format(selection_policy))

    if al_technique_selected == "rnd":
        incremental_technique = RandomALTechnique()

    elif al_technique_selected == "lcs":
        incremental_technique = LCSALTechnique(model)

    elif al_technique_selected == "egl":
        incremental_technique = EGLALTechnique(model, torch.nn.CrossEntropyLoss(), dataset.input_shape)

    elif al_technique_selected == "entropy":
        incremental_technique = EntropyALTechnique(model, torch.nn.CrossEntropyLoss(), dataset.input_shape)

    elif al_technique_selected == "badge":
        incremental_technique = BadgeALTechnique(model, dataset)

    tracker = EmissionsTracker()
    tracker.start()
    clprint("AL technique selected: {}".format(sys.argv[2]), Reason.INFO_TRAINING, loggable=True)
    clprint("Incremental technique selected is {}!".format(incremental_technique.__class__.__name__), Reason.INFO_TRAINING, loggable=True)
    cutted_training(model, criterion, optimizer, n_samples_to_select, dataset, incremental_technique, target_epochs, step_training_epochs, selection_policy, scheduler, rs2_enabled)
    clprint("Samples used in second phase: {}".format(n_samples_to_select), Reason.INFO_TRAINING, loggable=True)
    clprint("CPU usage: {}\nMemory usage: {}\nElapsed time: {} seconds\nEpochs elapsed: {}".format((sum_cpu_percent / counts), (sum_memory_usage / counts), int((support.get_time_in_millis() - start_time) / 1000), target_epochs), Reason.OTHER, loggable=True)
    tracker.stop()

    completed = True
