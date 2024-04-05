import sys
import os
import time
import psutil
from torchvision.models.resnet import resnet50

import src.support as support
import torch
import threading

from torch import optim
from src.active_learner.simple_active_learner import SimpleActiveLearner, SelectionPolicy
from src.active_learning_technique.badge_al_technique import BadgeALTechnique
from src.active_learning_technique.lcs_al_technique import LCSALTechnique
from src.al_dataset.imagenet_al_dataset import ImageNetALDataset
from src.support import Reason, clprint
from src.active_learning_technique.random_al_technique import RandomALTechnique
from src.neural_networks.mnist_nn import MNIST_nn
from src.neural_networks.fashion_mnist_nn import Fashion_MNIST_nn
from src.neural_networks.cifar10_nn import Cifar10_nn
from src.al_dataset.mnist_al_dataset import MNISTALDataset
from src.al_dataset.fashion_mnist_al_dataset import FashionMNISTALDataset
from src.al_dataset.cifar10_al_dataset import Cifar10ALDataset


def plain_training(model, target_accuracy, incremental_training_epochs, criterion, optimizer, dataset):
    clprint("Loading dataset...", Reason.INFO_TRAINING)
    clprint("Starting training with all samples!", Reason.INFO_TRAINING, loggable=True)
    clprint("Plain training model...", Reason.INFO_TRAINING)
    total_epochs = 0
    accuracy = 0
    while accuracy < target_accuracy:
        start_time = support.get_time_in_millis()
        model.fit(incremental_training_epochs, criterion, optimizer, dataset.get_train_loader())
        elapsed_time = support.get_time_in_millis() - start_time
        clprint("Evaluating model...", Reason.INFO_TRAINING)
        loss, accuracy = model.evaluate(criterion, dataset.get_test_loader())
        clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time / 1000)), Reason.LIGHT_INFO_TRAINING, loggable=True)
        total_epochs += incremental_training_epochs

    clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time / 1000)), Reason.OTHER, loggable=True)
    return total_epochs


def cutted_training(model, training_epochs, criterion, optimizer, n_samples_at_start, n_samples_to_select, al_dataset, incremental_technique, target_accuracy, selection_policy, pool_size):
    clprint("Loading dataset...", Reason.INFO_TRAINING)
    clprint("Starting training with {} samples!".format(n_samples_at_start), Reason.INFO_TRAINING, loggable=True)
    clprint("Starting incremental training...", Reason.INFO_TRAINING)
    active_learner = SimpleActiveLearner(al_dataset, incremental_technique, target_accuracy, selection_policy, pool_size)
    start_time = support.get_time_in_millis()
    total_epochs = active_learner.elaborate(model, training_epochs, n_samples_to_select, criterion, optimizer)
    elapsed_time = support.get_time_in_millis() - start_time
    clprint("Evaluating model...", Reason.INFO_TRAINING)
    loss, accuracy = model.evaluate(criterion, al_dataset.get_test_loader())
    clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time / 1000)), Reason.OTHER, loggable=True)
    return total_epochs


def load_data_and_model(dataset_name, n_samples_at_start):
    if dataset_name == "mnist":
        dataset = MNISTALDataset(n_samples_at_start)
        clprint("Loading model...", Reason.INFO_TRAINING)
        model = MNIST_nn(support.device)
        criterion = None
        optimizer = optim.SGD(model.parameters(), lr=support.model_learning_rate, momentum=support.model_momentum)

    elif dataset_name == "fmnist":
        dataset = FashionMNISTALDataset(n_samples_at_start)
        clprint("Loading model...", Reason.INFO_TRAINING)
        model = Fashion_MNIST_nn(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate)

    elif dataset_name == "cifar10":
        dataset = Cifar10ALDataset(n_samples_at_start)
        model = Cifar10_nn(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate, betas=(0.9,0.995), weight_decay=5e-4)

    elif dataset_name == "imagenet":
        dataset = ImageNetALDataset(n_samples_at_start)
        model = resnet50(weights=None)
        model.to(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate)

    else:
        raise Exception("Unknown dataset!")

    return dataset, model, criterion, optimizer


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
    incremental_training_epochs = 5
    incremental_n_samples_at_start = 5000
    incremental_n_samples_to_select = 1000

    if len(sys.argv) < 8:
        clprint("You  should put these params:"
                "\n\t - dataset to test (mnist, fminst, cifar10, imagenet);"
                "\n\t - al technique (rnd, lcs, badge);"
                "\n\t - device (cpu, gpu [if available]);"
                "\n\t - target_accuracy (value between 0 and 100);"
                "\n\t - selection policy (l2, kl);"
                "\n\t - skip classic training (True, False);"
                "\n\t - pool size (value between n_classes and infinity).", Reason.WARNING)
        sys.exit()

    dataset_selected = sys.argv[1]
    al_technique_selected = sys.argv[2]
    device = sys.argv[3]
    target_accuracy = float(sys.argv[4])
    selection_policy = sys.argv[5]
    skip_classic_training = support.str2bool(sys.argv[6])
    pool_size = int(sys.argv[7])

    clprint("Dataset selected: {}".format(dataset_selected), Reason.INFO_TRAINING, loggable=True)

    if device == "cpu":
        support.device = "cpu"

    support.warm_up()

    monitoring_thread = threading.Thread(target=monitor_current_process)
    monitoring_thread.start()

    # Plain training
    counts = 0
    sum_cpu_percent = 0
    sum_memory_usage = 0
    start_time = support.get_time_in_millis()

    if not skip_classic_training:
        clprint("-"*100, Reason.INFO_TRAINING, loggable=True)
        clprint("Plain training configuration -> target accuracy: {}, n_samples: all".format(target_accuracy), Reason.INFO_TRAINING, loggable=True)
        clprint("Samples used: all", Reason.INFO_TRAINING, loggable=True)
        dataset, model, criterion, optimizer = load_data_and_model(dataset_selected, -1)
        total_epochs = plain_training(model, target_accuracy, incremental_training_epochs, criterion, optimizer, dataset)
        clprint("CPU usage: {}\nMemory usage: {}\nElapsed time: {} seconds\nEpochs elapsed: {}".format((sum_cpu_percent/counts), (sum_memory_usage/counts), int((support.get_time_in_millis() - start_time) / 1000), total_epochs), Reason.OTHER, loggable=True)

    else:
        clprint("Skipping plain training!", Reason.WARNING, loggable=True)

    # Incremental training
    counts = 0
    sum_cpu_percent = 0
    sum_memory_usage = 0
    start_time = support.get_time_in_millis()

    clprint("-" * 100, Reason.INFO_TRAINING, loggable=True)
    clprint("Incremental training configuration -> epochs: {}, n_start_samples: {}, n_select_samples: {}, target_accuracy: {}, pool_size: {}.".format(incremental_training_epochs, incremental_n_samples_at_start, incremental_n_samples_to_select, target_accuracy, pool_size), Reason.INFO_TRAINING, loggable=True)
    dataset, model, criterion, optimizer = load_data_and_model(dataset_selected, incremental_n_samples_at_start)

    if selection_policy == "l2":
        selection_policy = SelectionPolicy.L2

    elif selection_policy == "kl":
        selection_policy = SelectionPolicy.KL

    else:
        raise Exception("Unknown selection policy: {}!".format(selection_policy))

    if al_technique_selected == "rnd":
        incremental_technique = RandomALTechnique(model)

    elif al_technique_selected == "lcs":
        incremental_technique = LCSALTechnique(model)

    elif al_technique_selected == "badge":
        incremental_technique = BadgeALTechnique(model, dataset)

    clprint("AL technique selected: {}".format(sys.argv[2]), Reason.INFO_TRAINING, loggable=True)
    clprint("Incremental technique selected is {}!".format(incremental_technique.__class__.__name__), Reason.INFO_TRAINING, loggable=True)
    total_epochs = cutted_training(model, incremental_training_epochs, criterion, optimizer, incremental_n_samples_at_start, incremental_n_samples_to_select, dataset, incremental_technique, target_accuracy, selection_policy, pool_size)
    clprint("Samples used: {}".format(int(incremental_n_samples_at_start + (incremental_n_samples_to_select * (total_epochs/incremental_training_epochs)))), Reason.INFO_TRAINING, loggable=True)
    clprint("CPU usage: {}\nMemory usage: {}\nElapsed time: {} seconds\nEpochs elapsed: {}".format((sum_cpu_percent / counts), (sum_memory_usage / counts), int((support.get_time_in_millis() - start_time) / 1000), total_epochs), Reason.OTHER, loggable=True)

    completed = True
