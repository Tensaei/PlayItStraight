import sys
import os
import time
import psutil
import src.support as support
import torch
import threading

from torch import optim
from src.active_learner.simple_active_learner import SimpleActiveLearner
from src.active_learning_technique.bait_al_technique import BaitALTechnique
from src.active_learning_technique.qbc_al_technique import QueryByCommiteeALTechnique
from src.active_learning_technique.lcs_al_technique import LCSALTechnique
from src.active_learning_technique.query_by_committee.decision_tree import DecisionTree
from src.active_learning_technique.query_by_committee.kvae import Kvae
from src.active_learning_technique.query_by_committee.svm import SVM
from src.active_learning_technique.query_by_committee.random_forest import RandomForest
from src.support import Reason, clprint
from src.active_learning_technique.random_al_technique import RandomALTechnique
from src.neural_networks.mnist_nn import MNIST_nn
from src.neural_networks.fashion_mnist_nn import Fashion_MNIST_nn
from src.neural_networks.cifar10_nn import Cifar10_nn
from src.al_dataset.mnist_al_dataset import MNISTALDataset
from src.al_dataset.fashion_mnist_al_dataset import FashionMNISTALDataset
from src.al_dataset.cifar10_al_dataset import Cifar10ALDataset


def plain_training(model, incremental_training_steps, incremental_training_epochs, criterion, optimizer, n_samples_at_start, dataset_class):
    clprint("Loading dataset...", Reason.INFO_TRAINING)
    clprint("Starting training with all samples!".format(n_samples_at_start), Reason.INFO_TRAINING, loggable=True)
    dataset = dataset_class(n_samples_at_start)
    clprint("Plain training model...", Reason.INFO_TRAINING)
    for i in range(incremental_training_steps):
        start_time = support.get_time_in_millis()
        model.fit(incremental_training_epochs, criterion, optimizer, dataset.get_train_loader())
        elapsed_time = support.get_time_in_millis() - start_time
        clprint("Evaluating model...", Reason.INFO_TRAINING)
        loss, accuracy = model.evaluate(criterion, dataset.get_test_loader())
        if i == incremental_training_steps - 1:
            reason = Reason.OTHER

        else:
            reason = Reason.LIGHT_INFO_TRAINING

        clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time/1000)), reason, loggable=True)


def cutted_training(model, training_epochs, criterion, optimizer, n_samples_at_start, n_samples_to_select, training_steps, dataset_class, incremental_technique):
    clprint("Loading dataset...", Reason.INFO_TRAINING)
    clprint("Starting training with {} samples!".format(n_samples_at_start), Reason.INFO_TRAINING, loggable=True)
    al_dataset = dataset_class(n_samples_at_start)
    clprint("Starting incremental training...", Reason.INFO_TRAINING)
    active_learner = SimpleActiveLearner(al_dataset, incremental_technique)
    start_time = support.get_time_in_millis()
    active_learner.elaborate(model, training_steps, training_epochs, n_samples_to_select, criterion, optimizer)
    elapsed_time = support.get_time_in_millis() - start_time
    clprint("Evaluating model...", Reason.INFO_TRAINING)
    loss, accuracy = model.evaluate(criterion, al_dataset.get_test_loader())
    clprint("Loss: {}\nAccuracy: {}\nReached in {} seconds".format(loss, accuracy, int(elapsed_time/1000)), Reason.OTHER, loggable=True)


def load_data_and_model(param):
    if param == "mnist":
        dataset_class = MNISTALDataset
        clprint("Loading model...", Reason.INFO_TRAINING)
        model = MNIST_nn(support.device)
        criterion = None
        optimizer = optim.SGD(model.parameters(), lr=support.model_learning_rate, momentum=support.model_momentum)

    elif param == "fmnist":
        dataset_class = FashionMNISTALDataset
        clprint("Loading model...", Reason.INFO_TRAINING)
        model = Fashion_MNIST_nn(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=support.model_learning_rate)

    elif param == "cifar10":
        dataset_class = Cifar10ALDataset
        model = Cifar10_nn(support.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=support.model_learning_rate, momentum=support.model_momentum)
    
    else:
        raise Exception("Unknow dataset!")
    
    return dataset_class, model, criterion, optimizer


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
    incremental_training_steps = 10
    incremental_training_epochs = 10
    incremental_n_samples_at_start = 1000
    incremental_n_samples_to_select = 1000

    plain_n_samples_at_start = 60000

    if len(sys.argv) < 4:
        clprint("You should put three params:\n\t - dataset to test (mnist, fminst, cifar10);\n\t - al technique (rnd, lcs);\n\t - device (cpu, gpu [if available]).", Reason.WARNING)
        sys.exit()

    clprint("Dataset selected: {}".format(sys.argv[1]), Reason.INFO_TRAINING, loggable=True)

    if sys.argv[3] == "cpu":
        support.device = "cpu"

    support.warm_up()

    monitoring_thread = threading.Thread(target=monitor_current_process)
    monitoring_thread.start()

    # Plain training
    counts = 0
    sum_cpu_percent = 0
    sum_memory_usage = 0
    start_time = support.get_time_in_millis()

    clprint("-"*100, Reason.INFO_TRAINING, loggable=True)
    clprint("Plain training configuration -> epochs: {}, n_samples: {}".format(incremental_training_steps * incremental_training_epochs, plain_n_samples_at_start), Reason.INFO_TRAINING, loggable=True)
    clprint("Samples used: {}".format(plain_n_samples_at_start), Reason.INFO_TRAINING, loggable=True)
    dataset_class, model, criterion, optimizer = load_data_and_model(sys.argv[1])
    plain_training(model, incremental_training_steps, incremental_training_epochs, criterion, optimizer, plain_n_samples_at_start, dataset_class)
    clprint("CPU usage: {}\nMemory usage: {}\nElapsed time: {}".format((sum_cpu_percent/counts), (sum_memory_usage/counts), (support.get_time_in_millis() - start_time)), Reason.OTHER, loggable=True)

    # Incremental training
    counts = 0
    sum_cpu_percent = 0
    sum_memory_usage = 0
    start_time = support.get_time_in_millis()

    clprint("-"*100, Reason.INFO_TRAINING, loggable=True)
    clprint("Incremental training configuration -> epochs: {}, steps, {}, n_start_samples: {}, n_select_samples: {}".format(incremental_training_epochs, incremental_training_steps, incremental_n_samples_at_start, incremental_n_samples_to_select), Reason.INFO_TRAINING, loggable=True)
    clprint("Samples used: {}".format((incremental_n_samples_at_start + (incremental_n_samples_to_select * incremental_training_steps))), Reason.INFO_TRAINING, loggable=True)
    dataset_class, model, criterion, optimizer = load_data_and_model(sys.argv[1])

    if sys.argv[2] == "rnd":
        incremental_technique = RandomALTechnique()
    
    elif sys.argv[2] == "lcs":
        incremental_technique = LCSALTechnique(model)

    clprint("AL technique selected: {}".format(sys.argv[2]), Reason.INFO_TRAINING, loggable=True)
    clprint("Incremental technique selected is {}!".format(incremental_technique.__class__.__name__), Reason.INFO_TRAINING, loggable=True)
    cutted_training(model, incremental_training_epochs, criterion, optimizer, incremental_n_samples_at_start, incremental_n_samples_to_select, incremental_training_steps, dataset_class, incremental_technique)
    clprint("CPU usage: {}\nMemory usage: {}\nElapsed time: {}".format((sum_cpu_percent/counts), (sum_memory_usage/counts), (support.get_time_in_millis() - start_time)), Reason.OTHER, loggable=True)

    completed = True
