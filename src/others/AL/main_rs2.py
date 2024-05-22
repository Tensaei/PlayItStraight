# Python
import os
import random
import sys

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from codecarbon import EmissionsTracker

# Torchvison
import torchvision.transforms as T

# Utils
from tqdm import tqdm

from src.others.AL.rs2 import split_dataset_for_rs2
from src.others.AL.utils import *

# Custom
from arguments import parser
from ptflops import get_model_complexity_info
import nets
import datasets as datasets
import methods as methods

# Seed
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Main
if __name__ == '__main__':
    # Training settings
    args = parser.parse_args()
    cuda = ""
    if len(args.gpu) > 1:
        cuda = 'cuda'
    elif len(args.gpu) == 1:
        cuda = 'cuda:' + str(args.gpu[0])

    if args.dataset == 'ImageNet':
        args.device = cuda if torch.cuda.is_available() else 'cpu'
    else:
        args.device = cuda if torch.cuda.is_available() else 'cpu'
    print("args: ", args)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_u_all, dst_test = datasets.__dict__[args.dataset](args)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
    print("im_size: ", dst_train[0][0].shape)

    #dst_subset = torch.utils.data.Subset(dst_train, indices)
    #print("Initial set size: ", len(dst_subset))

    # BackgroundGenerator for ImageNet to speed up dataloaders
    if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
        train_loader = DataLoaderX(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        test_loader = DataLoaderX(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    else:
        train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # Get Model
    print("| Training on model %s" % args.model)
    network = get_model(args, nets, args.model)

    macs, params = get_model_complexity_info(network, (channel, im_size[0], im_size[1]), as_strings=True, print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    tracker = EmissionsTracker()
    tracker.start()

    # RS2 boot training
    print("====================RS2 boot training====================")
    print('RS2 split size: {}'.format(int(len(dst_train) / args.n_split)))

    print("Epochs: {}".format(args.epochs))  #########
    #target_epochs = int(args.epochs / args.n_split)
    #splits_for_rs2 = split_dataset_for_rs2(dst_train, args)
    #criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, round(len(train_loader)/args.n_split) * target_epochs, eta_min=args.min_lr)
    # epoch = 0
    # accs = []
    # precs = []
    # recs = []
    # f1s = []
    # for i in range(target_epochs):
    #     train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
    #     indices = list(range(len(dst_train)))
    #     random.shuffle(indices)
    #     dst_shuffled = torch.utils.data.Subset(dst_train, indices)
    #     train_loader = torch.utils.data.DataLoader(dst_shuffled, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
    #     accuracy, precision, recall, f1 = test(test_loader, network, criterion, i, args, rec)
    #     accs.append([accuracy])
    #     precs.append([precision])
    #     recs.append([recall])
    #     f1s.append([f1])
    #     clprint('Training epoch {}/{} | Accuracy: {}, Precision: {}, Recall: {}, F1: {}'.format(i + 1, target_epochs, accuracy, precision, recall, f1), reason=Reason.OUTPUT_TRAINING)
    #
    # clprint('Boot completed | Accuracy: {}, Precision: {}, Recall: {}, F1: {}'.format(accuracy, precision, recall, f1), reason=Reason.OTHER)
    # print("Accuracies:")
    # print(accs)
    # print("Precisions:")
    # print(precs)
    # print("Recalls:")
    # print(recs)
    # print("F1s:")
    # print(f1s)


    splits_for_rs2 = split_dataset_for_rs2(dst_train, args)
    criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, round(len(train_loader)/args.n_split) * args.epochs, eta_min=args.min_lr)
    epoch = 0
    accs = []
    precs = []
    recs = []
    f1s = []
    logs = []
    while epoch < args.epochs:
        for split in splits_for_rs2:
            print("Performing RS2 training epoch n.{}".format(epoch + 1))
            train(split, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)
            epoch += 1

            if epoch % 10 == 0:
                accuracy, precision, recall, f1 = test(test_loader, network, criterion, epoch, args, rec)
                accs.append([accuracy])
                precs.append([precision])
                recs.append([recall])
                f1s.append([f1])
                clprint('Training epoch {}/{} | Accuracy: {}, Precision: {}, Recall: {}, F1: {}'.format(epoch, args.epochs, accuracy, precision, recall, f1), reason=Reason.OUTPUT_TRAINING)

        print("Finished splits, reshuffling data and resplitting!")
        splits_for_rs2 = split_dataset_for_rs2(dst_train, args)

    clprint('Boot completed | Accuracy: {}, Precision: {}, Recall: {}, F1: {}'.format(accuracy, precision, recall, f1), reason=Reason.OTHER)
    print("Accuracies:")
    print(accs)
    print("Precisions:")
    print(precs)
    print("Recalls:")
    print(recs)
    print("F1s:")
    print(f1s)

    #acc = test(test_loader, network, criterion, epoch, args, rec)
    #print('Completed || Label set size {}: Test acc {}'.format(len(dst_train), acc))

    #print("Final acc logs")
    #logs = np.array(logs).reshape((-1, 1))
    #print(logs, flush=True)
    tracker.stop()
