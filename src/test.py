#!/usr/bin/env python
# boot_epoch_runner.py

import argparse
from codecarbon import EmissionsTracker
import torch
from src.play_it_stright import datasets, nets
from src.play_it_stright.support.rs2 import split_dataset_for_rs2
from src.play_it_stright.support.utils import train, test, get_optim_configurations
from src.play_it_stright.support.arguments import parser as base_parser
from torch.utils.data import DataLoader

def main():
    # Basispaser erweitern
    p = base_parser
    p.add_argument("--single_epoch", type=int, required=True,
                   help="Nur bis zu dieser Boot‐Epoch trainieren und dann abbrechen")
    args = p.parse_args()

    # Device
    if len(args.gpu) > 1:
        args.device = "cuda"
    elif len(args.gpu) == 1:
        args.device = f"cuda:{args.gpu[0]}"
    else:
        args.device = "cpu"

    # Datensatz & Model
    ch, im_size, nclass,_,_, dst_train,_, dst_test = datasets.__dict__[args.dataset](args)
    train_loader = DataLoader(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader  = DataLoader(dst_test,  batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    net = nets.__dict__[args.model](nclass).to(args.device)

    # RS2-Split
    splits = split_dataset_for_rs2(dst_train, args)

    # Optimizer/Scheduler
    crit, opt, _, rec = get_optim_configurations(args, net, train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=round(len(train_loader)/args.n_split)*args.epochs,
        eta_min=args.min_lr
    )

    # genau eine Boot-Epoch trainieren, Energie messen
    tracker = EmissionsTracker(save_to_file=False, log_level="ERROR")
    tracker.start()

    split = splits[(args.single_epoch-1) % len(splits)]
    train(split, net, crit, opt, scheduler, args.single_epoch-1, args, rec, if_weighted=False)

    tracker.stop()
    energy = tracker._emissions_data.energy_consumed  # in kWh

    # Test nach dieser Epoche
    acc, *_ = test(test_loader, net, crit, args.single_epoch-1, args, rec)

    # Ausgabe, die das PS1-Skript parsen kann
    print(f"ENERGY_KWH={energy:.6f}")
    print(f"ACCURACY={acc:.4f}")

if __name__=="__main__":
    main()
