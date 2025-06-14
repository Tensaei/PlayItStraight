import csv
import random
import torch
import numpy as np
from codecarbon import EmissionsTracker
from src.play_it_stright import datasets, methods, nets
from src.play_it_stright.support.rs2 import split_dataset_for_rs2
from src.play_it_stright.support.utils import train, test, get_optim_configurations
from src.play_it_stright.support.arguments import parser
from ptflops import get_model_complexity_info
from torch.utils.data import DataLoader

# 1) Argumente parsen (wie gehabt)
args = parser.parse_args()
# setzt device
cuda = "cuda" if len(args.gpu) > 1 else f"cuda:{args.gpu[0]}" if len(args.gpu)==1 else "cpu"
args.device = cuda if torch.cuda.is_available() else "cpu"

# 2) Daten & Modell initialisieren
channel, im_size, num_classes, class_names, mean, std, \
    dst_train, dst_u_all, dst_test = datasets.__dict__[args.dataset](args)
train_loader = DataLoader(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
test_loader  = DataLoader(dst_test,  batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
network = nets.__dict__[args.model](num_classes)  # oder wie Du get_model benutzt
network.to(args.device)

# 3) CSV öffnen
with open("boot_power_accuracy.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["metric","epoch","energy_kWh","cum_energy_kWh","accuracy"])

    # 4) für jede Metrik
    for metric in ["Margin","Entropy","LeastConfidence"]:
        args.uncertainty = metric
        cum_energy = 0.0

        # RS2-Splits nur einmal vorbereiten
        splits = split_dataset_for_rs2(dst_train, args)

        # Optim/ Scheduler einmal initialisieren
        criterion, optimizer, _, rec = get_optim_configurations(args, network, train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=round(len(train_loader)/args.n_split)*args.epochs,
            eta_min=args.min_lr
        )

        # 5) Boot-Epoch-Schleife
        epoch = 0
        while epoch < args.boot_epochs:
            # wir gehen jeweils durch genau einen Split und zählen das als 1 Boot-Epoch
            split = splits[epoch % len(splits)]
            tracker = EmissionsTracker(  # neuer Tracker pro Epoche
                save_to_file=False,  # verhindert CSV-Log von CodeCarbon
                log_level="ERROR"
            )
            tracker.start()

            # TRAIN one boot-epoch
            train(split, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)

            tracker.stop()
            # CodeCarbon speichert intern in tracker._emissions_data
            energy = tracker._emissions_data.energy_consumed  # kWh
            cum_energy += energy

            # TEST nach jeder Boot-Epoche
            accuracy, _, _, _ = test(test_loader, network, criterion, epoch, args, rec)

            # in CSV schreiben
            writer.writerow([metric, epoch+1, f"{energy:.6f}", f"{cum_energy:.6f}", f"{accuracy:.4f}"])
            print(f"[{metric}] Epoch {epoch+1:2d}  energy={energy:.4f}kWh  cum={cum_energy:.4f}kWh  acc={accuracy:.2%}")

            epoch += 1

        # Ende Metric-Loop, ggf. Modell zurücksetzen
        # network.apply(reset_weights)  # falls Du jedes Mal mit untrainiertem Netz starten willst
