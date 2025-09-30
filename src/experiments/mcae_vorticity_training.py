import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchinfo
import einops
import wandb

sys.path.append("../")
from neuralnetwork.autoencoder.multi_scale_cae import MultiScaleCAE
from neuralnetwork.preprocessing import load_and_normalize_data, get_data_loaders
from utils.data_io import save_config, load_json_config
from preprocessing.data_split import (
    normalize_data,
    load_from_h5file,
    train_valid_test_split,
)
from neuralnetwork.earlystopping import EarlyStopper
from neuralnetwork.losses import LossTracker

# ------------------- Configuration -------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiScaleCAE on vorticity data.")
    parser.add_argument(
        "--model_folder",
        type=str,
        default="../models/mcae/re40/full_data",
        help="Path to save trained models and configs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="kolmogorov_config.json",
        help="Path to Kolmogorov config JSON file."
    )
    parser.add_argument(
        "--fln",
        type=str,
        required=True,
        help="Path to the Kolmogorov data file (HDF5)."
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="0",
        help="CUDA device index to use, e.g., '0' or '1'. Use 'cpu' to force CPU."
    )
    return parser.parse_args()

# ------------------- Utility Functions -------------------
def setup_model(config):
    """Initialize the MultiScale CAE model."""
    model = MultiScaleCAE(
        config["latent_size"],
        config["kernel_size_list"],
        weight_init_name=config["weight_init_name"],
    ).to(DEVICE)
    torchinfo.summary(model, input_size=(1, 1, KOLMOGOROV_CONFIG["Nx"], KOLMOGOROV_CONFIG["Ny"]))
    return model

# ------------------- Training Function -------------------
def train(model_folder):
    """Train the MultiScaleCAE model with sweep configuration."""
    wandb.login()
    config_defaults = {
        "latent_size": 128,
        "noise_level": 0.0,
        "batch_size": 128 * 20,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "epochs": 1000,
        "resolution": 128,
        "patience": 200,
        "architecture": "cae",
        "weight_init_name": "kaiming_uniform",
        "weighing_dissipation": 1.0,
        "kernel_size_list": [3, 5, 7],
    }
    wandb_run = wandb.init(config=config_defaults)
    print(f"WANDB sweep name: {wandb_run.name}")
    modelpath = Path(model_folder) / wandb_run.name
    modelpath.mkdir(parents=True, exist_ok=True)

    save_config(modelpath / "kolmogorov.json", KOLMOGOROV_CONFIG)
    save_config(modelpath / "wandb_config.json", dict(wandb_run.config))

    # Data preparation
    U_normalized, maxnorm = load_and_normalize_data(KOLMOGOROV_CONFIG)
    rng = np.random.default_rng(0)
    U_noise = rng.normal(scale=np.std(U_normalized), size=U_normalized.shape).astype(np.float32)
    U_noisy = U_normalized + wandb_run.config.noise_level * U_noise

    train_loader, valid_loader = get_data_loaders(U_noisy, KOLMOGOROV_CONFIG, wandb_run.config.batch_size)

    # Model, optimizer, scheduler, loss
    model = setup_model(wandb_run.config)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb_run.config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    criterion = nn.MSELoss()
    loss_tracker = LossTracker(len(train_loader), len(valid_loader))
    early_stopper = EarlyStopper(patience=wandb_run.config.patience, min_delta=1e-6)

    wandb.watch(model)
    best_model_state_dict = model.state_dict()

    for epoch in range(wandb_run.config.epochs):
        loss_tracker.set_start_time(time.time())
        loss_tracker.reset_current_loss()

        # Training
        model.train()
        for x_batch_train in train_loader:
            x_batch_train = x_batch_train.to(DEVICE)
            optimizer.zero_grad()
            _, output = model(x_batch_train)
            loss = criterion(output, x_batch_train)
            loss.backward()
            optimizer.step()
            loss_tracker.update_current_loss(
                "training", loss, loss, torch.tensor(0.0, device=DEVICE),
                torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE)
            )
        loss_tracker.print_current_loss(epoch, "training")

        # Validation
        loss_tracker.set_start_time(time.time())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            for x_batch_valid in valid_loader:
                x_batch_valid = x_batch_valid.to(DEVICE)
                _, output = model(x_batch_valid)
                loss = criterion(output, x_batch_valid)
                loss_tracker.update_current_loss(
                    "validation", loss, loss, torch.tensor(0.0, device=DEVICE),
                    torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE)
                )
        loss_tracker.print_current_loss(epoch, "validation")
        loss_tracker.calculate_and_store_average_losses()
        wandb.log({lt: loss_tracker.losses_dict[lt][-1] for lt in loss_tracker.loss_types})

        # Early stopping and checkpointing
        if loss_tracker.check_best_validation_loss():
            early_stopper.reset_counter()
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, modelpath / "best_model.pth")
            print(f"Saved best model at {modelpath}")
        if early_stopper.track(loss_tracker.get_current_validation_loss()):
            break

    loss_tracker.save_losses(path=modelpath)

# ------------------- Sweep Configuration -------------------
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "validation_loss", "goal": "minimize"},
    "parameters": {
        "latent_size": {"values": [48, 64, 96]},
        "noise_level": {"values": [0.0]},
        "batch_size": {"values": [128 * 16]},
        "learning_rate": {"values": [0.001]},
        "epochs": {"values": [1000]},
        "patience": {"values": [100]},
        "weight_init_name": {"values": ["kaiming_uniform"]},
        "weighing_dissipation": {"values": [0.0]},
        "Re": {"values": [40.0]},
    },
}

if __name__ == "__main__":
    args = parse_args()
    if args.cuda_device.lower() == "cpu" or not torch.cuda.is_available():
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(f"cuda:{args.cuda_device}")
    MODEL_FOLDER = Path(args.model_folder)
    KOLMOGOROV_CONFIG = load_json_config(args.config)
    KOLMOGOROV_CONFIG["fln"] = args.fln  # Inject the file path from the command line
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="MCAE")
    wandb.agent(sweep_id, function=lambda: train(MODEL_FOLDER), count=3)


#python mcae_vorticity_training.py --fln /path/to/data.h5 --cuda_device 1
# python mcae_vorticity_training.py --fln /path/to/data.h5 --cuda_device cpu