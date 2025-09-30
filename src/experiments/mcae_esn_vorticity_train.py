import sys
from pathlib import Path
from datetime import datetime
import pickle
import argparse
import json
import torch
import numpy as np
import einops
import matplotlib.pyplot as plt
sys.path.append("../")
sys.path.insert(0, "/home/eo821/Documents/EchoStateNetwork/")

from esn_preprocessing import generate_esn_noise_data, calculate_norm_mean
from esn.utils import errors, scalers
from esn.validation import validate
from esn.esn import ESN
from preprocessing.data_split import (
    normalize_data,
    load_from_h5file,
    train_valid_test_split,
)
from utils.config_tools import load_json_config, save_config
from plotting.esn import plot_esn_predictions
from CAE.mcae import MultiScaleCAE

def parse_args():
    parser = argparse.ArgumentParser(description="Train CAE-ESN on vorticity data.")
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to save experiment results and models."
    )
    parser.add_argument(
        "--kolmogorov_config",
        type=str,
        required=True,
        help="Path to Kolmogorov config JSON file."
    )
    parser.add_argument(
        "--esn_config",
        type=str,
        required=True,
        help="Path to ESN data parameters JSON file."
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

def error_metric(y, y_pred):
    return errors.mean_wasserstein_distance(y, y_pred) + errors.rel_L2(y, y_pred)

def main():
    args = parse_args()
    output_folder = Path(args.output_folder)
    kolmogorov_config = load_json_config(args.kolmogorov_config)
    esn_data_params = load_json_config(args.esn_config)
    kolmogorov_config["fln"] = args.fln

    # Set device
    if args.cuda_device.lower() == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.cuda_device}")

    # Load and normalize data
    U = load_from_h5file(kolmogorov_config, keyword="vorticity")
    U = einops.rearrange(U, "time x y -> time 1 x y")
    print(f"Vorticity data shape: {U.shape}")

    U_normalized, maxnorm = normalize_data(U, normtype=kolmogorov_config["normtype"])
    time_array_lyap = (
        np.arange(0, U.shape[0] * (kolmogorov_config["dt"] * kolmogorov_config["upsample"]),
                  step=(kolmogorov_config["dt"] * kolmogorov_config["upsample"]))
        / kolmogorov_config["max_lyap"]
    )
    del U

    # Timestamp for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    cae_path = output_folder
    esn_path = cae_path / "esn" / f"ESN_{timestamp}"
    imagepath = esn_path / "images"
    imagepath.mkdir(parents=True, exist_ok=True)
    save_config(esn_path / "experiment_setup.pkl", dict(kolmogorov_config, **esn_data_params))

    # Load CAE model
    cae_config = load_json_config(cae_path / "wandb_config.json")
    dim = cae_config["latent_size"]
    cae_model = MultiScaleCAE(cae_config["latent_size"], cae_config["kernel_size_list"])
    cae_model.load_state_dict(torch.load(cae_path / "best_model.pth", map_location=device))
    cae_model = cae_model.to(device)

    # Encode data
    U_encoded = cae_model.encode(torch.from_numpy(U_normalized).to(device))
    del U_normalized
    U_esn = generate_esn_noise_data(U_encoded.cpu().numpy(), 1, sigma_n=esn_data_params["noise_level"])
    del U_encoded

    # Split data
    u_encoded_train, u_encoded_valid, u_encoded_test = train_valid_test_split(U_esn, kolmogorov_config)
    washout_len = esn_data_params["N_washout"]
    train_len = min(len(U_esn) - washout_len, esn_data_params["train_size"])
    valid_len = min(len(U_esn) - washout_len, esn_data_params["valid_size"])

    U_washout = U_esn[:washout_len]
    U_train_input = U_esn[washout_len : washout_len + train_len - 1]
    U_train_label = U_esn[washout_len + 1 : washout_len + train_len]
    U_val_input = U_esn[train_len : train_len + valid_len - 1]
    U_val_label = U_esn[train_len + 1 : train_len + valid_len]
    U_test_washout = U_esn[train_len + valid_len : train_len + valid_len + washout_len]
    U_test = U_esn[train_len + valid_len + washout_len :]

    print(U_train_input.shape, U_val_input.shape, U_test.shape)

    # ESN dictionary
    ESN_dict = {
        "reservoir_size": esn_data_params["reservoir_size"],
        "dimension": dim,
        "reservoir_connectivity": esn_data_params["reservoir_connectivity"],
        "input_bias": esn_data_params["input_bias"],
        "output_bias": esn_data_params["output_bias"],
        "r2_mode": esn_data_params["r2_mode"],
        "reservoir_weights_mode": esn_data_params["reservoir_weights_mode"],
        "input_normalization": calculate_norm_mean(u_encoded_train, esn_data_params["normalization"]),
        "input_weights_mode": esn_data_params["input_weights_mode"],
        "input_seeds": esn_data_params["input_seeds"],
        "reservoir_seeds": esn_data_params["reservoir_seeds"],
    }

    # Hyperparameter search
    hyperparameter_dict = {
        "spectral_radius": esn_data_params["spectral_radius"],
        "input_scaling": esn_data_params["input_scaling"],
        "tikhonov": esn_data_params["tikhonov"],
    }
    min_dict = validate(
        grid_range=[getattr(scalers, params[-1])(params[:2]) for params in hyperparameter_dict.values()],
        param_names=list(hyperparameter_dict.keys()),
        param_scales=[values[-1] for values in hyperparameter_dict.values()],
        n_calls=esn_data_params["n_calls"],
        n_initial_points=esn_data_params["n_initial_points"],
        ESN_dict=ESN_dict,
        U_washout_train=U_washout,
        n_realisations=esn_data_params["n_realisations"],
        U_train=U_train_input,
        Y_train=U_train_label,
        U_val=U_val_input,
        Y_val=U_val_label,
        n_folds=esn_data_params["N_folds"],
        N_washout_steps=washout_len,
        N_val_steps=esn_data_params["N_val_steps"],
        random_seed=esn_data_params["random_seed"],
        error_measure=errors.rmse,
    )
    with open(esn_path / "best_dict.pkl", "wb") as f:
        pickle.dump(dict(ESN_dict, **min_dict), f)

    # Train and plot for best ESNs
    for j in range(3):
        my_ESN = ESN(
            reservoir_size=ESN_dict["reservoir_size"],
            dimension=ESN_dict["dimension"],
            reservoir_connectivity=ESN_dict["reservoir_connectivity"],
            spectral_radius=min_dict["spectral_radius"][j],
            input_scaling=min_dict["input_scaling"][j],
            tikhonov=min_dict["tikhonov"][j],
            input_bias=ESN_dict["input_bias"],
            output_bias=ESN_dict["output_bias"],
            r2_mode=ESN_dict["r2_mode"],
            reservoir_weights_mode=ESN_dict["reservoir_weights_mode"],
            input_normalization=ESN_dict["input_normalization"],
            input_weights_mode=ESN_dict["input_weights_mode"],
            input_seeds=ESN_dict["input_seeds"],
            reservoir_seeds=ESN_dict["reservoir_seeds"],
            verbose=False,
        )
        my_ESN.train(U_washout, U_train_input, U_train_label)

        # Plotting and evaluation
        plot_esn_predictions(
            my_ESN, cae_model, U_test, U_train_input, U_val_input, time_array_lyap,
            washout_len, dim, imagepath, j
        )

if __name__ == "__main__":
    main()


# python cae_esn_vorticity_train.py \
#   --output_folder /home/eo821/Documents/Kolmogorov-CAE-RNN/models/mcae/re40/full_data \
#   --kolmogorov_config /home/eo821/Documents/Kolmogorov-CAE-RNN/models/mcae/re40/full_data/kolmogorov.json \
#   --esn_config /home/eo821/Documents/Kolmogorov-CAE-RNN/models/mcae/re40/full_data/esn_data_params.json \
#   --fln /storage0/eo821/Kolmogorov/kolsol_RE40_01_t_40000_0005_upsampled_20.h5 \
#   --cuda_device 0