import sys
import math
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import einops
import matplotlib.pyplot as plt

from assimilation import EnKF, calculate_observation_error_covariance
from ESN.esn_preprocessing import generate_esn_noise_data
from esn.esn import ESN
from preprocessing.data_split import (
    normalize_data,
    load_from_h5file,
    train_valid_test_split,
)
from utils.config_tools import load_config
from CAE.simpler_cae_architecture_128 import CAE
from plotting.colors import (
    color_true, color_unbias, color_bias, color_obs, color_b,
    colors_alpha, y_unbias_props, esn_ens_props, washout_ens_props
)
from plotting.enkf import (
    plot_measurements,
    plot_comparison,
    plot_dissipation,
)

device = "cpu"

def closest_factors(n):
    for i in range(int(math.sqrt(n)), 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n

def add_noise(data, std_dev, shape):
    return data + np.random.normal(loc=0, scale=std_dev, size=shape)

def flatten_vorticity(vorticity):
    return einops.rearrange(vorticity, "u v x y -> u v (x y)")

def main(folderpath, esn_path):
    # --- Load data and models ---
    folderpath = Path(folderpath)
    esn_path = Path(esn_path)

    sweep_name = "upbeat-sweep-1"
    cae_path = folderpath / sweep_name

    kolmogorov_data = load_config(cae_path / "kolmogorov.json")
    U = load_from_h5file(kolmogorov_data, keyword="vorticity")
    U = einops.rearrange(U, "time x y -> time 1 x y")
    print(f"Vorticity of the Kolmogorov flow U with shape {U.shape}")

    U_normalized, maxnorm = normalize_data(U, normtype=kolmogorov_data["normtype"])
    del U

    config = load_config(cae_path / "wandb_config.json")
    cae_model = CAE(config["latent_size"])
    cae_model.load_state_dict(torch.load(cae_path / "best_model.pth", map_location=device))
    cae_model = cae_model.to(device)
    U_encoded = cae_model.encoder(torch.from_numpy(U_normalized).to(device)).numpy(force=True)

    # --- Prepare ESN ---
    ESN_DATA_PARAMS = load_config(esn_path / "experiment_setup.pkl")
    dim = config["latent_size"]
    U_esn = generate_esn_noise_data(U_encoded, 1, sigma_n=ESN_DATA_PARAMS["noise_level"])
    u_encoded_train, u_encoded_valid, u_encoded_test = train_valid_test_split(U_esn, kolmogorov_data)
    N_washout = ESN_DATA_PARAMS["N_washout"]
    train_len = min(len(u_encoded_train) - N_washout, ESN_DATA_PARAMS["train_size"])
    valid_len = min(len(u_encoded_valid) - N_washout, ESN_DATA_PARAMS["valid_size"])

    U_washout = u_encoded_train[:N_washout]
    U_train_input = u_encoded_train[N_washout : N_washout + train_len - 1]
    U_train_label = u_encoded_train[N_washout + 1 : N_washout + train_len]
    U_val_input = u_encoded_valid[: valid_len - 1]
    U_val_label = u_encoded_valid[1:valid_len]
    U_test_washout = u_encoded_test[:N_washout]
    U_test = u_encoded_test[N_washout:]

    with open(esn_path / "best_dict.pkl", "rb") as f:
        esn_loaded_dict = pickle.load(f)

    my_ESN = ESN(
        reservoir_size=esn_loaded_dict["reservoir_size"],
        dimension=config["latent_size"],
        reservoir_connectivity=esn_loaded_dict["reservoir_connectivity"],
        spectral_radius=esn_loaded_dict["spectral_radius"][0],
        input_scaling=esn_loaded_dict["input_scaling"][0],
        tikhonov=esn_loaded_dict["tikhonov"][0],
        input_bias=esn_loaded_dict["input_bias"],
        output_bias=esn_loaded_dict["output_bias"],
        reservoir_weights_mode=esn_loaded_dict["reservoir_weights_mode"],
        input_weights_mode=esn_loaded_dict["input_weights_mode"],
        input_seeds=esn_loaded_dict["input_seeds"],
        reservoir_seeds=esn_loaded_dict["reservoir_seeds"],
        verbose=False,
    )
    my_ESN.train(U_washout, U_train_input, U_train_label)
    inputdim = U_train_input.shape[1]

    # --- Assimilation and plotting setup ---
    N_ensemble = 50
    repeats = 250
    N_steps = 10
    std_obs = 0.1
    np.random.seed(1)
    N_start = 0

    # Main experiment loop
    for sampling in [8, 16, 32, 64]:
        H, W = 48, 48
        dim_sampling = H * W // sampling
        h_sub, w_sub = closest_factors(dim_sampling)
        print(f"{sampling=}, {dim_sampling=}, subgrid: {h_sub}x{w_sub}")
        assert h_sub * w_sub == dim_sampling, "dim_sampling must be a perfect square"

        x_idx = np.linspace(0, H - 1, h_sub, dtype=int)
        y_idx = np.linspace(0, W - 1, w_sub, dtype=int)
        xx, yy = np.meshgrid(x_idx, y_idx, indexing='ij')
        selected_indices = xx * W + yy
        selected_columns = np.sort(-selected_indices.flatten())

        # Measurement operator
        M = np.zeros(shape=(dim_sampling, esn_loaded_dict["reservoir_size"] + 48*48))
        M[-dim_sampling:, selected_columns] = np.eye(dim_sampling)

        # Observation error covariance
        Cdd = 0.0001 * (std_obs * np.std(flatten_vorticity(U_normalized), axis=0)) ** 2 * np.eye(48*48)

        # Prepare initial ensemble
        washout_snapshots = U_normalized[N_start:N_start + N_washout]
        washout_encoded = cae_model.encoder(torch.from_numpy(washout_snapshots).float().to(device)).numpy(force=True)
        washout_noise_ensemble = add_noise(washout_encoded, std_obs * np.std(U_encoded, axis=0), (N_ensemble, N_washout, inputdim))

        # Prepare truth and observations
        U_valid_series = U_normalized
        truth = U_valid_series[N_start + N_washout : N_start + N_washout + repeats * N_steps * 2, :]
        true_encoded, true_vorticity = cae_model(torch.from_numpy(truth).float().to(device))
        true_encoded = true_encoded.numpy(force=True)
        true_vorticity = true_vorticity.numpy(force=True)
        vorticity_obs_ensemble = true_vorticity[:, 0, :] + np.random.normal(
            loc=0, scale=std_obs * np.std(U_valid_series, axis=0),
            size=(N_ensemble, true_vorticity.shape[0], true_vorticity.shape[-2], true_vorticity.shape[-1])
        )
        vorticity_obs_ensemble = flatten_vorticity(vorticity_obs_ensemble)


        # Function to add noise to observations
def add_noise(data, std_dev, shape):
    return data + np.random.normal(loc=0, scale=std_dev, size=shape)

def reservoir_with_bias(reservoir, bias_value):
    bias_column = np.ones((reservoir.shape[0], 1)) * bias_value
    return np.hstack((reservoir, bias_column))

def compute_measurement_idx(predictions, threshold=0.03):
    variances = np.sum(np.var(predictions, axis=(0)), axis=1)
    indices_above_threshold = np.where(variances > threshold)[0]
    return indices_above_threshold[0] if indices_above_threshold.size > 0 else len(predictions[0])

def flatten_vorticity(vorticity):
    return einops.rearrange(vorticity, "u v x y -> u v (x y)")

N_start = 3000

U_valid = generate_esn_noise_data(cae.encoder(torch.from_numpy(U_valid_series).float().to(device)).numpy(force=True), 1, sigma_n=ESN_DATA_PARAMS["noise_level"])

N_ensemble = 50
repeats= 250
N_steps = 10
 # Set seed for reproducibility
np.random.seed(1)
N_start = 0
washout_snapshots = U_valid_series[N_start:N_start + N_washout]
washout_encoded = np.ones(shape=(N_washout, inputdim)) * cae.encoder(torch.from_numpy(washout_snapshots).float().to(device)).numpy(force=True)[-1]
washout_encoded = cae.encoder(torch.from_numpy(washout_snapshots).float().to(device)).numpy(force=True)
std_obs = 0.01
washout_noise_ensemble = add_noise(washout_encoded, std_obs * np.std(U_encoded, axis=0), (N_ensemble, N_washout, inputdim))
# start_indices = np.random.randint(U_train_input.shape[0] - 5* N_washout, U_train_input.shape[0] - 1* N_washout, size=N_ensemble)
# washout_ensemble = np.array([U_train_input[start:start + N_washout] for start in start_indices])
# print(washout_ensemble.shape)

truth = U_valid[N_start + N_washout: N_start + N_washout+repeats*N_steps*2, :]
true_encoded, true_vorticity = cae(torch.from_numpy(U_valid_series[N_start + N_washout: N_start + N_washout+repeats*N_steps*2, :]).float().to(device))
true_encoded = true_encoded.numpy(force=True)
true_vorticity = true_vorticity.numpy(force=True)
std_obs=0.1
vorticity_obs_ensemble = true_vorticity[:, 0, :] + np.random.normal(loc=0, scale=std_obs * np.std(U_valid_series, axis=0), size=(N_ensemble, true_vorticity.shape[0], true_vorticity.shape[-2], true_vorticity.shape[-1]))
vorticity_obs_ensemble = flatten_vorticity(vorticity_obs_ensemble)


import math
def closest_factors(n):
    """Return integer pair (h, w) such that h*w = n and h ≈ w"""
    for i in range(int(math.sqrt(n)), 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n  # fallback (prime case)

for sampling in [8, 16, 32, 64]:
    np.random.seed()
    H, W = 48, 48
    dim_sampling = H * W // sampling  # total number of sensors

    # Get closest grid shape
    h_sub, w_sub = closest_factors(dim_sampling)
    print(f"{sampling=}, {dim_sampling=}, subgrid: {h_sub}x{w_sub}")
    assert h_sub*w_sub== dim_sampling, "dim_sampling must be a perfect square"


    # Generate evenly spaced grid indices in both directions
    x_idx = np.linspace(0, H - 1, h_sub, dtype=int)
    y_idx = np.linspace(0, W - 1, w_sub, dtype=int)

    # Mesh the grid
    xx, yy = np.meshgrid(x_idx, y_idx, indexing='ij')

    # Flatten and convert to 1D flattened array indices
    selected_indices = xx * W + yy  # shape (side, side)
    selected_columns = -sampling * (dim_sampling - np.arange(dim_sampling))[::-1]
    selected_columns = np.sort(-selected_indices.flatten())  # sorted decreasin

    # selected_columns = sorted(np.random.choice(np.arange(-dim_sampling * sampling, 0), size=dim_sampling, replace=False))
    M = np.zeros(shape=(dim_sampling,  esn_loaded_dict["reservoir_size"]+48*48))
    M[-dim_sampling:, selected_columns] = np.eye(dim_sampling)


    Cdd = (std_obs * np.std(flatten_vorticity(U_valid_series), axis=0)) ** 2 * np.eye(48*48) # calculate_observation_error_covariance(vorticity_obs_ensemble)


    reservoir_size = esn_loaded_dict["reservoir_size"]
    obs_ensemble = np.zeros((N_ensemble, N_steps, reservoir_size ))
    obs_ensemble_step = np.zeros((N_ensemble, N_steps, reservoir_size ))
    predictions = np.zeros((N_ensemble, N_steps, inputdim))
    # bias = np.repeat((my_ESN.W_out[esn_loaded_dict["reservoir_size"]:] * my_ESN.b_out), repeats=N_ensemble, axis=0).T
    obs_ensemble = np.zeros((N_ensemble, 1, reservoir_size ))
    predictions_at = np.zeros((N_ensemble, 1, inputdim))
    measurements = truth[0:1]
    locations = 0


    for i in range(N_ensemble):
        reservoir, prediction = my_ESN.closed_loop_with_washout( washout_noise_ensemble[i], N_steps)
        obs_ensemble_step[i] = reservoir[1:] #reservoir_with_bias(reservoir[1:], my_ESN.b_out)
        predictions[i] = prediction[1:]


    for j in range(repeats):

        measurement_idx = N_steps
        obs_ensemble = np.append(obs_ensemble, obs_ensemble_step[:, :measurement_idx], axis=1)
        predictions_at = np.append(predictions_at, predictions[:, :measurement_idx], axis=1)

        print(measurement_idx, obs_ensemble.shape[1] - 1)
        sample_loc =  obs_ensemble.shape[1] - 1
        measurements = np.append(measurements, truth[sample_loc:sample_loc+1], axis=0)
        locations = np.append(locations, sample_loc)

    
        res_dec_pred = np.append(obs_ensemble[:, -1, :esn_loaded_dict["reservoir_size"]].T, flatten_vorticity(cae.decoder(torch.from_numpy(predictions[:, -1, :]).float().to(device)).numpy(force=True)).T[:, 0, :], axis=0)
        #  observations_ensemble_new[:, sample_loc, :].T  - bias
        measurement_vorticity = vorticity_obs_ensemble[:,sample_loc, :].T[selected_columns, :]
        Aa = EnKF(Af=res_dec_pred, d=measurement_vorticity, Cdd=Cdd[selected_columns, :][:, selected_columns], M=M)

        for i in range(N_ensemble):
            reservoir, prediction = my_ESN.closed_loop(Aa[: esn_loaded_dict["reservoir_size"], i], N_steps)
            obs_ensemble_step[i] = reservoir[1:] #reservoir_with_bias(reservoir[1:], my_ESN.b_out) 
            predictions[i] = prediction[1:]

    measurement_idx = N_steps
    print(measurement_idx)
    obs_ensemble = np.append(obs_ensemble[:, 1:, :], obs_ensemble_step[:, :measurement_idx], axis=1)
    predictions_at = np.append(predictions_at[:, 1:, :], predictions[:, :measurement_idx], axis=1)
    measurements = measurements[1:]
    locations = locations[1:] + N_washout

    prediction = np.mean(predictions_at, axis=0)
    np.save(f"images/kolmogorov/v3/nr/prediction_nr_{sampling}_{dim_sampling}_{N_steps}_{repeats}_{N_ensemble}.npy", prediction)
    
    plot_measurements(U_test_series, N_washout, maxnorm, vort_measurements_sampled, locations, dt, sampling, dim_sampling, N_steps, repeats, N_ensemble, "images/kolmogorov/v3/nr")

    plot_comparison(reference, prediction_decoded, prediction_decoded_enkf, t_snapshot, dt, maxnorm, "images/kolmogorov/v3/nr", sampling, dim_sampling, N_steps, repeats, N_ensemble)

    plot_dissipation(reference, prediction_decoded, prediction_decoded_enkf, ks_cpu, dt, "images/kolmogorov/v3/nr", sampling, dim_sampling, N_steps, repeats, N_ensemble)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kolmogorov CAE-ESN EnKF experiment")
    parser.add_argument("--folderpath", type=str, required=True, help="Path to CAE experiment folder")
    parser.add_argument("--esn_path", type=str, required=True, help="Path to ESN experiment folder")
    args = parser.parse_args()
    main(args.folderpath, args.esn_path)