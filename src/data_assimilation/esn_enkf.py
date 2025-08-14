import numpy as np

def run_esn_ensemble(my_ESN, input_data, kalman_params, reservoir_size, washout=False):
    """
    Runs ESN closed-loop predictions for an ensemble.

    Parameters
    ----------
    my_ESN : object
        ESN model with `.closed_loop()` and `.closed_loop_with_washout()` methods.
    input_data : np.ndarray
        Input time series for each ensemble member. Shape: (N_ensemble, time_steps, input_dim).
    kalman_params : dict
        Dictionary containing:
            - "N_ensemble": number of ensemble members
            - "N_steps": number of closed-loop prediction steps
            - "latentdim": output latent space dimension
    reservoir_size : int
        Number of reservoir nodes in the ESN.
    washout : bool, optional
        If True, uses `.closed_loop_with_washout()`. Otherwise, `.closed_loop()` is used.

    Returns
    -------
    obs_ensemble_step : np.ndarray
        Reservoir states for each ensemble member after the first step.
        Shape: (N_ensemble, N_steps, reservoir_size)
    predictions : np.ndarray
        Predictions for each ensemble member after the first step.
        Shape: (N_ensemble, N_steps, latentdim)
    """
    N_ensemble = kalman_params["N_ensemble"]
    N_steps = kalman_params["N_steps"]
    latentdim = kalman_params["latentdim"]

    # Preallocate arrays
    obs_ensemble_step = np.zeros((N_ensemble, N_steps, reservoir_size))
    predictions = np.zeros((N_ensemble, N_steps, latentdim))

    # Select ESN method based on washout flag
    esn_method = my_ESN.closed_loop_with_washout if washout else my_ESN.closed_loop

    for i in range(N_ensemble):
        reservoir, prediction = esn_method(input_data[i], N_steps)
        obs_ensemble_step[i] = reservoir[1:]  # skip t=0
        predictions[i] = prediction[1:]

    return obs_ensemble_step, predictions
