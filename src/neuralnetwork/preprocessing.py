import numpy as np

def normalize_data(U, normtype="max"):
    """
    Normalize the input data array
    Parameters:
    U (numpy.ndarray): input data (time, N_x, N_y, uv)
    normtype (str, optional): The normalization method to be used. Options are:
                             - 'max': Normalize by the maximum absolute value.
                             - 'maxmin': Normalize by the range between maximum and minimum values.
                             - 'meanstd': Normalize by subtracting mean and dividing by standard deviation.

    Returns depending on the chosen normtype:
                            - 'max': normalized data , maximum absolute value.
                            - 'maxmin': normalized data, maximum value, minimum value.
                            - 'meanstd': normalized data, mean value, standard deviation
    """

    if normtype == "max":
        U_max = np.amax(np.abs(U))
        return U / U_max, U_max
    elif normtype == "maxmin":
        U_max = np.amax(U)
        U_min = np.amin(U)
        norm = U_max - U_min
        return (U - U_min) / norm, U_max, U_min
    elif normtype == "meanstd":
        U_mean = np.mean(U)
        U_std = np.std(U)
        return (U - U_mean) / U_std, U_mean, U_std


def train_valid_test_split(U, kolmogorov_data):
    """
    Split the input data into training, validation, and test sets based on kolmogorov dictionary specifiying the split
    """

    # Calculate the number of samples for each split
    N_train = int(kolmogorov_data["N_data"] * kolmogorov_data["train_ratio"])
    N_valid = int(kolmogorov_data["N_data"] * kolmogorov_data["valid_ratio"])

    # Split the data into sets
    train_set = U[:N_train]
    valid_set = U[N_train : N_train + N_valid]
    test_set = U[N_train + N_valid :]

    return train_set, valid_set, test_set


def generate_esn_noise_data(U, norm, sigma_n=0.02, seed=0):
    U_normalized = U / norm
    rnd1 = np.random.RandomState(seed)
    data_std = np.std(U_normalized, axis=0)
    U_noise = rnd1.normal(0, sigma_n * data_std, size=U.shape)
    return U_normalized + U_noise