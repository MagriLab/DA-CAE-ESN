import numpy as np
 
def EnKF(Af, d, Cdd, M):
    """Ensemble Kalman Filter as derived in Evensen (2009) eq. 9.27.
        Inputs:
            Af: forecast ensemble at time t
            d: observation at time t
            Cdd: observation error covariance matrix
            M: matrix mapping from state to observation space
        Returns:
            Aa: analysis ensemble (or Af is Aa is not real)
    """
    m = Af.shape[1]
 
    psi_f_m = np.mean(Af, 1, keepdims=True)
    Psi_f = Af - psi_f_m
    C_psipsi = Psi_f @ Psi_f.T
    # Create an ensemble of observations
    if d.ndim == 2 and d.shape[-1] == m:
        D = d
    else:
        D = np.random.rng.multivariate_normal(d, Cdd, m).transpose()
 
    # Mapped forecast matrix M(Af) and mapped deviations M(Af')
    M_Af = M@Af
    # S = np.dot(M, Psi_f)
 
    # Matrix to invert
    C = M@C_psipsi@M.T + Cdd
    Cinv = np.linalg.inv(C)
 
    K = C_psipsi@M.T@Cinv
    Aa = Af + K@(D - M_Af)
 
    if not np.isreal(Aa).all():
        Aa = Af
        print('Aa not real')
        
    return Aa


def calculate_observation_error_covariance(observations_ensemble):
    """
    Calculate the observation error covariance matrix from an ensemble of observations.
    
    Parameters:
    observations_ensemble (numpy.ndarray): The ensemble of observations with shape (N_ensemble, time_steps, inputdim).
    
    Returns:
    numpy.ndarray: The observation error covariance matrix with shape (inputdim, inputdim).
    """
    # Get dimensions from input
    N_ensemble, time_steps, inputdim = observations_ensemble.shape
    mean_observation = np.mean(observations_ensemble, axis=0, keepdims=True)
    observation_errors = observations_ensemble - mean_observation
    observation_error_covariance = np.einsum('nti,ntj->ij', observation_errors, observation_errors) / (N_ensemble * time_steps)
    return observation_error_covariance