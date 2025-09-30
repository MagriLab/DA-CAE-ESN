import numpy as np
import matplotlib.pyplot as plt

def plot_decoded_comparison(test_snapshot, U_prediction_decoded, maxnorm, N_plot, N_lyap, 
                          vmin_main=-3, vmax_main=3, vmin_err=0, vmax_err=3, fs=14, 
                          cmap='RdBu_r', cmap_error='Reds'):
    """
    Plot spatial-temporal state for reference, prediction, and error.

    Parameters
    ----------
    test_snapshot : np.ndarray
        Ground truth data array (time, channels, space).
    U_prediction_decoded : np.ndarray
        Model-predicted data array (time, channels, space).
    maxnorm : float
        Normalization factor used in preprocessing.
    N_plot : int
        Number of timesteps to plot.
    N_lyap : int
        Number of timesteps per Lyapunov time (used for extent scaling).
    vmin_main, vmax_main : float
        Color limits for the  state.
    vmin_err, vmax_err : float
        Color limits for the error decoded.
    fs : int
        Font size.
    cmap, cmap_error : str
        Colormaps for state and error.
    """
    
    # Scale back to original units
    data1 = test_snapshot[:N_plot, 0, :] * maxnorm
    data2 = U_prediction_decoded[:N_plot, 0, :] * maxnorm
    data3 = data1 - data2

    # Lyapunov time array
    lyapunov_time = 0.08 * np.arange(0, 10000, 0.25)
    N_plot = 10 * N_lyap  # override if needed

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(5, 3), sharey=True)

    for i, data in enumerate([data1, data2, data3]):
        axes = axs[i]
        if i == 2:  # error decoded
            im = axes.imshow(np.abs(data), vmin=vmin_err, vmax=vmax_err, 
                             aspect='auto', cmap=cmap_error,
                             extent=[0, 2 * 10 * np.pi, int(lyapunov_time[N_plot]), lyapunov_time[0]])
        else:
            im = axes.imshow(data, vmin=vmin_main, vmax=vmax_main, 
                             aspect='auto', cmap=cmap,
                             extent=[0, 2 * 10 * np.pi, int(lyapunov_time[N_plot]), lyapunov_time[0]])

        # Axis labels
        axes.set_xlabel(r'$\mathit{x}$', fontsize=fs)
        axes.set_yticklabels([f'{int(tick)}' for tick in axes.get_yticks()], fontsize=fs)
        axes.yaxis.tick_left()

        axes.set_xticks(
            np.arange(0, 2 * 10 * np.pi + 0.01, step=(1 * 10 * np.pi)), 
            ['0', r'10$\pi$', r'20$\pi$'],
            fontsize=fs
        )

        # Add colorbar only for the last subplot
        if i == 2:
            cbar = fig.colorbar(im, ax=axes)
            cbar.ax.tick_params(labelsize=fs)

    axs[0].set_ylabel(r'$\tau_{\lambda}$', fontsize=fs)

    axs[0].set_title("Reference", fontsize=fs)
    axs[1].set_title("CAE-ESN", fontsize=fs)
    axs[2].set_title("Error", fontsize=fs)

    plt.tight_layout()
    return fig, axs

