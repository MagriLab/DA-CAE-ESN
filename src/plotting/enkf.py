import numpy as np
import matplotlib.pyplot as plt

def plot_esn_prediction(my_ESN, datasets, time_array_lyap, N_start, N_washout, N_plot, latentdim, errors):
    """
    Plot ESN closed-loop predictions vs. actual data for multiple datasets.
    
    Parameters
    ----------
    my_ESN : object
        ESN model with `closed_loop_with_washout` method.
    datasets : list of tuples
        List of (U_input, label) datasets to plot.
    time_array_lyap : np.ndarray
        Time array in Lyapunov units.
    N_start : int
        Start index for the washout sequence.
    N_washout : int
        Number of washout steps.
    N_plot : int
        Number of steps to plot after washout.
    latentdim : int
        Number of latent dimensions used.
    errors : module or object
        Must have an `rmse(pred, actual)` function.
    """
    
    def _plot_single(ax, U_input, label):
        # Run ESN in closed loop
        _, prediction = my_ESN.closed_loop_with_washout(
            U_input[N_start:N_start + N_washout, :latentdim], N_plot + N_washout
        )
        actual = U_input[N_washout + N_start:N_start + N_washout + N_plot, :latentdim]
        pred = prediction[:N_plot, :]

        # Plot
        ax.plot(time_array_lyap[:N_plot], actual, 'k', label='Actual')
        ax.plot(time_array_lyap[:N_plot], pred, '--', label='Prediction')
        ax.set_title(f'{label} Data')

        # RMSE
        rmse_val = errors.rmse(pred, actual)
        print(f'{label} RMSE: {rmse_val:.6f}')
        return rmse_val

    # Create figure
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 6), sharex=True)

    if len(datasets) == 1:
        axes = [axes]  # Ensure iterable

    for ax, (U_data, label) in zip(axes, datasets):
        _plot_single(ax, U_data, label)

    axes[-1].set_xlabel(r"Time [$\tau_{\lambda}$]")

    plt.tight_layout()
    return fig, axes



def plot_esn_predictions(my_ESN, cae_model, U_test, U_train_input, U_val_input, time_array_lyap, washout_len, dim, imagepath, idx):
    t_snapshot = [0, 10, 50, 100, 200, 300, 500]
    dt = kolmogorov_data["dt"] * kolmogorov_data["upsample"]
    N_max = t_snapshot[-1] + washout_len + 1
    N_start = 500
    input_data_esn = U_test[N_start : N_start + washout_len, :dim]
    reservoir, prediction = my_ESN.closed_loop_with_washout(input_data_esn, N_max)

    plot_true = cae_model.decode(
        torch.from_numpy(U_test[N_start + washout_len : N_start + washout_len + N_max, :]).float().to(device)
    ).cpu().numpy()
    plot_ae = cae_model.decode(
        torch.from_numpy(prediction).float().to(device)
    ).cpu().numpy()

    vmin = np.min(plot_true[:100])
    vmax = np.max(plot_true[:100])
    t_instances = len(t_snapshot)
    fig, axs = plt.subplots(t_instances, 3, figsize=(10, 2 * t_instances))
    cmap = "RdYlBu_r"
    for i, t in enumerate(t_snapshot):
        axs[i, 0].imshow(plot_true[t, 0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f"CAE-Reference (t={t*dt})")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(plot_ae[t, 0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i, 1].set_title(f"CAE-ESN Autonomous Prediction (t={t*dt})")
        axs[i, 1].axis("off")
        difference = plot_true[t, 0, :, :] - plot_ae[t, 0, :, :]
        axs[i, 2].imshow(difference, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i, 2].set_title(f"Difference (t={t*dt})")
        axs[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(imagepath / f"esn_closed_loop_t{idx}.png")
    plt.close()

    # Time series plots
    N_plot = 500
    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    N_start = 500
    for ax, (data, label) in zip(
        axes,
        [
            (U_train_input, "Training Data"),
            (U_val_input, "Validation Data"),
            (U_test, "Test Data"),
        ],
    ):
        reservoir, prediction = my_ESN.closed_loop_with_washout(
            data[N_start : N_start + washout_len, :dim], N_plot
        )
        ax.plot(
            time_array_lyap[:N_plot],
            data[washout_len + N_start : N_start + washout_len + N_plot, :dim],
            "k",
            label="Actual",
        )
        ax.plot(
            time_array_lyap[:N_plot], prediction[:N_plot, :], "--", label="Prediction"
        )
        ax.set_title(label)
    plt.tight_layout()
    plt.savefig(imagepath / f"esn_closed_loop_t_encoded{idx}.png")
    plt.close()