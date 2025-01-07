import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

from flybrain import model, utils
from flybrain.lyapunov import Lyapunov


def test_convergence_vs_tons(
    TOns, output_path, model=model.RNN, nLE=10, dt=0.1, tSim=200
):
    """
    Test the convergence of the maximum Lyapunov exponent (lambda_max) as a function of simulation time.

    Parameters:
        TOns (list): Time intervals between successive Lyapunov exponent computations.
        output_path (str): Path to save the output plot.
        model (RNN): Recurrent Neural Network model instance.
        nLE (int): Number of Lyapunov exponents to compute.
        dt (float): Time step size for simulation.
        tSim (int): Total simulation time.
    """
    print("****** 1_Convergence_Tons ******")

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    axs[0].set_title(r"$\lambda_{max}$ convergence")
    axs[0].set_ylabel(r"$\lambda_{max}$")
    axs[0].set_xlabel("$t$")

    axs[1].set_title(r"$\lambda_{max}$ convergence")
    axs[1].set_xlabel("$t$")
    axs[1].set_yscale("log")

    labels = [r"$t_{ons}=$" + f"{ons}" for ons in TOns]
    initial_states = model.H_0

    for tOns, label in tqdm(zip(TOns, labels), total=len(TOns)):
        model.set_states(initial_states)
        lyapunov = Lyapunov()
        lyapunov.compute_spectrum(
            model=model, dt=dt, tSim=tSim, nLE=nLE, tONS=tOns, logs=True
        )

        axs[0].plot(
            lyapunov.time_history[1:],
            np.array(lyapunov.spectrum_history)[:, 1],
            label=label,
            marker="o",
            markersize=0.5,
            lw=1,
            alpha=0.5,
        )
        axs[1].plot(
            lyapunov.time_history[1:],
            np.array(lyapunov.spectrum_history)[:, 1],
            label=label,
            marker="o",
            markersize=0.5,
            lw=1,
            alpha=0.5,
        )

    axs[0].legend()
    plt.tight_layout()
    plt.savefig(output_path)


def test_efficiency_time_vs_exponent(
    output_path, N=200, g=1.5, dt=0.1, tONs=0.2, tSim=200
):
    """
    Benchmark computation time as a function of the number of Lyapunov exponents.

    Parameters:
        output_path (str): Path to save the output plot.
        N (int): Number of neurons in the network.
        g (float): Coupling strength.
        dt (float): Time step size for simulation.
        tONs (float): Time interval for Lyapunov exponent computation.
        tSim (int): Total simulation time.
    """
    print("****** 0_time_vs_#exponent_spectrum ******")

    n_exponent = np.arange(1, N, int(N * 0.1))
    timings = np.zeros(len(n_exponent))

    W, C = utils.construct_Random_Matrix_simple(n_neurons=N, coupling=g)
    initial_conditions = np.zeros(N)
    rnn = model.RNN(
        connectivity_matrix=C, weights_matrix=W, initial_condition=initial_conditions
    )
    rnn.eval()
    rnn.reset_states()

    for i, n in tqdm(enumerate(n_exponent), total=len(n_exponent)):
        t0 = time.time()
        lyapunov = Lyapunov()
        lyapunov.compute_spectrum(model=rnn, tSim=tSim, dt=dt, tONS=tONs, nLE=n)
        timings[i] = time.time() - t0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(n_exponent / N, timings, marker="D", markersize=5, lw=1.5)
    ax.set_ylabel("Execution time [s]")
    ax.set_xlabel("% Exponents computed")
    plt.suptitle("Time Vs % Exponent computed")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    plt.tight_layout()
    plt.savefig(output_path)


def test_efficiency_time_vs_simultime(
    output_path, it_max=200, N=400, g=1.5, dt=0.1, tONs=0.2
):
    """
    Benchmark computation time as a function of simulation time.

    Parameters:
        output_path (str): Path to save the output plot.
        it_max (int): Maximum number of iterations.
        N (int): Number of neurons in the network.
        g (float): Coupling strength.
        dt (float): Time step size for simulation.
        tONs (float): Time interval for Lyapunov exponent computation.
    """
    print("****** 1_time_vs_simulTime_spectrum ******")

    W, C = utils.construct_Random_Matrix_simple(n_neurons=N, coupling=g)
    initial_conditions = np.zeros(N)
    rnn = model.RNN(
        connectivity_matrix=C, weights_matrix=W, initial_condition=initial_conditions
    )
    rnn.eval()
    rnn.reset_states()

    iterations = np.arange(10, it_max, 20)
    timings = np.zeros(len(iterations))

    for i, iter_count in tqdm(enumerate(iterations), total=len(iterations)):
        t0 = time.time()
        Lyapunov().compute_spectrum(
            model=rnn, tSim=iter_count, dt=dt, tONS=tONs, nLE=10
        )
        timings[i] = time.time() - t0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(iterations, timings, marker="D", markersize=5, lw=1.5)
    ax.set_ylabel("Execution time [s]")
    ax.set_xlabel("Simulation Duration [tau]")
    plt.suptitle("Time Vs Simulation Time")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    plt.tight_layout()
    plt.savefig(output_path)


def test_efficiency_vs_netSize(
    output_path, N_max, g=1.5, dt=0.1, tONs=0.2, tSim=200, nLE_ratio=0.1
):
    """
    Benchmark computation time as a function of network size.

    Parameters:
        output_path (str): Path to save the output plot.
        N_max (int): Maximum number of neurons in the network.
        g (float): Coupling strength.
        dt (float): Time step size for simulation.
        tONs (float): Time interval for Lyapunov exponent computation.
        tSim (int): Total simulation time.
        nLE_ratio (float): Ratio of Lyapunov exponents to network size.
    """
    print("****** 2_time_vs_netSize_spectrum ******")

    sizes = np.arange(100, N_max, 100)
    timings = np.zeros(len(sizes))

    for i, size in tqdm(enumerate(sizes), total=len(sizes)):
        W, C = utils.construct_Random_Matrix_simple(n_neurons=size, coupling=g)
        initial_conditions = np.zeros(size)
        rnn = model.RNN(
            connectivity_matrix=C,
            weights_matrix=W,
            initial_condition=initial_conditions,
        )
        rnn.eval()

        t0 = time.time()
        Lyapunov().compute_spectrum(
            model=rnn, tSim=tSim, dt=dt, tONS=tONs, nLE=int(nLE_ratio * size)
        )
        timings[i] = time.time() - t0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(sizes, timings, marker="D", markersize=5, lw=1.5)
    ax.set_ylabel("Execution time [s]")
    ax.set_xlabel("Network size")
    plt.suptitle("Time Vs Network size")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    plt.tight_layout()
    plt.savefig(output_path)


# Main Script
def main():
    ROOT_PATH = utils.get_root()
    output_folder = os.path.join(ROOT_PATH, "data", "fig", "lyapunov_benchmark")
    os.makedirs(output_folder, exist_ok=True)

    N = 200

    # Test A: Convergence vs T_ONS
    test_convergence_vs_tons(
        TOns=[0.2, 1, 5, 10, None],
        model=model.RNN(
            connectivity_matrix=utils.construct_Random_Matrix_simple(
                n_neurons=N, coupling=5
            )[1],
            weights_matrix=utils.construct_Random_Matrix_simple(
                n_neurons=N, coupling=5
            )[0],
            initial_condition=np.random.normal(0, 1, N),
        ),
        output_path=os.path.join(output_folder, f"0_convergence_vs_Tons_N{N}.svg"),
    )

    # Test B: Efficiency Time vs Number of Exponents
    test_efficiency_time_vs_exponent(
        output_path=os.path.join(output_folder, f"1_time_vs_nle_N{N}.svg"), N=N
    )

    # Test C: Efficiency Time vs Simulation Time
    test_efficiency_time_vs_simultime(
        output_path=os.path.join(output_folder, f"2_time_vs_tSim_N{N}.svg"), N=N
    )

    # Test D: Efficiency Time vs Network Size
    test_efficiency_vs_netSize(
        output_path=os.path.join(output_folder, f"3_time_vs_netSize_N{N}.svg"),
        N_max=1000,
    )


if __name__ == "__main__":
    main()
