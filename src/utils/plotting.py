import os
import csv
import h5py
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# from utils.data_process import *
from dataclasses import dataclass
from typing import Union, Optional
import matplotlib.colors as colors
from collections import defaultdict
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
from matplotlib.ticker import AutoMinorLocator

import glob
import h5py
from scipy import stats

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


@dataclass
class PlottingArgs:
    truth: np.array
    title: str
    xlabel: str
    ylabel: str
    sup_title: str
    log: Optional[bool]


class ReconstructionPlotting:

    def __init__(self) -> None:
        pass

    def hist_2d(self):
        pass

    def hist_1d(self):
        pass

    def hist_2d_error_bar(self):
        pass

    def hist_1d_error_bar(self):
        pass

    def _find_contours_2D(x_values, y_values, xbins, weights=None, c1=16, c2=84):
        """
        Find upper and lower contours and median
        x_values = array, input for hist2d for x axis (typically truth)
        y_values = array, input for hist2d for y axis (typically reconstruction)
        xbins = values for the starting edge of the x bins (output from hist2d)
        c1 = percentage for lower contour bound (16% - 84% means a 68% band, so c1 = 16)
        c2 = percentage for upper contour bound (16% - 84% means a 68% band, so c2=84)
        Returns:
            x = values for xbins, repeated for plotting (i.e. [0,0,1,1,2,2,...]
            y_median = values for y value medians per bin, repeated for plotting (i.e. [40,40,20,20,50,50,...]
            y_lower = values for y value lower limits per bin, repeated for plotting (i.e. [30,30,10,10,20,20,...]
            y_upper = values for y value upper limits per bin, repeated for plotting (i.e. [50,50,40,40,60,60,...]
        """
        if weights is not None:
            import wquantiles as wq
        y_values = np.array(y_values)
        indices = np.digitize(x_values, xbins)
        r1_save = []
        r2_save = []
        median_save = []
        for i in range(1, len(xbins)):
            mask = indices == i
            if len(y_values[mask]) > 0:
                if weights is None:
                    r1, m, r2 = np.percentile(y_values[mask], [c1, 50, c2])
                else:
                    r1 = wq.quantile(y_values[mask], weights[mask], c1 / 100.0)
                    r2 = wq.quantile(y_values[mask], weights[mask], c2 / 100.0)
                    m = wq.median(y_values[mask], weights[mask])
            else:
                r1 = 0
                m = 0
                r2 = 0
            median_save.append(m)
            r1_save.append(r1)
            r2_save.append(r2)
        median = np.array(median_save)
        lower = np.array(r1_save)
        upper = np.array(r2_save)

        x = list(itertools.chain(*zip(xbins[:-1], xbins[1:])))
        y_median = list(itertools.chain(*zip(median, median)))
        y_lower = list(itertools.chain(*zip(lower, lower)))
        y_upper = list(itertools.chain(*zip(upper, upper)))

        return x, y_median, y_lower, y_upper


def dom_plot():

    # Reading original dom positions from dat file
    dat = "/home/bread/Documents/projects/neutrino/data/model/DOM_Position_GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.dat"
    df = pd.read_csv(
        dat,
        delimiter=" ",
        header=None,
        names=["String", "Count", "X", "Y", "Z"],
        skiprows=[0],
    )

    dc = df[df["String"] > 78]
    df = df[df["String"] <= 78]

    pulse, truth = get_db(
        "/home/bread/Documents/projects/neutrino/data/database/oscNext_genie_level5_v02.00_pass2.141122.000000.db"
    )

    print(pulse)

    data = PulseDataProcessing((pulse, truth))

    data.sublist()

    pulse = data.pulse

    maximum = 0
    index = 0

    for i in range(len(pulse)):
        if len(pulse[i]) > maximum:
            index = i
            maximum = len(pulse[i])

    pulse = np.array(pulse[index])

    print(pulse)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(df["X"], df["Y"], df["Z"], s=0.55, alpha=0.55)
    ax.scatter(dc["X"], dc["Y"], dc["Z"], s=0.55, alpha=0.55, color="orange")
    ax.scatter(pulse[:, 0], pulse[:, 1], pulse[:, 2], s=pulse[:, 3] * 15, color="red")
    plt.show()


def true_energy_distribution():

    # df = pd.read_csv('/home/bread/Documents/projects/neutrino/data/model/events.csv')
    df = build_files(
        "/home/bread/Documents/projects/neutrino/data/archive/data_colletion",
        "output_label_names",
        "labels",
    )

    inelasticity = np.array(df["Cascade"] / df["Energy"])

    energy = np.array(df["Energy"])

    energy_dist = []
    inelasticity_dist = []

    for i in range(len(energy)):
        if inelasticity[i] > 0.4 and inelasticity[i] <= 0.6:
            energy_dist.append(energy[i])
            inelasticity_dist.append(inelasticity[i])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(energy, bins=100, label=f"Energy", alpha=0.6, color="olive")
    ax.hist(
        inelasticity_dist,
        bins=np.arange(0, 3, 0.03),
        label=f"Inelasticity (0.4 - 0.6]",
        alpha=0.6,
        color="deepskyblue",
    )
    ax.set_title(
        f"Energy Distribution When Associacted\n Inelasticity is in Range (0.4, 0.6] - 1d Histogram"
    )
    ax.set_xlabel(f"Distribution")
    ax.set_ylabel("Count")
    ax.grid("on", alpha=0.2, linestyle="-")
    plt.legend()
    plt.savefig(
        f"/home/bread/Documents/projects/neutrino/data/fig/save/energy_inelasticity_mid_1d.png"
    )
    plt.show()
    plt.clf()

    plt.hist2d(energy_dist, inelasticity_dist, bins=30)
    plt.xlabel(f"Energy")
    plt.ylabel(f"Inelasticity")
    plt.title(
        f"Energy Distribution When Associacted\n Inelasticity is in Range (0.4, 0.6] - 2d Histogram"
    )
    plt.savefig(
        f"/home/bread/Documents/projects/neutrino/data/fig/save/energy_inelasticity_mid_2d.png"
    )
    plt.show()
    plt.clf()


def correlation_mapping():

    import seaborn as sns

    files = glob.glob(
        "/home/bread/Documents/MuonNeutrinoReconstruction/src/data/archive/flercnn_IC19cut/*"
    )

    inelasticity = []
    anti = []
    charge = []
    time = []
    track = []
    zenith = []
    azimuth = []

    for file in files:
        hdf = h5py.File(file, "r")
        inelasticity.extend(np.array(hdf["Y_test"][:, 14]))
        anti.extend(np.array(hdf["Y_test"][:, 11]))
        charge.extend(np.array(hdf["Y_test"][:, 12]))
        time.extend(np.array(hdf["Y_test"][:, 3]))
        zenith.extend(np.array(hdf["Y_test"][:, 1]))
        azimuth.extend(np.array(hdf["Y_test"][:, 7]))

    data = pd.DataFrame(
        {
            "Inelasticity": inelasticity,
            "Charge": charge,
            "Time": time,
            "IsAntineutrino": anti,
            "Zenith": zenith,
            "Azimuth": azimuth,
        }
    )

    fig = plt.figure(figsize=(10, 8))

    ax = plt.axes()
    ax.set_title(
        "Reconstruction Variable Correlation Map",
        # fontweight=700,
        fontsize=14,
    )
    # ax.set_facecolor()
    # fig.patch.set_facecolor()
    cor = data.corr()
    sns.heatmap(cor, annot=True, cmap="Blues")
    plt.savefig("src/data/plot/correlation_map.png")
    plt.show()
    plt.clf()


def plot_loss(history, path, title, log, save, show):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(history["mae"])
    ax.plot(history["val_mae"])
    if log:
        ax.set_yscale("log")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.suptitle("Convolutional Neural Network", fontweight=700)
    ax.set_title(f"Mean Absolute Error - {title}")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_xlabel("Epoch")
    ax.grid("on", alpha=0.2, linestyle="-")
    ax.legend(["Training", "Validating"], loc="upper left")

    path = os.path.join(path, "cnn_mae_history.jpg")

    if save:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()

    return path


def reconstructed_hist_2d(truth, reconstructed, path, title, train_var, save, show):

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.hist2d(truth, reconstructed, bins=30)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Convolutional Neural Network", fontweight=700)
    ax.set_title(f"Low Energy Muon Neutrino Inelasticity Reconstruction - 2d Histogram")
    ax.set_xlabel(f"True Inelasticity")
    ax.set_ylabel(f"Reconstructed Inelasticity")

    h, xedges, yedges, im = ax.hist2d(truth, reconstructed, bins=50, cmap="YlOrRd")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Event Count", rotation=270, labelpad=15)

    ax.plot([0, 1], [0, 1], "k--", label="Ideal")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.grid("on", alpha=0.2, linestyle="-")
    path = os.path.join(path, "cnn_2d_hist.jpg")
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()

    return path


def reconstructed_hist_1d(truth, reconstructed, path, title, train_var, save, show):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(truth, bins=30, label=f"True {title}", alpha=0.6, color="olive")
    ax.hist(
        reconstructed,
        bins=30,
        label=f"Reconstructed {train_var.title()}",
        alpha=0.6,
        color="deepskyblue",
    )
    fig.suptitle("Convolutional Neural Network", fontweight=700)
    ax.set_title(f"{title} - 1d Histogram")
    ax.set_xlabel("Range")
    ax.set_ylabel("Count")
    ax.grid("on", alpha=0.2, linestyle="-")
    ax.legend()
    path = os.path.join(path, "cnn_1d_hist.jpg")
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()

    return path


def energy_reconstruction_cut_1d(
    energy, reconstruction, range_cut, path, title, save, show
):

    reconstruction = np.array(reconstruction)

    assert range_cut in (
        "high",
        "middle",
        "low",
    ), "range value must be `high`, `middle`, or `low`"

    if range_cut == "high":
        energy_cut = energy[
            np.all([reconstruction > 0.6, reconstruction <= 1.0], axis=0)
        ]
        reco_cut = reconstruction[
            np.all([reconstruction > 0.6, reconstruction <= 1.0], axis=0)
        ]
        txt = "(0.6, 1.0]"
    if range_cut == "middle":
        energy_cut = energy[
            np.all([reconstruction > 0.4, reconstruction <= 0.6], axis=0)
        ]
        reco_cut = reconstruction[
            np.all([reconstruction > 0.4, reconstruction <= 0.6], axis=0)
        ]
        txt = "(0.4, 0.6]"
    if range_cut == "low":
        energy_cut = energy[
            np.all([reconstruction >= 0, reconstruction <= 0.4], axis=0)
        ]
        reco_cut = reconstruction[
            np.all([reconstruction >= 0, reconstruction <= 0.4], axis=0)
        ]
        txt = "[0.0, 0.4]"

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(energy_cut, bins=50, label=f"True Energy", alpha=0.6, color="olive")
    ax.hist(
        reco_cut,
        bins=np.arange(
            min(energy_cut), max(energy_cut), (max(energy_cut) - min(energy_cut)) / 50
        ),
        label=f"Reconstructed {title} {txt}",
        alpha=0.6,
        color="deepskyblue",
    )
    ax.set_yscale("log")
    fig.suptitle("Convolutional Neural Network", fontweight=600)
    ax.set_title(
        f"Reconstructed {title} Cut in Range {txt}\n And Associated Energy Values - 1d Histogram"
    )
    ax.grid("on", alpha=0.2, linestyle="-")
    plt.legend()
    if save:
        plt.savefig(f"{path}/nn_reco_true_energy_cut_{range_cut}_1d.png")
    if show:
        plt.show()
    plt.close()


def parse_training_data(file_content):
    lines = file_content.strip().split("\n")
    header = lines[0].split("\t")

    all_runs = []
    current_run = defaultdict(list)

    for line in lines[1:]:
        values = line.split("\t")
        epoch = int(values[0])

        if epoch == 1 and current_run:
            all_runs.append(dict(current_run))
            current_run = defaultdict(list)

        for key, value in zip(header, values):
            try:
                current_run[key].append(float(value))
            except ValueError:
                current_run[key].append(value)

    if current_run:
        all_runs.append(dict(current_run))

    return all_runs


def plot_data_transform():

    path = "/home/bread/MuonNeutrinoReconstruction/src/data/archive/contained/0_0_00_contained.hdf5"

    with h5py.File(path, "r") as hdf:

        X_test_DC = np.array(hdf["X_test_DC"])
        X_test_IC = np.array(hdf["X_test_IC"])

    dat = (
        "src/data/model/DOM_Position_GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.dat"
    )

    df = pd.read_csv(
        dat,
        delimiter=" ",
        header=None,
        names=["String", "Count", "X", "Y", "Z"],
        skiprows=[0],
    )

    dc = df[df["String"] > 78]

    plot_df = pd.DataFrame({"X": [1], "Y": [2], "Z": [2], "Charge": [3]})

    x_pos = []
    y_pos = []
    z_pos = []
    charge = []

    for row in range(len(X_test_DC[0])):

        indices = np.where(X_test_DC[0][row][:, 0] > 0.0)

        string = dc[dc["String"] == 79 + row]

        if len(indices[0]) > 0:

            for index in indices[0]:
                charge.append(X_test_DC[0][row][index][0])

            print(indices[0])

            x_positions = [
                string.loc[string["Count"] == indices[0][i]]["X"].values[0]
                for i in range(len(indices[0]))
            ]
            for x in x_positions:
                x_pos.append(x)

            y_positions = [
                string.loc[string["Count"] == indices[0][i]]["Y"].values[0]
                for i in range(len(indices[0]))
            ]
            for y in y_positions:
                y_pos.append(y)

            z_positions = [
                string.loc[string["Count"] == indices[0][i]]["Z"].values[0]
                for i in range(len(indices[0]))
            ]
            for z in z_positions:
                z_pos.append(z)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x_pos, y_pos, z_pos, c=range(len(x_pos)), cmap="viridis")
    plt.show()


def find_contours_2D(x_values, y_values, xbins, weights=None, c1=16, c2=84):
    """
    Find upper and lower contours and median
    x_values = array, input for hist2d for x axis (typically truth)
    y_values = array, input for hist2d for y axis (typically reconstruction)
    xbins = values for the starting edge of the x bins (output from hist2d)
    c1 = percentage for lower contour bound (16% - 84% means a 68% band, so c1 = 16)
    c2 = percentage for upper contour bound (16% - 84% means a 68% band, so c2=84)
    Returns:
        x = values for xbins, repeated for plotting (i.e. [0,0,1,1,2,2,...]
        y_median = values for y value medians per bin, repeated for plotting (i.e. [40,40,20,20,50,50,...]
        y_lower = values for y value lower limits per bin, repeated for plotting (i.e. [30,30,10,10,20,20,...]
        y_upper = values for y value upper limits per bin, repeated for plotting (i.e. [50,50,40,40,60,60,...]
    """
    if weights is not None:
        import wquantiles as wq
    y_values = np.array(y_values)
    indices = np.digitize(x_values, xbins)
    r1_save = []
    r2_save = []
    median_save = []
    for i in range(1, len(xbins)):
        mask = indices == i
        if len(y_values[mask]) > 0:
            if weights is None:
                r1, m, r2 = np.percentile(y_values[mask], [c1, 50, c2])
            else:
                r1 = wq.quantile(y_values[mask], weights[mask], c1 / 100.0)
                r2 = wq.quantile(y_values[mask], weights[mask], c2 / 100.0)
                m = wq.median(y_values[mask], weights[mask])
        else:
            r1 = 0
            m = 0
            r2 = 0
        median_save.append(m)
        r1_save.append(r1)
        r2_save.append(r2)
    median = np.array(median_save)
    lower = np.array(r1_save)
    upper = np.array(r2_save)

    x = list(itertools.chain(*zip(xbins[:-1], xbins[1:])))
    y_median = list(itertools.chain(*zip(median, median)))
    y_lower = list(itertools.chain(*zip(lower, lower)))
    y_upper = list(itertools.chain(*zip(upper, upper)))

    return x, y_median, y_lower, y_upper


def plot_2D_prediction(
    truth,
    nn_reco,
    save=False,
    savefolder=None,
    weights=None,
    syst_set="",
    bins=60,
    minval=None,
    maxval=None,
    switch_axis=False,
    cut_truth=False,
    axis_square=False,
    zmax=None,
    log=False,
    variable="Zenith",
    units="",
    epochs=None,
    reco_name="CNN",
):
    """
    Plot testing set reconstruction vs truth
    Recieves:
        truth = array, Y_test truth
        nn_reco = array, neural network prediction output
        save = optional, bool to save plot
        savefolder = optional, output folder to save to, if not in current dir
        syst_set = string, name of the systematic set (for title and saving)
        bins = int, number of bins plot (will use for both the x and y direction)
        minval = float, minimum value to cut nn_reco results
        maxval = float, maximum value to cut nn_reco results
        cut_truth = bool, true if you want to make the value cut on truth rather than nn results
        axis_square = bool, cut axis to be square based on minval and maxval inputs
        variable = string, name of the variable you are plotting
        units = string, units for the variable you are plotting
    Returns:
        2D plot of True vs Reco
    """

    path = os.path.join(savefolder, "cnn_2d_hist_errorbars.jpg")

    # hack!
    if cut_truth:

        if not minval:
            minval = min(truth)
        if not maxval:
            maxval = max(truth)
        mask1 = np.logical_and(truth >= minval, truth <= maxval)
        name = "True %s [%.2f, %.2f]" % (variable, minval, maxval)

    else:
        if not minval:
            minval = min([min(nn_reco), min(truth)])
        if not maxval:
            maxval = max([max(nn_reco), max(truth)])
        mask1 = np.ones(len(truth), dtype=bool)
        # mask = np.logical_and(nn_reco >= minval, nn_reco <= maxval)
        name = "%s %s [%.2f, %.2f]" % (reco_name, variable, minval, maxval)

    cutting = False
    if axis_square:
        mask2 = np.logical_and(nn_reco >= minval, nn_reco <= maxval)
        overflow = abs(sum(mask1) - sum(mask2))
        print("Axis overflow: ", overflow)
        mask = np.logical_and(mask1, mask2)
    else:
        mask = mask1

    maxplotline = min([max(nn_reco), max(truth)])
    minplotline = max([min(nn_reco), min(truth)])

    truth = truth  # [mask]
    nn_reco = nn_reco  # [mask]

    # Cut axis
    if axis_square:
        xmin = minval
        ymin = minval
        xmax = maxval
        ymax = maxval
    else:
        xmin = min(truth)
        ymin = min(nn_reco)
        xmax = max(truth)
        ymax = max(nn_reco)
    if switch_axis:
        xmin, ymin = ymin, xmin
        xmax, ymax = ymax, xmax

    if weights is None:
        cmin = 1
    else:
        cmin = 1e-12

    fig, ax = plt.subplots(figsize=(10, 7))

    fig.suptitle("Convolutional Neural Network", fontweight=700, fontsize=14)
    ax.set_title(f"Low Energy Muon Neutrino Inelasticity Reconstruction", fontsize=13)

    cts, xbin, ybin, img = ax.hist2d(
        truth,
        nn_reco,
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]],
        cmap="Blues",
        weights=weights,
        cmax=zmax,
        cmin=cmin,
    )

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Event Count", rotation=270, labelpad=15)

    ax.set_xlabel("True Inelasticity", fontsize=14)
    ax.set_ylabel("Reconstructed Inelasticity", fontsize=14)

    # Plot 1:1 line
    if axis_square:
        ax.plot([minval, maxval], [minval, maxval], "w:", linewidth=2)
    else:
        ax.plot(
            [minplotline, maxplotline], [minplotline, maxplotline], "w:", linewidth=2
        )

    if switch_axis:
        x, y, y_l, y_u = find_contours_2D(nn_reco, truth, xbin, weights=weights)
    else:
        x, y, y_l, y_u = find_contours_2D(truth, nn_reco, xbin, weights=weights)

    ax.plot(x, y, color="r", label="Median", linewidth=2)
    ax.plot(x, y_l, color="r", label="68% band", linestyle="dashed", linewidth=2)
    ax.plot(x, y_u, color="r", linestyle="dashed", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax.legend(fontsize=14)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    textstr = "IceCube Work in Progress"
    ax = plt.gca()
    ax.text(0.35, 0.02, textstr, transform=ax.transAxes, color="black")

    reco_name = reco_name.replace(" ", "")
    variable = variable.replace(" ", "")
    nocut_name = ""
    if weights is not None:
        nocut_name = "Weighted"
    if not axis_square:
        nocut_name = "_nolim"
    if zmax:
        nocut_name += "_zmax%i" % zmax
    if switch_axis:
        nocut_name += "_SwitchedAxis"
    if save:
        plt.savefig(path)
    plt.show()
    plt.close()

    return path


def neutrino_antineutrino_ratios():

    files = glob.glob(
        "/home/bread/Documents/projects/MuonNeutrinoReconstruction/data/archive/*"
    )

    inelasticity = []
    anti = []
    weights = []

    for file in files:
        hdf = h5py.File(file, "r")
        inelasticity.extend(np.array(1 - hdf["labels"][:, 1] / hdf["labels"][:, 0]))
        anti.extend(np.array(hdf["labels"][:, 11]))
        weights.extend(np.array(hdf["weights"][:, 10]))

    df = pd.DataFrame(
        {"Inelasticity": inelasticity, "IsAntineutrino": anti, "Weights": weights}
    )

    df = df.drop(df[df["Inelasticity"] < 0].index)

    anti = df[df["IsAntineutrino"] == 1.0]
    neutrino = df[df["IsAntineutrino"] == 0.0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle("IceCube", fontweight=700, fontsize=14)

    # Define the range and bins
    min_val = min(anti["Inelasticity"].min(), neutrino["Inelasticity"].min())
    max_val = max(anti["Inelasticity"].max(), neutrino["Inelasticity"].max())
    bins = np.linspace(min_val, max_val, 51)

    # Calculate histograms
    anti_hist, _ = np.histogram(
        anti["Inelasticity"], bins=bins, weights=anti["Weights"]
    )
    neutrino_hist, _ = np.histogram(
        neutrino["Inelasticity"], bins=bins, weights=neutrino["Weights"]
    )

    # Plot histograms
    cmap = plt.get_cmap("YlOrRd")
    cmap = plt.get_cmap("Blues")
    color1 = cmap(0.2)
    color2 = cmap(0.8)
    color3 = cmap(0.5)

    ax1.hist(
        neutrino["Inelasticity"],
        bins=bins,
        # histtype="step",
        label="Neutrino",
        color=color2,
        edgecolor="black",
        alpha=1,
        # linewidth=2,
    )
    ax1.hist(
        anti["Inelasticity"],
        bins=bins,
        # histtype="step",
        label="Antineutrino",
        color=color1,
        edgecolor="black",
        alpha=1,
        # linewidth=2,
    )
    ax1.set_ylabel("Occurrence")
    ax1.legend(frameon=False, handlelength=1.5, handleheight=0)
    ax1.grid("on", alpha=0.2, linestyle="-")
    ax1.set_title("Antineutrino vs Neutrino Inelasticity Distribution", fontsize=13)

    # Calculate ratio
    ratio = np.divide(
        anti_hist, neutrino_hist, where=neutrino_hist != 0
    )  # Avoid division by zero
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot ratio
    ax2.plot(bin_centers, ratio, color=color3)
    ax2.axhline(
        y=1, color="gray", linestyle="--"
    )  # Add a horizontal line at y=1 for reference
    ax2.set_ylabel(r"Ratio ($\frac{\bar{\nu}}{\nu}$)")
    ax2.set_xlabel("Inelasticity Range")
    ax2.set_title("Ratio of Antineutrino to Neutrino", fontsize=13)
    ax2.grid("on", alpha=0.2, linestyle="-")

    # Set y-axis limits for ratio plot
    ax2.set_ylim(0, 1.35)  # Adjust these values as needed

    plt.tight_layout()
    # plt.show()
    plt.savefig(
        "/home/bread/Documents/MuonNeutrinoReconstruction/src/data/plot/neutrino_anti.png"
    )
    plt.show()


def error_against_energy(truth, reco, energy, savefolder):

    path = os.path.join(savefolder, "cnn_energy_errorbars.jpg")

    bins = 50
    color = "Blues"

    from scipy.stats import binned_statistic

    error = reco - truth

    # Create bins for energy
    energy_bins = np.linspace(np.min(energy), np.max(energy), bins + 1)

    # Calculate median of error for each energy bin
    median_error, bin_edges, _ = binned_statistic(
        energy, error, statistic="median", bins=energy_bins
    )

    # Calculate 2.5th and 97.5th percentiles for 95% confidence interval
    lower_95, _, _ = binned_statistic(
        energy, error, statistic=lambda x: np.percentile(x, 2.5), bins=energy_bins
    )
    upper_95, _, _ = binned_statistic(
        energy, error, statistic=lambda x: np.percentile(x, 97.5), bins=energy_bins
    )

    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("IceCube", fontweight=700, fontsize=14)

    # Get colors from colormap
    cmap = plt.get_cmap(color)
    fill_color = cmap(0.5)
    line_color = cmap(0.8)

    # Find the bin with the narrowest 95% CI
    ci_width = upper_95 - lower_95
    narrowest_ci_index = np.argmin(ci_width)
    narrowest_ci_energy = bin_centers[narrowest_ci_index]
    narrowest_ci_median = median_error[narrowest_ci_index]
    narrowest_ci_width = ci_width[narrowest_ci_index]

    # Add a point and annotation for the narrowest CI
    ax.plot(narrowest_ci_energy, narrowest_ci_median, "ro", markersize=5)
    ax.annotate(
        f"Narrowest 95% CI\nEnergy: {narrowest_ci_energy:.2f}$*10^2$ GeV\nCI Width: {narrowest_ci_width:.2f}",
        xy=(narrowest_ci_energy, narrowest_ci_median),
        xytext=(10, 30),
        textcoords="offset points",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc=cmap(0.7), alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    # Plot the median line
    ax.plot(
        bin_centers, median_error, "-", color=line_color, linewidth=2, label="Median"
    )

    # Plot the 95% confidence interval
    ax.fill_between(
        bin_centers, lower_95, upper_95, color=fill_color, alpha=0.3, label="95% CI"
    )

    # Set labels and title
    ax.set_xlabel("Energy ($10^2$ GeV)")
    ax.set_ylabel("Error (Reconstructed - Truth)")
    fig.suptitle("Convolutional Neural Network", fontweight=700, fontsize=14)
    ax.set_title(
        "Low Energy Muon Inelasticity Reconstruction Error as a Function of Energy",
        fontsize=13,
    )

    # Add a horizontal line at y=0
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.grid("on", alpha=0.2, linestyle="-")

    # Add legend
    ax.legend()

    # Adjust layout and display plot
    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def plot_inelasticity_distribution(
    inelasticity, weights=None, bins=50, save=False, savefolder=None
):
    """
    Plot the distribution of inelasticity using the YlOrRd colormap.

    Parameters:
    inelasticity (array-like): Array of inelasticity values
    weights (array-like, optional): Array of weights for each inelasticity value
    bins (int): Number of bins for the histogram
    save (bool): Whether to save the plot
    savefolder (str): Folder to save the plot in (if save is True)
    """

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("IceCube", fontweight=700, fontsize=14)

    # Create a custom colormap based on YlOrRd
    color = "Blues"
    cmap = plt.get_cmap(color)

    # Plot histogram
    n, bins, patches = ax.hist(
        inelasticity,
        bins=bins,
        weights=weights,
        range=(0, 1),
        edgecolor="black",
        alpha=0.5,
        color=cmap(0.65),
    )  # Use a color from our custom colormap

    # Calculate and plot median
    median = np.median(inelasticity)
    ax.axvline(
        median,
        color="black",
        linestyle="dashed",
        linewidth=2,
        label=f"Median: {median:.2f}",
    )

    # Set labels and title
    ax.set_xlabel("True Inelasticity")
    ax.set_ylabel("Occurence")
    ax.set_title("Low Energy Muon Neutrino Inelasticity Distribution", fontsize=13)

    # Add grid
    ax.grid("on", alpha=0.2, linestyle="-")

    # Add legend
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Save plot if requested
    if save:
        if savefolder is None:
            savefolder = "."
        plt.savefig(
            f"{savefolder}/inelasticity_distribution_YlOrRd.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
    plt.close()


def plot_2D_prediction_energy_ranges(
    truth,
    nn_reco,
    energy,
    save=True,
    savefolder=None,
    weights=None,
    bins=60,
    reco_name="CNN",
):
    """
    Plot testing set reconstruction vs truth for three energy ranges and all energies in a 2x2 grid

    Parameters:
        truth: array, Y_test truth for inelasticity
        nn_reco: array, neural network prediction output for inelasticity
        energy: array, true neutrino energies
        save: optional, bool to save plot
        savefolder: optional, output folder to save to, if not in current dir
        weights: optional, array of weights for each event
        bins: int, number of bins plot (will use for both the x and y direction)
        reco_name: string, name of the reconstruction method
    """

    energy_ranges = [
        (0, 30 / 100),
        (30 / 100, 100 / 100),
        (100 / 100, np.inf),
        (0, np.inf),
    ]
    energy_labels = [
        "$E_{true}$ < 30 GeV",
        "$E_{true}$ 30-100 GeV",
        "$E_{true}$ > 100 GeV",
        "All Energies",
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Convolutional Neural Network\nLow Energy Muon Neutrino Inelasticity Reconstruction With Energy Cuts",
        fontweight=700,
        fontsize=14,
    )

    for i, ((emin, emax), elabel) in enumerate(zip(energy_ranges, energy_labels)):
        ax = axs[i // 2, i % 2]

        if emax == np.inf:
            mask = energy >= emin
        else:
            mask = (energy >= emin) & (energy < emax)

        truth_masked = truth[mask]
        nn_reco_masked = nn_reco[mask]
        weights_masked = weights[mask] if weights is not None else None

        cts, xbin, ybin, img = ax.hist2d(
            truth_masked,
            nn_reco_masked,
            bins=bins,
            range=[[0, 1], [0, 1]],
            cmap="Blues",
            weights=weights_masked,
            cmin=1 if weights is None else 1e-12,
        )

        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label("Event Count", rotation=270, labelpad=15)

        ax.set_xlabel("True Inelasticity")
        ax.set_ylabel("Reconstructed Inelasticity")
        ax.set_title(elabel, fontsize=13)

        ax.plot([0, 1], [0, 1], "k--", label="Ideal")

        x, y, y_l, y_u = find_contours_2D(
            truth_masked, nn_reco_masked, xbin, weights=weights_masked
        )

        ax.plot(x, y, color="black", label="Median", linewidth=2)
        ax.plot(
            x, y_l, color="black", label="68% band", linestyle="dashed", linewidth=2
        )
        ax.plot(x, y_u, color="black", linestyle="dashed", linewidth=2)
        ax.legend()

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        ax.text(
            0.05,
            0.95,
            "IceCube Work in Progress",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            color="black",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    if save:
        filename = f"{reco_name.lower()}_2d_hist_energy_ranges.jpg"
        path = os.path.join(savefolder, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")

    # plt.show()
    plt.close()


def plot_2D_prediction_overlay(
    truth1,
    nn_reco1,
    truth2,
    nn_reco2,
    save=False,
    savefolder=None,
    weights1=None,
    weights2=None,
    syst_set="",
    bins=60,
    minval=None,
    maxval=None,
    switch_axis=False,
    cut_truth=False,
    axis_square=False,
    zmax=None,
    log=False,
    variable="Zenith",
    units="",
    epochs=None,
    reco_name1="CNN-pulse",
    reco_name2="FLERCNN plus BDT",
):
    """
    Plot testing set reconstruction vs truth for two sets of data
    """
    bins = 60

    path = os.path.join(savefolder, "cnn_2d_hist_errorbars_overlay.jpg")

    if not minval:
        minval = min([min(nn_reco1), min(nn_reco2), min(truth1), min(truth2)])
    if not maxval:
        maxval = max([max(nn_reco1), max(nn_reco2), max(truth1), max(truth2)])

    fig, ax = plt.subplots(figsize=(12, 6))

    fig.suptitle(
        "Convolutional Neural Network Comparison", fontweight=700, fontsize=14, x=0.55
    )

    # Plot first set
    cts1, xbin, ybin, img1 = ax.hist2d(
        truth1,
        nn_reco1,
        bins=bins,
        range=[[minval, maxval], [minval, maxval]],
        cmap="Blues",
        weights=weights1,
        cmax=zmax,
        cmin=1e-12,
        alpha=0.6,
    )

    # Plot second set
    cts2, _, _, img2 = ax.hist2d(
        truth2,
        nn_reco2,
        cmap="Reds",
        weights=weights2,
        cmax=zmax,
        bins=bins,
        cmin=1e-12,
        alpha=0.6,
    )

    cbar1 = fig.colorbar(img1, ax=ax, location="left", pad=0.15)
    cbar1.set_label(f"{reco_name1} Event Count", rotation=270, labelpad=15)

    cbar2 = fig.colorbar(img2, ax=ax, location="right", pad=0.15)
    cbar2.set_label(f"{reco_name2} Event Count", rotation=270, labelpad=15)

    ax.set_xlabel("True Inelasticity")
    ax.set_ylabel("Reconstructed Inelasticity")

    # Plot 1:1 line
    ax.plot([minval, maxval], [minval, maxval], "k:", linewidth=2, label="Ideal")

    # Plot medians and 68% bands for both sets
    for truth, nn_reco, color, label in [
        (truth1, nn_reco1, "blue", reco_name1),
        (truth2, nn_reco2, "red", reco_name2),
    ]:
        x, y, y_l, y_u = find_contours_2D(
            truth, nn_reco, xbin, weights=weights1 if truth is truth1 else weights2
        )
        ax.plot(x, y, color=color, label=f"{label} Median", linewidth=2)
        ax.plot(x, y_l, color=color, linestyle="dashed", linewidth=2)
        ax.plot(x, y_u, color=color, linestyle="dashed", linewidth=2)

    ax.legend(fontsize=12)

    ax.set_xlim([minval, maxval])
    ax.set_ylim([minval, maxval])

    textstr = "IceCube Work in Progress"
    ax.text(0.35, 0.02, textstr, transform=ax.transAxes, color="black")

    if save:
        plt.savefig(path)
    plt.show()
    plt.close()

    return path


def error_against_energy_comparison(
    truth1, reco1, energy1, truth2, reco2, energy2, savefolder
):
    path = os.path.join(savefolder, "cnn_energy_errorbars_comparison.jpg")
    bins = 50
    colors = ["Blues", "Reds"]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("IceCube", fontweight=700, fontsize=16)

    for i, (truth, reco, energy) in enumerate(
        [(truth1, reco1, energy1), (truth2, reco2, energy2)]
    ):
        error = reco - truth
        # Create bins for energy
        energy_bins = np.linspace(np.min(energy), np.max(energy), bins + 1)
        # Calculate median of error for each energy bin
        median_error, bin_edges, _ = binned_statistic(
            energy, error, statistic="median", bins=energy_bins
        )
        # Calculate 2.5th and 97.5th percentiles for 95% confidence interval
        lower_95, _, _ = binned_statistic(
            energy, error, statistic=lambda x: np.percentile(x, 2.5), bins=energy_bins
        )
        upper_95, _, _ = binned_statistic(
            energy, error, statistic=lambda x: np.percentile(x, 97.5), bins=energy_bins
        )
        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Get colors from colormap
        cmap = plt.get_cmap(colors[i])
        fill_color = cmap(0.5)
        line_color = cmap(0.8)

        # Find the bin with the narrowest 95% CI
        ci_width = upper_95 - lower_95
        narrowest_ci_index = np.argmin(ci_width)
        narrowest_ci_energy = bin_centers[narrowest_ci_index]
        narrowest_ci_median = median_error[narrowest_ci_index]
        narrowest_ci_width = ci_width[narrowest_ci_index]

        # Add a point and annotation for the narrowest CI
        ax.plot(
            narrowest_ci_energy,
            narrowest_ci_median,
            "o",
            color=line_color,
            markersize=5,
        )

        if i == 0:
            annotate = "CNN-pulse"
        else:
            annotate = "CNN-reco"
        if i == 0:
            y = 185
        else:
            y = -200

        ax.annotate(
            f"Narrowest 95% CI ({annotate})\nEnergy: {narrowest_ci_energy:.2f}×10²GeV\nCI Width: {narrowest_ci_width:.2f}",
            xy=(narrowest_ci_energy, narrowest_ci_median),
            xytext=(5, y),
            textcoords="offset points",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc=cmap(0.7), alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        # Plot the median line
        ax.plot(
            bin_centers,
            median_error,
            "-",
            color=line_color,
            linewidth=2,
            label=f"Median {annotate}",
        )
        # Plot the 95% confidence interval
        ax.fill_between(
            bin_centers,
            lower_95,
            upper_95,
            color=fill_color,
            alpha=0.3,
            label=f"95% CI",
        )

    # Set labels and title
    ax.set_xlabel("Energy (10² GeV)")
    ax.set_ylabel("Error (Reconstructed - Truth)")
    fig.suptitle("Convolutional Neural Network Comparison", fontweight=700, fontsize=14)
    ax.set_title(
        "Low Energy Muon Inelasticity Reconstruction Error as a Function of Energy",
        fontsize=14,
    )

    # Add a horizontal line at y=0
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.2, linestyle="-")

    # Add legend
    ax.legend(fontsize=12)

    # Adjust y range
    ax.set_ylim(-1, 1)

    # Adjust layout and display plot
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

    return path


def plot_stacked_1d_histograms(
    truth_cnn,
    reco_cnn,
    truth_cnn_bdt,
    reco_cnn_bdt,
    savefolder,
    variable_name="Inelasticity",
    bins=50,
):
    path = os.path.join(savefolder, f"{variable_name}_stacked_1d_histograms.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 15), sharex=True)

    # Determine the range for the histogram
    all_data = np.concatenate([truth_cnn, reco_cnn, truth_cnn_bdt, reco_cnn_bdt])
    min_val = np.min(all_data)
    max_val = np.max(all_data)

    # Create histogram bins
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    # Plot CNN-pulse model (top plot)
    ax1.hist(
        truth_cnn,
        bins=bin_edges,
        # alpha=0.5,
        label="Truth",
        color="blue",
        edgecolor="black",
        histtype="step",
        linewidth=2,
    )
    ax1.hist(
        reco_cnn,
        bins=bin_edges,
        # alpha=0.5,
        label="Reconstructed",
        color="lightblue",
        edgecolor="black",
    )
    ax1.set_title("CNN-pulse Model")
    ax1.legend(fontsize=10)
    ax1.set_ylabel("Occurence")
    ax1.set_yscale("log")

    # Plot CNN plus BDT model (bottom plot)
    ax2.hist(
        truth_cnn_bdt,
        bins=bin_edges,
        # alpha=0.5,
        label="Truth",
        color="red",
        edgecolor="black",
        histtype="step",
        linewidth=2,
    )
    ax2.hist(
        reco_cnn_bdt,
        bins=bin_edges,
        # alpha=0.5,
        label="Reconstructed",
        color="lightcoral",
        edgecolor="black",
    )
    ax2.set_title("CNN-reco Model")
    ax2.legend(fontsize=10)
    ax2.set_xlabel("Inelasticity Range")
    ax2.set_ylabel("Occurence")
    ax2.set_yscale("log")

    # Add IceCube watermark
    fig.text(
        0.22,
        0.94,
        "IceCube Work in Progress",
        fontsize=10,
        horizontalalignment="right",
        verticalalignment="bottom",
        # alpha=0.7,
    )

    plt.subplots_adjust(top=0.90)

    # Set overall title
    fig.suptitle(
        f"Distribution of {variable_name}: Truth vs Reconstruction",
        fontweight=700,
        fontsize=14,
        y=0.99,
    )

    ax1.grid("on", alpha=0.2, linestyle="-")
    ax2.grid("on", alpha=0.2, linestyle="-")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    return path


if __name__ == "__main__":

    pass
