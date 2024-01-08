import pandas as pd
import stumpy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.signal import butter, filtfilt, savgol_filter, resample
from numba import cuda

main_dir = "/storage/datasets_public/InfantPA/"
data_file = "data/"
sub_files = ["0_4/", "4_6/", "6_9/", "9_12/", "12_18/"]
res_file = "results/motifs/new_groups/"

all_gpu_devices = [device.id for device in cuda.list_devices()]

AR_good = {
    "AR_03",
    "AR_04",
    "AR_05",
    "AR_06",
    "AR_08",
    "AR_09",
    "AR_10",
    "AR_13",
    "AR_24",
    "AR_26",
}
AR_poor = {
    "AR_02",
    "AR_11",
    "AR_12",
    "AR_14",
    "AR_16",
    "AR_17",
    "AR_19",
    "AR_22",
    "AR_25",
}


def apply_preprocessing(
    data, cutoff_frequency=1, sampling_rate=20, window_size=11, degree=2
):
    nyquist_frequency = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(4, normal_cutoff, btype="low", analog=False)
    f_data = filtfilt(b, a, data)
    f_data = np.abs(f_data)
    downsampled_data = resample(f_data, int(len(f_data) * (1 / 20)))
    # norm_data = savgol_filter(rec_data, window_size, degree)
    return downsampled_data


def motif_discovery(task="TD vs AR"):
    ## Classification Task: TD vs AR
    if task == "TD vs AR":
        for sfile in sub_files:
            print("Processing ", sfile)
            dataset = []
            labels = []
            Hz = 20
            tags = []
            for _, _, files in os.walk(main_dir + data_file + sfile):
                for f in files:
                    if f[-4:] != ".csv":
                        print("Skipping file..", f)
                    else:
                        d = pd.read_csv(
                            main_dir + data_file + sfile + f, sep=",", header=[0]
                        )
                        # label
                        labels.append(1 if f[:2] == "AR" else 0)
                        labels.append(
                            1 if f[:2] == "AR" else 0
                        )  # naively append two times for both left and right leg
                        tags.append(f[:-4])
                        tags.append(f[:-4])
                        # left
                        d_left = d["left_acc"].values
                        d_left = apply_preprocessing(d_left)
                        dataset.append(d_left)
                        # right
                        d_right = d["right_acc"].values
                        d_right = apply_preprocessing(d_right)
                        dataset.append(d_right)
            # extract motifs for each group
            extract_motifs(tags, labels, dataset, sfile)
    ## Classification Task: AR with good outcomes vs AR with poor outcomes
    elif task == "ARgood vs ARpoor":
        dataset = []
        labels = []
        Hz = 20
        tags = []
        for sfile in sub_files:
            for _, _, files in os.walk(main_dir + data_file + sfile):
                for f in files:
                    if (
                        f[-4:] != ".csv"
                        or f[:2] == "TD"
                        or (f[:5] not in AR_good and f[:5] not in AR_poor)
                    ):
                        print(f)
                    else:
                        d = pd.read_csv(
                            main_dir + data_file + sfile + f, sep=",", header=[0]
                        )
                        # label
                        labels.append(1 if f[:5] in AR_poor else 0)
                        labels.append(
                            1 if f[:5] in AR_poor else 0
                        )  # naively append two times for both left and right leg
                        tags.append(f + main_dir + data_file + sfile)
                        tags.append(f + main_dir + data_file + sfile)
                        # left
                        d_left = d["left_acc"].values
                        d_left = apply_preprocessing(d_left)
                        dataset.append(d_left)
                        # right
                        d_right = d["right_acc"].values
                        d_right = apply_preprocessing(d_right)
                        dataset.append(d_right)
        # extract motifs for all AR infants
        extract_motifs(tags, labels, dataset, dir_name="all_AR_2/")


def extract_motifs(tags, labels, dataset, dir_name):
    Hz = 1
    windows = [Hz * 10, Hz * 20, Hz * 30, Hz * 40, Hz * 50, Hz * 60]
    min_neighbors = [1]
    n_motifs = 10
    N = len(dataset)
    mp = {}

    for m in windows:
        for _ in min_neighbors:
            params = {"window": m, "n_motifs": n_motifs}
            print("Matrix Profile for ", m, "...")
            for idx in range(0, N, 2):
                # compute motifs
                L = dataset[idx]
                R = dataset[idx + 1]
                key = tags[idx]
                y = labels[idx]
                P1 = stumpy.gpu_stump(L, m)
                _, indices1 = stumpy.motifs(L, P1[:, 0])
                P2 = stumpy.gpu_stump(R, m)
                _, indices2 = stumpy.motifs(R, P2[:, 0])
                mp[key] = {
                    "Left_idx": indices1,
                    "Right_idx": indices2,
                    "Right": R,
                    "Left": L,
                    "y": y,
                }

            all_data = {"params": params, "result": mp}
            file = str(params["window"])

            with open(
                main_dir + res_file + dir_name + file + ".pickle", "wb"
            ) as handle:
                pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


motif_discovery(task="ARgood vs ARpoor")
