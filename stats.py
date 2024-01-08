import numpy as np
from itertools import chain
import stumpy
from scipy.stats import skew, kurtosis, entropy
from scipy.stats import zscore
from scipy.stats import ttest_ind, mannwhitneyu


def sig_test(group_1, group_2, alpha=0.1, label=""):
    print(label, "Group 1:", np.mean(group_1), "Group 2:", np.mean(group_2))
    _, p_value = mannwhitneyu(group_1, group_2)
    if p_value < alpha:
        print("The difference is statistically significant with p =", p_value)
    else:
        print("The difference is not statistically significant with p =", p_value)
    print()


def compute_features(motif, raw_ts, stats_dict, th=1.0):
    stats_dict["min"] += np.min(motif)
    stats_dict["mean"] += np.mean(motif)
    stats_dict["peak"] += np.max(motif)
    stats_dict["std_dev"] += np.std(motif)
    # stats_dict["skewness"] += skew(motif)
    # stats_dict["kurtosis"] += kurtosis(motif)
    # stats_dict["kurtosis"] += kurtosis(motif)
    # fft = np.fft.fft(motif)
    # power_spectrum = np.abs(fft) ** 2
    # stats_dict['periodicity'] += np.sum(power_spectrum > np.mean(power_spectrum))
    matches = stumpy.match(motif, raw_ts, max_distance=th)
    stats_dict["repetition"] += len(matches)


def stats(data, motifs, labels):
    data = list(chain(*data))
    motifs = list(chain(*motifs))
    labels = list(chain(*labels))

    Tstats = {
        "min": 0,
        "peak": 0,
        "mean": 0,
        "std_dev": 0,
        "repetition": 0,
        # "skewness": 0,
    }
    Fstats = {
        "min": 0,
        "peak": 0,
        "mean": 0,
        "std_dev": 0,
        "repetition": 0,
        # "skewness": 0,
    }

    n_true = 0
    n_false = 0
    min_vals = {"true_group": [], "false_group": []}
    peak_vals = {"true_group": [], "false_group": []}
    avg_vals = {"true_group": [], "false_group": []}
    std_vals = {"true_group": [], "false_group": []}
    rep_vals = {"true_group": [], "false_group": []}

    for i, mtf in enumerate(motifs):
        if labels[i] == 0:
            min_vals["false_group"].append(np.min(mtf))
            peak_vals["false_group"].append(np.max(mtf))
            std_vals["false_group"].append(np.std(mtf))
            avg_vals["false_group"].append(np.mean(mtf))
            matches = stumpy.match(mtf, data[i], max_distance=1.0)
            rep_vals["false_group"].append(len(matches))
            n_false += 1
        else:
            min_vals["true_group"].append(np.min(mtf))
            peak_vals["true_group"].append(np.max(mtf))
            std_vals["true_group"].append(np.std(mtf))
            avg_vals["true_group"].append(np.mean(mtf))
            matches = stumpy.match(mtf, data[i], max_distance=1.0)
            rep_vals["true_group"].append(len(matches))
            n_true += 1

    # for i, mtf in enumerate(motifs):
    #     if labels[i] == 0:
    #         compute_features(mtf, data[i], Fstats)
    #         n_false += 1
    #     else:
    #         compute_features(mtf, data[i], Tstats)
    #         n_true += 1

    # print(n_false, n_true)
    # for key in Tstats:
    #     Tstats[key] /= n_true
    # for key in Fstats:
    #     Fstats[key] /= n_false
    alpha = 0.1  # Set your significance level
    sig_test(
        np.array(min_vals["false_group"]),
        np.array(min_vals["true_group"]),
        alpha=alpha,
        label="Min",
    )
    sig_test(
        np.array(peak_vals["false_group"]),
        np.array(peak_vals["true_group"]),
        alpha=alpha,
        label="Peak",
    )
    sig_test(
        np.array(avg_vals["false_group"]),
        np.array(avg_vals["true_group"]),
        alpha=alpha,
        label="Avg",
    )
    sig_test(
        np.array(std_vals["false_group"]),
        np.array(std_vals["true_group"]),
        alpha=alpha,
        label="Std Dev",
    )
    sig_test(
        np.array(rep_vals["false_group"]),
        np.array(rep_vals["true_group"]),
        alpha=alpha,
        label="Repetition",
    )