import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import seaborn as sns

import audeer


scale = 1.0
plt.rcParams["figure.figsize"] = (scale * 6.4, scale * 4.8)
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams["font.size"] = 12

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    simulation_titles = {
        "relative_difference_per_class": "Rel. Diff. per Class",
        "class_shift_diff": "Class Proportion Shift Diff.",
        "relative_difference_per_bin": "Rel. Diff. per Bin",
        "bin_shift_diff": "Bin Proportion Shift Diff.",
        "mean": "Mean Value Diff.",
        "mean_shift_diff": "Mean Shift Diff.",
        "recall_per_class": "Diff. Recall per Class",
        "precision_per_class": "Diff. Precision per Class",
        "recall_per_bin": "Diff. Recall per Bin",
        "precision_per_bin": "Diff. Precision per Bin",
        "ccc": "Diff. CCC",
        "mde": "Diff. Mean Directional Error",
        "mae": "Diff. MAE",
        "recall": "Diff. UAR",
        "precision": "Diff. UAP",
    }
    plot_dir = audeer.mkdir(audeer.path(CURRENT_DIR, "plots"))
    n_samples = [
        20,
        40,
        60,
        80,
        100,
        120,
        140,
        160,
        180,
        200,
        220,
        240,
        260,
        280,
        300,
        400,
        600,
        800,
        1000,
    ]
    ticks = [20, 100, 200, 300, 400, 600, 800, 1000]

    simulation_file = audeer.path(CURRENT_DIR, "simulation.csv")
    res_df = pd.read_csv(simulation_file)

    res_df["n_groups"] = res_df["n_groups"].astype("str")
    res_df.rename(columns={"n_groups": "Number of groups"}, inplace=True)

    value = "max"
    value_name = "Maximum"
    for simulation, title in simulation_titles.items():
        
        if simulation in [
            "precision",
            "precision_per_class",
            "recall",
            "recall_per_class",
        ]:
            # For these metrics, create plots for a sparse ground truth and a uniform predicition
            plot_df = res_df[
                (res_df["simulation"] == simulation)
                & (res_df["varied"] != True)
                & (res_df["n_samples"].isin(n_samples))
                & (res_df["distribution_pred"] == "uniform")
                & (res_df["distribution_truth"] == "sparse")
            ]
            plot_title = (
                f"{value_name} {simulation_titles[simulation]}\n"
                f"(uniform random model, sparse truth)"
            )
            plot_file = f"{value}_truthsparse_preduniform_{simulation}.png"
        
        elif simulation in ["relative_difference_per_class", "class_shift_diff"]:
            # For these metrics, create plots for a uniform prediction
            plot_df = res_df[
                (res_df["simulation"] == simulation)
                & (res_df["varied"] != True)
                & (res_df["n_samples"].isin(n_samples))
                & (res_df["distribution"] == "uniform")
            ]
            plot_title = (
                f"{value_name} {simulation_titles[simulation]}\n"
                "(uniform random model)"
            )
            plot_file = f"{value}_uniform_{simulation}.png"
        else:
            plot_df = res_df[
                (res_df["simulation"] == simulation)
                & (res_df["varied"] != True)
                & (res_df["n_samples"].isin(n_samples))
            ]
            plot_title = (
                f"{value_name} {simulation_titles[simulation]}\n"
                "(random model)"
            )
            plot_file = f"{value}_{simulation}.png"

        ax = sns.lineplot(
            data=plot_df,
            x="n_samples",
            y=value,
            hue="Number of groups",
        )
        if plot_df[value].max() - plot_df[value].min() > 0.5:
            tick_distance = 0.05
        else:
            tick_distance = 0.025
        ax.yaxis.set_major_locator(MultipleLocator(tick_distance))
        ax.set_ylabel(f"{value_name} {simulation_titles[simulation]}")
        ax.set_xlabel("Number of samples per group")
        ax.set_xticks(ticks)
        ax.set_xlim(n_samples[0], n_samples[-1])
        plt.grid(alpha=0.4)
        plt.title(plot_title)
        sns.despine()
        plt.tight_layout()
        outfile = audeer.path(plot_dir, plot_file)
        plt.savefig(outfile, dpi=150)
        plt.close()
if __name__ == "__main__":
    main()
