import os
import pandas
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root_files = r"_transcripts/csv"

# prepare list of dataframes by control groups
dfs = []
for csv in os.listdir(os.path.join(root_files)):
    one_csv = root_files + r"/" + csv
    df_from_csv = pandas.read_csv(one_csv)
    df_only_wer = df_from_csv[~df_from_csv.WER.isnull()]
    dfs.append(df_only_wer)

# plot WER data
matplotlib.rcParams["figure.figsize"] = [9, 6]
matplotlib.rcParams["figure.dpi"] = 200.0
labels = ["YHC", "EHC", "PPD"]
typy_nahravek = ["B", "FB", "PR"]
colors = ["aquamarine", "dodgerblue", "crimson"]

wer_b = []
wer_fb = []
wer_pr = []
for df in dfs:
    for typ in typy_nahravek:
        if typ == "B":
            wer_b.append(df[df.code == typ])
        if typ == "FB":
            wer_fb.append(df[df.code == typ])
        if typ == "PR":
            wer_pr.append(df[df.code == typ])

df_list = [wer_b, wer_fb, wer_pr]
for wer_df, typ_nahravky in zip(df_list, typy_nahravek):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    if typ_nahravky == "FB":
        violin_plot = ax.violinplot([dfs[1].WER, dfs[2].WER],
                                    vert=True,
                                    showmedians=True)
    else:
        violin_plot = ax.violinplot([dfs[0].WER, dfs[1].WER, dfs[2].WER],
                                    vert=True,
                                    showmedians=True)

    ax.set_title(f"WER skupin Italského datasetu - nahrávky {typ_nahravky}")
    ax.set_ylabel("Poměr WER")
    ax.set_xlabel("Skupiny")
    ax.yaxis.grid(True)

    if typ_nahravky == "FB":
        ax.legend(labels=["EHC = 1.0", "PPD = 2.0"],
                  loc="upper left",
                  fontsize=12)
    else:
        ax.legend(labels=["YHC = 1.0", "EHC = 2.0", "PPD = 3.0"],
                  loc="upper left",
                  fontsize=12)


    # median_line = Line2D([], [], color="crimson", label="Medián")
    # mean_line = Line2D([], [], color="dodgerblue", label="Průměr", linestyle="--")
    #
    # ax.legend(handles=[median_line, mean_line], loc="upper left", fontsize=12)
    # if typ_nahravky == "FB":
    #     for box, color in zip(bplot["boxes"], [colors[1], colors[2]]):
    #         box.set_facecolor(color)
    # else:
    #     for box, color in zip(bplot["boxes"], colors):
    #         box.set_facecolor(color)

    plt.savefig(f"_plots/violin_plot_{typ_nahravky}.png")
    plt.show()
    plt.close("all")

# boxplot přes všechny druhy nahrávek

fig, ax = plt.subplots(nrows=1, ncols=1)
violin_plot = ax.violinplot([dfs[0].WER, dfs[1].WER, dfs[2].WER],
                            vert=True,
                            showmedians=True)

ax.set_title("WER skupin Italského datasetu - všechny nahrávky")
ax.set_ylabel("Poměr WER")
ax.set_xlabel("Skupiny")
ax.yaxis.grid(True)
ax.legend(labels=["YHC = 1.0", "EHC = 2.0", "PPD = 3.0"],
          loc="upper left",
          fontsize=12)

# median_line = Line2D([], [], color="crimson", label="Medián")
# mean_line = Line2D([], [], color="dodgerblue", label="Průměr", linestyle="--")

plt.savefig(f"_plots/violin_plot_all.png")
plt.show()
plt.close("all")
