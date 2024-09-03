import argparse
import math
import sys
import warnings

import pandas as pd
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter
from scikits.bootstrap.bootstrap import InstabilityWarning
import seaborn as sns
import palettable

matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

from scipy.stats.mstats import gmean
from npbench.infrastructure import utilities as util, bench_info

import tabulate


def bootstrap_ci_interval_length(data, statfunction=np.median, alpha=0.05, n_samples=1000):
    """inspired by https://github.com/cgevans/scikits-bootstrap"""

    # import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InstabilityWarning)
        start, end = boot.ci(data.array, statfunction, alpha, n_samples)
    return end - start


def draw_graph(data):
    def combine_framework_details(framework, details):
        if details == 'default':
            return framework.strip()

        return f"{framework} {details}".strip()

    # Filter by preset
    data = data[data['preset'] == args['preset']]
    data['framework'] = list([combine_framework_details(*pair) for pair in zip(data['framework'], data['details'])])
    data = data.drop(['preset', 'mode', 'domain', 'details', 'experiment_id'], axis=1).reset_index(drop=True)
    data = data[data['benchmark'].isin(bench_info.npbench_benchmarks_short)]
    plt.figure(figsize=(10.0, 2.5))
    plt.rcParams['pgf.texsystem'] = "pdflatex"
    sns.set_context("paper", rc={
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": 12,
        "axes.labelsize": 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.titlesize': 16,
        ## 'ytick.major.pad': 20,
        ## "figure.autolayout": True,
    })
    colors = palettable.colorbrewer.qualitative.Set1_4.mpl_colors
    data = data[data['framework'].isin({'numpy', 'numba nopython-mode', 'cmlq'})]
    numpy_results = data[data['framework'] == 'numpy']
    if len(numpy_results) == 0:
        raise Exception("'numpy' framework not found in data")

    numpy_medians = numpy_results.groupby(["benchmark"], dropna=False).agg({
        "time": "median"
    }).reset_index()
    numpy_medians = dict(zip(numpy_medians['benchmark'], numpy_medians['time']))
    data = data[data['framework'] != 'numpy']

    def calculate_relative(row):
        row['time'] = numpy_medians[row['benchmark']] / row['time'] - 1
        return row

    data = data.apply(calculate_relative, axis=1)
    axis = sns.barplot(y="time", x="benchmark", hue="framework", data=data, palette=colors, zorder=2,
                       errorbar=("pi", 95))
    # axis.yaxis.grid(True, which='minor', linestyle='dotted', color='gainsboro')
    # axis.yaxis.set_minor_locator(AutoMinorLocator())
    # axis.yaxis.grid(True, which='major', linestyle='dashed', color='gainsboro')
    # axis.yaxis.set_major_locator(MultipleLocator(5))
    # axis.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    # axis.tick_params(left=False, bottom=True)
    for item in axis.get_xticklabels():
        item.set_rotation(90)
        item.set_ha('right')
        item.set_verticalalignment("top")
    #
    # bottom, top = plt.ylim()
    # plt.ylim(-1, top)
    #
    # plt.xlabel("Performance Impact (%)")
    # plt.ylabel(None)
    #
    # ## plot_name = metric_name.replace("..", "-") + '-relative.' + format
    # h, l = axis.get_legend_handles_labels()
    # axis.legend(h, name_mapping.values(), title='Configuration', loc='lower right', prop={'size': 14},
    #             title_fontsize=14)
    # plt.tight_layout()
    plt.show()


def print_table(data):
    # for each framework and benchmark and configuration get the median value of all runs
    data = data.drop(['domain', 'mode'], axis=1)
    aggregated = data.groupby(["benchmark", "framework", "details"], dropna=False).agg({
        "time": "median"
    }).reset_index()

    frameworks = list(data['framework'].unique())
    if 'numpy' not in frameworks:
        raise Exception("'numpy' framework not found in data")

    # the resulting dataframe has a multiindex consisting of framework name and details
    perf_in_columns = aggregated.pivot_table(index=["benchmark"],
                                             columns=["framework", "details"],
                                             values="time").reset_index()

    columns = perf_in_columns.head()
    framework_columns = list([c for c in columns if c[0] in frameworks])
    numpy_column = list([c for c in framework_columns if c[0] == 'numpy'])[0]

    # calculate the improvement over numpy
    best_wide_time = perf_in_columns.copy(deep=True)
    for f in framework_columns:
        # for numpy we want to keep the time
        if f[0] == "numpy":
            continue
        perf_in_columns[f] = best_wide_time[numpy_column] / perf_in_columns[f]

    # best_wide.rename(columns=lambda c: f"{c[0]} {c[1]}".strip(), inplace=True)

    # compute ci-size in a separate dataframe
    cidata = (data.groupby(["benchmark", "details", "framework"], dropna=False)
              .agg({"time": [bootstrap_ci_interval_length, "median", "std"]}).reset_index())
    # cidata.columns = ['_'.join(col).strip() for col in cidata.columns.values]
    cidata['perc'] = (cidata[('time', 'bootstrap_ci_interval_length')] / cidata[('time', 'median')]) * 100
    cidata['stdd'] = cidata[('time', 'std')]
    cidata.columns = cidata.columns.droplevel(1)
    cidata_in_columns = cidata.pivot_table(index=["benchmark"],
                                           columns=["framework", "details"],
                                           values=["perc", "stdd"]).reset_index()

    def flatten_columns(column):
        if len(column) == 3:
            first, second, third = column
        else:
            first, second = column

        if second == 'default':
            second = ''

        name = f"{first} {second}".strip()

        return name

    perf_in_columns.columns = perf_in_columns.columns.map(flatten_columns)
    perf_in_columns.set_index('benchmark', inplace=True)
    cidata_in_columns.columns = cidata_in_columns.columns.map(flatten_columns)
    cidata_in_columns.set_index('benchmark', inplace=True)

    # joined = frameworks_in_columns.join(cidata, on='benchmark')

    cleaned_data = perf_in_columns.join(cidata_in_columns, on='benchmark', how='inner', rsuffix='ci')
    print(cleaned_data.to_markdown())
    print(f"Geomean improvement: {scipy.stats.gmean(cleaned_data['cmlq'].dropna())}")
    print(f"Max improvement: {scipy.stats.tmax(cleaned_data['cmlq'].dropna())}")
    print(f"Min improvement: {scipy.stats.tmin(cleaned_data['cmlq'].dropna())}")


def my_round(x, width):
    float_format = "{:." + f"{width}" + "f}"
    return float_format.format(x)


# geomean which ignores NA values
def my_geomean(x):
    x = x.dropna()
    res = gmean(x)
    return res


# make nice/short numbers with up/down indicator
def my_speedup_abbr(x):
    prefix = ""
    label = ""
    if math.isnan(x):
        return ""
    if x < 1:
        prefix = u"\u2191"
        x = 1 / x
    elif x > 1:
        prefix = u"\u2193"
    if x > 100:
        x = int(x)
    if x > 1000:
        label = prefix + str(my_round(x / 1000, 1)) + "k"
    else:
        label = prefix + str(my_round(x, 1))
    return str(label)


# make nice/short runtime numbers with seconds / milliseconds
def my_runtime_abbr(x):
    suffix = " s"
    if math.isnan(x):
        return ""
    if x < 0.1:
        x = x * 1000
        suffix = " ms"
    return str(my_round(x, 2)) + suffix


import scikits.bootstrap as boot
import scipy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--preset",
                        choices=['S', 'M', 'L', 'paper', 'builtin'],
                        nargs="?",
                        default='S')
    parser.add_argument("--experiment-id",
                        dest='experiment_id')
    parser.add_argument('--ascii', action='store_true')
    parser.add_argument("DATABASE", default="npbench.db")
    parsed = parser.parse_args()
    args = vars(parsed)

# create a database connection
database = parsed.DATABASE
conn = util.create_connection(database)
filter = "1=1"
params = []
if args["experiment_id"]:
    filter = f"experiment_id=?"
    params = [args["experiment_id"]]
data = pd.read_sql_query(f"SELECT * FROM results WHERE {filter}", conn, params=params)

# get rid of kind and dwarf, we don't use them
data = data.drop(['timestamp', 'kind', 'dwarf', 'version'],
                 axis=1).reset_index(drop=True)

# remove everything that does not validate, then get rid of validated column
not_validated = data[data['validated'] != True]

if len(not_validated) > 0:
    print(f"WARNING: {len(not_validated)} not validated")
    print(not_validated.to_markdown())

data = data[data['validated'] == True]
data = data.drop(['validated'], axis=1).reset_index(drop=True)

if not parsed.ascii:
    draw_graph(data)
else:
    print_table(data)

#
#
#
# plt.figure()
# plt.rcParams['pgf.texsystem'] = "pdflatex"
#
# sns.set_context("paper", rc={
#     "font.family": "serif",
#     "text.usetex": True,
#     "pgf.rcfonts": False,
#     "font.size": 12,
#     "axes.labelsize": 13,
#     'axes.titlesize': 13,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'legend.titlesize': 16,
#     ## 'ytick.major.pad': 20,
#     ## "figure.autolayout": True,
# })
#
# colors = palettable.colorbrewer.qualitative.Set1_4.mpl_colors
#
# axis = sns.barplot(y="benchmark", x="value", hue="variable", data=cleaned_data,
#                    palette=colors,
#                    zorder=2)
#
# axis.xaxis.grid(True, which='minor', linestyle='dotted', color='gainsboro')
# axis.xaxis.set_minor_locator(AutoMinorLocator())
# axis.xaxis.grid(True, which='major', linestyle='dashed', color='gainsboro')
# axis.xaxis.set_major_locator(MultipleLocator(5))
# axis.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# axis.tick_params(left=False, bottom=True)
#
# for item in axis.get_yticklabels():
#     ## item.set_rotation(30)
#     item.set_ha('left')
#     item.set_verticalalignment("center")
#
# bottom, top = plt.xlim()
# plt.xlim(0, top)
#
# plt.xlabel("Performance Impact (%)")
# plt.ylabel(None)
#
# ## plot_name = metric_name.replace("..", "-") + '-relative.' + format
# h, l = axis.get_legend_handles_labels()
#
# # axis.legend(h, name_mapping.values(), title='Configuration', loc='lower right', prop={'size': 14},
# #             title_fontsize=14)
# plt.tight_layout()
#
# plt.show()
# # we dont care about the rest for now
# sys.exit(0)
#
# # overall = perf_in_columns.drop(['domain'], axis=1)
# # overall = pd.melt(overall, [
# #     'benchmark',
# # ])
# # overall = overall.groupby(['framework']).value.apply(my_geomean).reset_index(
# # )  # this throws warnings if NA is found, which is ok
# # overall_wide = overall.pivot_table(columns="framework",
# #                                    values="value",
# #                                    dropna=False).reset_index(drop=True)
# # overall_wide = overall_wide[frameworks]
# #
# # overall_time = best_wide_time.drop(['domain'], axis=1)
# # overall_time = pd.melt(overall_time, ['benchmark'])
# # overall_time = overall_time.groupby(
# #     ['framework']).value.apply(my_geomean).reset_index(
# # )  # this throws warnings if NA is found, which is ok
# # overall_time_wide = overall_time.pivot_table(
# #     columns="framework", values="value", dropna=False).reset_index(drop=True)
# # overall_time_wide = overall_wide[frameworks]
# #
# # plt.style.use('classic')
# # figsz = (len(frameworks) + 1, 12)
# # fig, (ax2, ax1) = plt.subplots(2,
# #                                1,
# #                                figsize=figsz,
# #                                sharex=True,
# #                                gridspec_kw={'height_ratios': [0.1, 5.7]})
# #
# # hm_data_all = overall_wide
# # im0 = ax2.imshow(hm_data_all.to_numpy(),
# #                  cmap='RdYlGn_r',
# #                  interpolation='nearest',
# #                  vmin=0,
# #                  vmax=2,
# #                  aspect="auto")
# # ax2.set_yticks(np.arange(1))
# # ax2.set_yticklabels(["Total"])
# # for j in range(len(overall_wide.columns)):
# #     if j < len(overall_wide.columns) - 1:
# #         label = hm_data_all.to_numpy()[0, j]
# #         t = label
# #         if t < 1:
# #             t = 1 / t
# #         if t < 1.3:
# #             text = ax2.text(j,
# #                             0,
# #                             my_speedup_abbr(label),
# #                             ha="center",
# #                             va="center",
# #                             color="grey",
# #                             fontsize=8)
# #         else:
# #             text = ax2.text(j,
# #                             0,
# #                             my_speedup_abbr(label),
# #                             ha="center",
# #                             va="center",
# #                             color="white",
# #                             fontsize=8)
# #     else:
# #         label = overall_time_wide['numpy'].to_numpy()[0]
# #
# # # plot benchmark heatmap
# # hm_data = perf_in_columns.drop(['benchmark', 'domain'], axis=1)
# # im = ax1.imshow(hm_data.to_numpy(),
# #                 cmap='RdYlGn_r',
# #                 interpolation='nearest',
# #                 vmin=0,
# #                 vmax=2,
# #                 aspect="auto")
# #
# # # We want to show all ticks...
# # ticks = ax1.set_xticks(np.arange(len(hm_data.columns)))
# # ticks = ax1.set_yticks(np.arange(len(perf_in_columns['benchmark'])))
# # # ... and label them with the respective list entries
# # ticks = ax1.set_xticklabels(hm_data.columns)
# # ticks = ax1.set_yticklabels(perf_in_columns['benchmark'])
# #
# # # Rotate the tick labels and set their alignment.
# # plt.setp(ax1.get_xticklabels(),
# #          rotation=90,
# #          ha="right",
# #          rotation_mode="anchor")
# #
# # for i in range(len(perf_in_columns['benchmark'])):
# #     # annotate with improvement over numpy
# #     for j in range(len(hm_data.columns)):
# #         b = perf_in_columns['benchmark'][i]
# #         f = hm_data.columns[j]
# #         if j < len(hm_data.columns) - 1:
# #             label = hm_data.to_numpy()[i, j]
# #             if math.isnan(label):
# #                 r = ""
# #                 if len(r) > 0:
# #                     text = ax1.text(j,
# #                                     i,
# #                                     str(r.to_numpy()[0]),
# #                                     ha="center",
# #                                     va="center",
# #                                     color="red",
# #                                     fontsize=7)
# #             else:
# #                 p = cidata[(cidata['framework_'] == f)
# #                            & (cidata['benchmark_'] == b)]['perc']
# #                 ci = int(p.to_numpy()[0])
# #                 if ci > 0:
# #                     ci = "$^{(" + str(ci) + ")}$"
# #                 else:
# #                     ci = ""
# #                 t = label
# #                 if t < 1:
# #                     t = 1 / t
# #                 if t < 1.3:
# #                     text = ax1.text(j,
# #                                     i,
# #                                     my_speedup_abbr(label) + ci,
# #                                     ha="center",
# #                                     va="center",
# #                                     color="grey",
# #                                     fontsize=8)
# #                 else:
# #                     text = ax1.text(j,
# #                                     i,
# #                                     my_speedup_abbr(label) + ci,
# #                                     ha="center",
# #                                     va="center",
# #                                     color="white",
# #                                     fontsize=8)
# #         else:
# #             label = best_wide_time['numpy'].to_numpy()[i]
# #             p = cidata[(cidata['framework_'] == f)
# #                        & (cidata['benchmark_'] == b)]['perc']
# #             try:
# #                 ci = int(p.to_numpy()[0])
# #                 if ci > 0:
# #                     ci = "$^{(" + str(ci) + ")}$"
# #                 else:
# #                     ci = ""
# #             except:
# #                 pass
# #             finally:
# #                 ci = ""
# #             text = ax1.text(j,
# #                             i,
# #                             my_runtime_abbr(label) + ci,
# #                             ha="center",
# #                             va="center",
# #                             color="black",
# #                             fontsize=8)
# #
# # ax1.set_ylabel("Benchmarks", labelpad=0)
# #
# # plt.tight_layout()
# # plt.savefig("heatmap.pdf", dpi=600)
# # plt.show()
