import argparse
import math
import os
import warnings
from enum import StrEnum
from pathlib import Path

import inflect
import palettable
import scikits.bootstrap as boot
import scipy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlean as sqlite3
import tol_colors as tc
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, ScalarFormatter, FuncFormatter
from scikits.bootstrap.bootstrap import InstabilityWarning
from scipy.stats.mstats import gmean


def bootstrap_ci_interval_length(data, statfunction=np.median, alpha=0.05, n_samples=1000):
    """inspired by https://github.com/cgevans/scikits-bootstrap"""

    # import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InstabilityWarning)
        start, end = boot.ci(data.array, statfunction, alpha, n_samples)
    return end - start


def prepare_data(conn, mapping):
    # in general a terrible idea not to use a prepared statement, but I trust myself not to provide malicious data ;)
    query = f"""SELECT *, 
                    benchmark in (select distinct benchmark from statistics s where opcode_name like '%EXTERNAL%' or cache_id is not null) as specialized,
                    benchmark in (select distinct benchmark from statistics s where (opcode_name like '%EXTERNAL%' or cache_id is not null) and in_bench) as specialized_in_benchmark
                    FROM results WHERE experiment_id in ({str.join(",", mapping.values())})"""
    data = pd.read_sql_query(query, conn)

    not_validated = data[data['validated'] != True]
    if len(not_validated) > 0:
        print(f"WARNING: {len(not_validated)} not validated")
        print(not_validated.to_markdown())
    data = data.drop(['timestamp', 'kind', 'dwarf', 'version', 'domain', 'validated', 'preset', 'details', 'mode'],
                     axis=1).reset_index(drop=True)
    data['benchmark'] = data['benchmark'].str.replace(r'^ph_', '', regex=True)

    numpy_medians = dict()
    for experiment_id in mapping.values():
        # I am sure there is a better way, but no time to play with pandas :'(
        experiment_data = data[data['experiment_id'] == experiment_id]
        numpy_results = experiment_data[experiment_data['framework'] == 'numpy']
        medians = numpy_results.groupby(["benchmark"], dropna=False).agg({
            "time": "median"
        }).reset_index()
        numpy_medians[experiment_id] = dict(zip(medians['benchmark'], medians['time']))

    def calculate_relative(row):
        row['time'] = numpy_medians[row['experiment_id']][row['benchmark']] / row['time']
        return row

    data = data[data['framework'] != 'numpy']
    data.drop(['framework'], axis=1, inplace=True)
    data = data.apply(calculate_relative, axis=1)
    id_to_name = {value: key for key, value in mapping.items()}
    data['experiment_id'] = data['experiment_id'].apply(lambda x: id_to_name[x])

    data.rename(columns={'experiment_id': 'machine'}, inplace=True)
    data.sort_values(['benchmark', 'machine'], inplace=True, key=lambda col: col.str.lower())
    return data


def print_cache_statistics(conn):
    cache_query = """
    select s.benchmark as benchmark,
       sum(result_cache_misses) as cache_misses,
       sum(shape_misses)        as shape_misses,
       sum(ndims_misses)        as ndim_misses,
       sum(refcnt_misses)       as refcnt_misses
from statistics s
         join cache c on s.cache_id = c.id
group by s.benchmark;
    """
    cache_statistics = pd.read_sql_query(cache_query, conn)

    def miss_reason(row):
        if row['shape_misses'] > 0:
            return 'SHAPE'

        if row['ndim_misses'] > 0:
            return 'NDIMS'

        if row['refcnt_misses'] > 0:
            return 'REFCNT'

    cache_statistics['Reason'] = cache_statistics.apply(miss_reason, axis=1)
    cache_statistics.drop(['shape_misses', 'ndim_misses', 'refcnt_misses'], axis=1, inplace=True)
    cache_statistics.rename(columns={'benchmark': 'Benchmark', 'cache_misses': 'Total Cache Misses'}, inplace=True)
    print(cache_statistics.to_markdown())


def draw_graph(data, benchmark_filter, mapping, noise_tolerance, paper_path=None, output_name=None):
    cset = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    plt.rc('axes', prop_cycle=plt.cycler('color', list(cset)))
    plt.rcParams.update({
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
    })

    sns.set_context("paper", rc={
        "font.family": "serif",
        "pgf.rcfonts": False,
        "font.size": 6,
        "axes.labelsize": 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.titlesize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.5,
        ## 'ytick.major.pad': 20,
        ## "figure.autolayout": True,
    })

    ylimit = 0.5

    if benchmark_filter is not BenchmarkFilter.ALL:
        data = data[data[str(benchmark_filter)] == 1]
        grouped_df = data.groupby(['benchmark', benchmark_filter]).first().reset_index()

    benchmarks = list(data['benchmark'].unique())
    last_benchmark_in_first_half = benchmarks[int(len(benchmarks) / 2)]
    first_benchmark_in_second_half = benchmarks[int(len(benchmarks) / 2) + 1]

    last_in_first_half = data[data['benchmark'] == last_benchmark_in_first_half].iloc[-1]
    first_in_second_half = data[data['benchmark'] == first_benchmark_in_second_half].iloc[0]

    f, (ax1, ax2) = plt.subplots(figsize=(5.5, 4), ncols=1, nrows=2)

    # scale the factor to be negative for slowdowns
    data['improvement'] = data['time'].apply(lambda x: x - 1)

    first_half = data.loc[:last_in_first_half.name]
    second_half = data.loc[first_in_second_half.name:]

    sns.barplot(y="improvement", x="benchmark", hue="machine", data=first_half, zorder=2,
                errorbar=("ci", 95), estimator='median', ax=ax1, n_boot=1000, seed=10)

    sns.barplot(y="improvement", x="benchmark", hue="machine", data=second_half, zorder=2,
                errorbar=("ci", 95), estimator='median', ax=ax2, n_boot=1000, seed=10)

    def add_decoration(axis):

        offset = -0.10
        x_labels = [label.get_text() for label in axis.get_xticklabels()]

        for machine in axis.containers:
            for idx, rect in enumerate(machine):
                if not benchmark_filter:
                    bench_name = x_labels[idx]
                    if grouped_df[grouped_df["benchmark"] == bench_name][str(benchmark_filter)].iloc[0] == 0:
                        rect.set_hatch('xxx')
                height = rect.get_height()
                if height > ylimit:
                    d = .03
                    kwargs = dict(color="k", linewidth=0.5, clip_on=False)
                    axis.plot(
                        (rect.get_x() - d - 0.05 + rect.get_width() / 2 - 0.05,
                         rect.get_x() + d + 0.05 + rect.get_width() / 2 - 0.05),
                        (ylimit - d + 0.01, ylimit + d - 0.01),
                        **kwargs)
                    axis.plot(
                        (rect.get_x() - d - 0.05 + rect.get_width() / 2 + 0.05,
                         rect.get_x() + d + 0.05 + rect.get_width() / 2 + 0.05),
                        (ylimit - d + 0.01, ylimit + d - 0.01),
                        **kwargs)
                    # axis.plot((1 - d, 1 + d), (-d - 0.01, +d + 0.01), **kwargs)  # top-right diagonal

                    axis.text(rect.get_x() + offset, ylimit + 0.05, str(round(height + 1, 2)), fontsize=7, rotation=60)
                    offset += 0.10

    def mark_below_y(axis):
        bottom_limit = 1.0
        num_bars = len(axis.containers[0])

        for idx in range(num_bars):
            offset = -0.10
            for group in axis.containers:
                rect = group[idx]
                height = rect.get_height()
                if height <= 1.0 - noise_tolerance:
                    d = .03
                    kwargs = dict(color="k", linewidth=0.5, clip_on=False)
                    axis.plot(
                        (rect.get_x() - d - 0.05 + rect.get_width() / 2 - 0.05,
                         rect.get_x() + d + 0.05 + rect.get_width() / 2 - 0.05),
                        (bottom_limit - d + 0.01, bottom_limit + d - 0.01),
                        **kwargs)
                    axis.plot(
                        (rect.get_x() - d - 0.05 + rect.get_width() / 2 + 0.05,
                         rect.get_x() + d + 0.05 + rect.get_width() / 2 + 0.05),
                        (bottom_limit - d + 0.01, bottom_limit + d - 0.01),
                        **kwargs)

                    axis.text(rect.get_x() + offset, bottom_limit + 0.05, str(round(height, 2)), fontsize=7,
                              rotation=60)
                    offset += 0.10

        return bottom_limit

    def adjust_y(axis):
        """Find the base of the diagram, ignoring a certain noise range"""
        smallest = min([rect.get_height() for group in axis.containers for rect in group])
        if smallest + noise_tolerance >= 1.0:
            return 1.0

        axis.axhline(1.0, color='#d3b3b3', linestyle='dashed', linewidth=1)
        return smallest

    add_decoration(ax1)
    add_decoration(ax2)

    ax1.set_xlabel("")

    ax1.set_ylim(bottom=adjust_y(ax1), top=ylimit)
    ax2.set_ylim(bottom=adjust_y(ax2), top=ylimit)
    ax2.set_xlabel("Benchmark")

    label_replacements = {
        "arc_distance": "arcdist",
        "check_mask": "mask",
        "create_grid": "grid",
        "euclidean_distance_square": "eucl_dist",
        "local_maxima": "local_max",
        "log_likelihood": "log_like",
    }

    def sanitize_bench_name(name):
        escaped = name.replace("_", "\\_")
        return f"\\texttt{{{escaped}}}"

    class OffsetScalarFormatter(ScalarFormatter):
        def __call__(self, x, pos=None):
            return super().__call__(x + 1.0, pos)

    for a in (ax1, ax2):
        a.legend(loc='upper right')
        a.set_ylabel("Improvement")
        a.yaxis.grid(True, which='minor', linestyle='dotted', color='gainsboro')
        a.yaxis.set_major_formatter(OffsetScalarFormatter())
        a.yaxis.set_minor_locator(AutoMinorLocator(5))
        a.yaxis.set_major_locator(MultipleLocator(0.1))
        a.yaxis.grid(True, which='major', linestyle='dashed', color='gainsboro')
        a.tick_params(left=True, bottom=True)
        a.tick_params(axis='x', pad=0)

        xtick_labels = [sanitize_bench_name(text) for text in
                        [label.get_text() if label.get_text() not in label_replacements
                         else label_replacements[label.get_text()]
                         for label in a.get_xticklabels()]]
        xticks = a.get_xticks()
        a.set_xticks(xticks, xtick_labels, rotation=55, rotation_mode="anchor", ha='right', va='top')

    ax2.get_legend().remove()

    patches = []
    color_cycler = plt.cycler(color=list(cset))()

    def machine_name_to_latex(name):
        p = inflect.engine()
        translated_name = []
        for char in name:
            if char.isdigit():
                translated_part = p.number_to_words(char)
                translated_name.append(translated_part)
            else:
                translated_name.append(char)
        return ''.join(translated_name)

    # we need to iterate the machines in the dataframe order, otherwise the cycler will
    # return incorrect colors
    latex_defs = ["% This file is generated with a script, do not edit manually\n"]
    for machine in data['machine'].unique():
        geomean = round(scipy.stats.gmean(data[data['machine'] == machine]['time']), 2)
        latex_machine_name = machine_name_to_latex(machine).lower()
        latex_defs.append(f"\\newcommand{{\\geomean{output_name}{latex_machine_name}}}{{{geomean:.2f}}}\n")
        patches.append(matplotlib.patches.Patch(color=next(color_cycler)['color'], label='%.2f' % geomean))

    latex_defs.append(f"\\newcommand{{\\maxSpeedup{output_name}}}{{{data['time'].max():.2f}}}")

    if output_name == "npbench":
        hspace = 0.7
        legend_pos = (0.91, 0.0)
    else:
        hspace = 0.8
        legend_pos = (0.91, -0.05)

    f.subplots_adjust(hspace=hspace)
    f.legend(handles=patches, bbox_to_anchor=legend_pos,
             bbox_transform=plt.gcf().transFigure,
             title='Geometric mean', ncol=len(mapping.keys()), columnspacing=0.7, alignment='center')


    plt.savefig("performance.png", bbox_inches='tight', dpi=600)

    if paper_path is not None:
        figure_path = Path(paper_path) / "figures" / Path("performance-" + output_name + ".pgf")
        defs_path = Path(paper_path) / Path(output_name + "-defs.tex")
        with open(defs_path, 'w') as file:
            file.writelines(latex_defs)
        plt.savefig(figure_path, bbox_inches='tight', dpi=600)

    plt.show()


def print_table(data, output_path, output_name):
    aggregated = data.groupby(["benchmark", "machine"], dropna=False).agg({
        "time": "median"
    }).reset_index()
    aggregated.sort_values(['benchmark', 'machine'], inplace=True, key=lambda col: col.str.lower())
    aggregated = aggregated.pivot_table(index=["benchmark"],
                                        columns=["machine"],
                                        values="time", sort=False).reset_index()

    aggregated.sum(numeric_only=True)
    geomean_row = aggregated.select_dtypes("number").fillna(1).apply(gmean)
    geomean_row['benchmark'] = 'Geomean'
    aggregated["benchmark"] = aggregated["benchmark"].map(lambda b: f"\\propername{{{b}}}")
    aggregated = pd.concat([aggregated, pd.DataFrame([geomean_row], columns=geomean_row.index)])

    aggregated = aggregated.style.hide(axis="index").format(precision=2)

    latex_table = aggregated.to_latex(hrules=True)
    latex_table_lines = latex_table.splitlines()
    latex_table_lines.insert(len(latex_table_lines) - 3, "\\midrule")
    latex_table = str.join("\n", latex_table_lines)
    print(latex_table)

    if output_path is not None:
        with open(Path(output_path) / "src" / Path("benchmark-table-" + output_name + ".tex"), 'w') as file:
            file.writelines(latex_table)


class KeyValParser(argparse.Action):
    def __call__(self, parser, namespace, parts, option_string=None):
        setattr(namespace, self.dest, dict())
        for part in parts:
            key, value = part.split(':')
            getattr(namespace, self.dest)[key] = value


class BenchmarkFilter(StrEnum):
    ALL = 'all',  # notice the trailing comma
    SPECIALIZED = 'specialized'
    SPECIALIZED_IN_BENCH = 'specialized_in_benchmark'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--preset",
                        choices=['S', 'M', 'L', 'paper', 'builtin'],
                        nargs="?",
                        default='S')
    parser.add_argument("--output-name")
    parser.add_argument("--noise-tolerance", default=0.03, type=float)
    parser.add_argument('--type', choices=['results-table', 'results-figure', 'statistics'], default='results-ascii')
    parser.add_argument('--include-benchmarks', action='store', default=BenchmarkFilter.SPECIALIZED_IN_BENCH)
    parser.add_argument('--names', nargs='*', action=KeyValParser)
    parser.add_argument('--paper-path', nargs='?', default=None)
    parser.add_argument("--database", nargs='?', default="npbench.db")
    parsed = parser.parse_args()
    args = vars(parsed)

    # create a database connection
    database = parsed.database
    conn = None
    # standard sqlite does not have statistical functions, we use the sqlean wrapper
    sqlite3.extensions.enable_all()

    try:
        conn = sqlite3.connect(database)
    except sqlite3.Error as e:
        print(e)

    if parsed.type == 'results-figure':
        data = prepare_data(conn, parsed.names)
        draw_graph(data, parsed.include_benchmarks, parsed.names, parsed.noise_tolerance, parsed.paper_path,
                   parsed.output_name)

    if parsed.type == 'results-table':
        data = prepare_data(conn, parsed.names)
        print_table(data, parsed.paper_path, parsed.output_name)

    if parsed.type == 'statistics':
        print_cache_statistics(conn)


if __name__ == "__main__":
    main()
