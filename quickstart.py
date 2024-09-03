import argparse
import getpass
import os
import pathlib
import random
import subprocess
import sys

from npbench.infrastructure import (Benchmark, generate_framework, LineCount,
                                    Test, utilities as util, bench_info)


def run_benchmark(benchname, fname, preset, validate, repeat, timeout):
    frmwrk = generate_framework(fname)
    numpy = generate_framework("numpy")
    bench = Benchmark(benchname)
    lcount = LineCount(bench, frmwrk, numpy)
    lcount.count()
    test = Test(bench, frmwrk, numpy)
    test.run(preset, validate, repeat, timeout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--preset",
                        choices=['S', 'M', 'L', 'paper', 'builtin'],
                        nargs="?",
                        default='S')
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-f", "--framework", dest="frameworks", type=str, nargs="+", default=[],
                        choices=['numba', 'numpy', 'cmlq'], action='extend')
    parser.add_argument("-b", "--benchmark", dest="benchmarks", type=str, nargs="+", default=[],
                        action='extend')
    parser.add_argument("-s", "--shield", type=str, action='store')
    parser.add_argument("--cmlq-path", type=str, nargs="?", default="")
    parser.add_argument("-v",
                        "--validate",
                        type=util.str2bool,
                        nargs="?",
                        default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-t", "--timeout", type=float, nargs="?", default=10.0)
    parser.add_argument("-d", "--dace", type=util.str2bool, nargs="?", default=False)
    parser.add_argument("--numpy-threads", type=int, default=16, dest='numpy_threads')
    parser.add_argument("--numba-threads", type=int, default=16, dest='numba_threads')
    parser.add_argument("--test-papi", default=True)
    parser.add_argument("--output", default='table', choices=['table', 'figure', 'stats'])
    parser.add_argument("--database", default='npbench.db')

    subcommands = parser.add_subparsers(dest='subcommand')
    benchmark_parser = subcommands.add_parser("benchmark", description='Run the benchmarks as in the paper.')
    statistics_parser = subcommands.add_parser("statistics", description='Collect statistics as in the paper.')

    parsed = parser.parse_args()

    if parsed.subcommand == 'benchmark' or parsed.subcommand == 'statistics':
        parsed.benchmarks = ['all']
        if parsed.subcommand == 'benchmark':
            parsed.cmlq_path = 'venv_cmq/bin/python'
            parsed.frameworks = ['numpy', 'cmlq']

        if parsed.subcommand == 'statistics':
            parsed.cmlq_path = 'venv_cmq_stats/bin/python'
            parsed.frameworks = ['cmlq']
            parsed.output = 'stats'

        parsed.preset = 'paper'
        parsed.numpy_threads = 1
        parsed.numba_threads = 1
        parsed.validate = True
        parsed.repeat = 20
        parsed.timeout = 120.0
        parsed.test_papi = False

    if len(parsed.benchmarks) == 1 and parsed.benchmarks[0] == "all":
        benchmarks = bench_info.all_benchmarks
    elif len(parsed.benchmarks) == 1 and parsed.benchmarks[0] == "cmlq":
        benchmarks = bench_info.cmlq_benchmarks_short
    elif len(parsed.benchmarks) == 1 and parsed.benchmarks[0] == "npbench":
        benchmarks = bench_info.npbench_benchmarks_short
    elif len(parsed.benchmarks) == 1 and parsed.benchmarks[0] == "phoronix":
        benchmarks = bench_info.phoronix_benchmarks_short
    elif len(parsed.benchmarks) > 0:
        benchmarks = parsed.benchmarks

    if len(benchmarks) == 0:
        raise Exception("No benchmarks selected to run.")

    frameworks = parsed.frameworks
    if len(frameworks) == 0:
        raise Exception("No frameworks selected to run.")

    papi_support = False
    if parsed.cmlq_path is not None and parsed.test_papi:
        result = subprocess.run(["ldd", parsed.cmlq_path], capture_output=True, text=True)
        if result.returncode != 0:
            print("Unable to detect PAPI support: " + result.stderr)
        else:
            for line in result.stdout.splitlines():
                if "papi" in line:
                    papi_support = True
                    print("Found papi library in dependencies of CMLQ CPython, assuming papi support: " + line.strip())
                    break

    if "cmlq" in frameworks and not parsed.cmlq_path:
        print("Need to provide a path to CMLQ python with --cmlq-path", file=sys.stderr)
        sys.exit(1)

    experiment_id = str(random.randint(0, 10000000))
    print(f"{f' Experiment {experiment_id} ':=^140}")
    print("Selected benchmarks: " + ", ".join(benchmarks))

    shield_configured = False
    try:
        if parsed.shield is not None:
            subprocess.run(["sudo", "/usr/bin/cset", "shield", "--cpu", parsed.shield], check=True)
            shield_configured = True

        for benchname in benchmarks:
            for fname in frameworks:
                exe = sys.executable

                env = {"OPENBLAS_NUM_THREADS": str(parsed.numpy_threads),
                       "NUMBA_NUM_THREADS": str(parsed.numba_threads),
                       "OMP_NUM_THREADS": str(parsed.numba_threads)}

                # otherwise we can't see subprocess output in Docker runs
                if os.getenv("PYTHONUNBUFFERED", 'False').lower() in ('true', '1', 't'):
                    env["PYTHONUNBUFFERED"] = "1"

                if fname == "cmlq":
                    exe = parsed.cmlq_path
                    if papi_support:
                        papi_path = pathlib.Path("papi_results", benchname).absolute()
                        print("Writing PAPI results to " + str(papi_path))
                        papi_path.mkdir(parents=True, exist_ok=True)
                        env["PAPI_OUTPUT_DIRECTORY"] = str(papi_path)

                cmd = [exe, "run_benchmark.py", "--database", parsed.database, "-f", fname, "--preset", parsed.preset, "--validate",
                       str(parsed.validate),
                       "--repeat",
                       str(parsed.repeat), "--timeout", str(parsed.timeout), "-b", benchname, "--experiment-id",
                       experiment_id]

                if parsed.shield:
                    cmd.insert(1, "--")
                    # NOTE: os.getuser() does not work on the fuzzing machines
                    cmd = ["sudo", "-E", "/usr/bin/cset", "proc", "--user", getpass.getuser(),
                           "--exec", "user"] + cmd

                print(f"Running: {' '.join(cmd)}")
                proc = subprocess.run(cmd, env=env)
    finally:
        if shield_configured:
            subprocess.run(["sudo", "/usr/bin/cset", "shield", "--reset"], check=True)

    print(f"{f' Experiment {experiment_id} ':=^140}")


    if parsed.output in ('table', 'figure'):
        if parsed.output == 'table':
            type = "results-table"

        if parsed.output == 'figure':
            type = "results-figure"

        subprocess.run([sys.executable, "evaluation_data.py", "--type", type,
                        "--names", f"AE:{experiment_id}",
                        "--database", parsed.database])

    if parsed.output == 'stats':
        subprocess.run([sys.executable, "evaluation_data.py", "--type", "statistics",
                        "--database", parsed.database])

    # numpy = generate_framework("numpy")
    # numba = generate_framework("numba")

    # for benchname in benchmarks:
    #     bench = Benchmark(benchname)
    #     for frmwrk in [numpy, numba]:
    #         lcount = LineCount(bench, frmwrk, numpy)
    #         lcount.count()
    #         test = Test(bench, frmwrk, numpy)
    #         try:
    #             test.run(args["preset"], args["validate"], args["repeat"],
    #                      args["timeout"])
    #         except:
    #             continue
