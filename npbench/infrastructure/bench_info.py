import json
import pathlib

from bidict import bidict

mod_to_long = bidict()
phoronix_benchmarks_short = []

for path in pathlib.Path("bench_info").iterdir():
    if path.is_file() and path.name.endswith(".json"):
        with open(str(path), 'r') as f:
            metadata = json.load(f)
            module_name = metadata['benchmark']['module_name']
            if "phoronix" in metadata['benchmark']['relative_path']:
                phoronix_benchmarks_short.append(metadata['benchmark']['name'])
            else:
                name = metadata['benchmark']['name']
                mod_to_long[module_name] = name


npbench_benchmarks_short = [
    'adi', 'adist', 'atax', 'azimhist', 'bicg', 'cavtflow',
    'cholesky2', 'compute', 'doitgen', 'floydwar', 'gemm', 'gemver',
    'gesummv', 'npgofast', 'hdiff', 'jacobi2d', 'lenet', 'syr2k', 'trmm',
    'vadv'
]

cmlq_benchmarks_short = ["adi", "adist", "azimnaiv", "cavtflow", "chanflow", "durbin", "fdtd_2d", "floydwar", "gemver",
                         "gramschm", "hdiff", "jacobi1d", "jacobi2d", "mandel1", "mandel2", "nbody", "symm", "syr2k",
                         "syrk", "vadv"]

all_benchmarks = mod_to_long.keys()
