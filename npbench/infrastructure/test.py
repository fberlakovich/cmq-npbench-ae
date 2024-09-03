# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import socket
import time
import timeit

from npbench.infrastructure import (Benchmark, Framework, timeout_decorator as tout, utilities as util)
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np


class Test(object):
    """ A class for testing a framework on a benchmark. """

    def __init__(self, bench: Benchmark, frmwrk: Framework, npfrmwrk: Framework = None, experiment_id: str = None):
        self.bench = bench
        self.frmwrk = frmwrk
        self.numpy = npfrmwrk
        self.experiment_id = experiment_id

    def _execute(self, frmwrk: Framework, impl: Callable, impl_name: str, mode: str, bdata: Dict[str, Any], repeat: int,
                 ignore_errors: bool) -> Tuple[Any, Sequence[float]]:
        report_str = frmwrk.info["full_name"] + " - " + impl_name
        try:
            copy = frmwrk.copy_func()
            setup_str = frmwrk.setup_str(self.bench, impl)
            exec_str = frmwrk.exec_str(self.bench, impl)
        except Exception as e:
            print("Failed to load the {} implementation.".format(report_str))
            print(e)
            if not ignore_errors:
                raise
            return None, None
        ldict = {'__npb_impl': impl, '__npb_copy': copy, **bdata}
        try:
            if self.bench.is_phoronix_benchmark():
                bench_fun = Test.benchmark_phoronix
            else:
                bench_fun = Test.benchmark
            out, timelist = bench_fun(exec_str, setup_str, report_str + " - " + mode, repeat, ldict,
                                      '__npb_result')
        except Exception as e:
            print("Failed to execute the {} implementation.".format(report_str))
            print(e)
            if not ignore_errors:
                raise
            return None, None
        if out is not None:
            if isinstance(out, (tuple, list)):
                out = list(out)
            else:
                out = [out]
        else:
            out = []
        if "out_args" in self.bench.info.keys():
            out += [ldict[a] for a in self.frmwrk.args(self.bench)]
        return out, timelist

    def run(self, preset: str, validate: bool, repeat: int, timeout: float = 200.0, ignore_errors: bool = True,
            database='npbench.db'):
        """ Tests the framework against the benchmark.
        :param preset: The preset to use for testing (S, M, L, paper).
        :param validate: If true, it validates the output against NumPy.
        :param repeat: The number of repeatitions.
        """
        print("***** Testing {f} with {b} on the {p} dataset *****".format(b=self.bench.bname,
                                                                           f=self.frmwrk.info["full_name"],
                                                                           p=preset))

        gather_statistics = callable(getattr(np.core.multiarray, 'get_cmlq_stats', None))

        if gather_statistics:
            print("Statistics functions found in NumPy installation. Enabling statistics collection...")

        bdata = self.bench.get_data(preset)

        # Run NumPy for validation
        if validate and self.frmwrk.fname != "numpy" and self.numpy:
            np_impl, np_impl_name = self.numpy.implementations(self.bench)[0]
            np_out, _ = self._execute(self.numpy, np_impl, np_impl_name, "validation", bdata, 1, ignore_errors)
        else:
            validate = False
            np_out = None

        # Extra information
        kind = ""
        if "kind" in self.bench.info.keys():
            kind = self.bench.info["kind"]
        domain = ""
        if "domain" in self.bench.info.keys():
            domain = self.bench.info["domain"]
        dwarf = ""
        if "dwarf" in self.bench.info.keys():
            dwarf = self.bench.info["dwarf"]
        version = self.frmwrk.version()

        @tout.exit_after(timeout)
        def first_execution(impl, impl_name):
            return self._execute(self.frmwrk, impl, impl_name, "first/validation", context, 1, ignore_errors)

        bvalues = []
        context = {**bdata, **self.frmwrk.imports()}
        for impl, impl_name in self.frmwrk.implementations(self.bench):
            # First execution
            try:
                frmwrk_out, _ = first_execution(impl, impl_name)
            except KeyboardInterrupt:
                print("Implementation \"{}\" timed out.".format(impl_name), flush=True)
                continue
            except Exception:
                if not ignore_errors:
                    raise
                continue

            # Validation
            valid = True
            if validate and np_out is not None:
                try:
                    frmwrk_name = self.frmwrk.info["full_name"]

                    rtol = 1e-5 if not 'rtol' in self.bench.info else self.bench.info['rtol']
                    atol = 1e-8 if not 'atol' in self.bench.info else self.bench.info['atol']
                    norm_error = 1e-5 if not 'norm_error' in self.bench.info else self.bench.info['norm_error']
                    valid = util.validate(np_out, frmwrk_out, frmwrk_name, rtol=rtol, atol=atol, norm_error=norm_error)
                    if valid:
                        print("{} - {} - validation: SUCCESS".format(frmwrk_name, impl_name))
                    elif not ignore_errors:
                        raise ValueError("{} did not validate!".format(frmwrk_name))
                except Exception:
                    print("Failed to run {} validation.".format(self.frmwrk.info["full_name"]))
                    if not ignore_errors:
                        raise
            # Main execution
            _, timelist = self._execute(self.frmwrk, impl, impl_name, "median", context, repeat, ignore_errors)
            if timelist:
                for t in timelist:
                    bvalues.append(dict(details=impl_name, validated=valid, time=t))

        # create a database connection
        conn = util.create_connection(database)

        # create tables
        if conn is not None:
            # create results table
            util.create_table(conn, util.sql_create_results_table)

            if gather_statistics:
                util.create_table(conn, util.sql_create_statistics_table)
                util.create_table(conn, util.sql_create_cache_table)
        else:
            print("Error! cannot create the database connection.")

        if gather_statistics:
            print(f"Saving statistics to {database}")
            self.frmwrk.save_statistics(conn, self.bench, self.experiment_id)

        # Write data
        timestamp = int(time.time())
        for d in bvalues:
            new_d = {
                'timestamp': timestamp,
                'benchmark': self.bench.info["short_name"],
                'kind': kind,
                'domain': domain,
                'dwarf': dwarf,
                'preset': preset,
                'mode': "main",
                'framework': self.frmwrk.info["simple_name"],
                'version': version,
                'details': d["details"],
                'validated': d["validated"],
                'time': d["time"],
                'experiment_id': self.experiment_id,
                'host': socket.gethostname()
            }
            # print(result)
            util.create_result(conn, util.build_sql_insert("results", new_d), tuple(new_d.values()))

    timeit_tmpl = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        {stmt}
    _t1 = _timer()
    return _t1 - _t0, {output}
    """

    @staticmethod
    def benchmark(stmt, setup="pass", out_text="", repeat=1, context={}, output=None, verbose=True):
        timeit.template = Test.timeit_tmpl.format(init='{init}', setup='{setup}', stmt='{stmt}', output=output)

        ldict = {**context}
        output = timeit.repeat(stmt, setup=setup, repeat=repeat, number=1, globals=ldict)
        res = output[0][1]
        raw_time_list = [a for a, _ in output]
        raw_time = np.median(raw_time_list)
        print(raw_time_list)
        ms_time = util.time_to_ms(raw_time)
        if verbose:
            print("{}: {}ms".format(out_text, ms_time))
        return res, raw_time_list

    @staticmethod
    def benchmark_phoronix(stmt, setup="pass", out_text="", repeat=1, context={}, output=None, verbose=True):
        """ This reflects the benchmarking logic used in benchit.py of the Phoronix benchmarks """
        timeit.template = Test.timeit_tmpl.format(init='{init}', setup='{setup}', stmt='{stmt}', output=output)

        ldict = {**context}
        number = 40
        output = timeit.repeat(stmt, setup=setup, repeat=repeat, number=number, globals=ldict)
        res = output[0][1]
        raw_time_list = [a for a, _ in output]
        if verbose:
            print("raw times:", " ".join(["%.*g" % (3, x) for x in raw_time_list]))

        time_scaled = [(x * 1e6 / number) for x in raw_time_list]
        raw_time = np.median(time_scaled)
        print(time_scaled)
        if verbose:
            print("{}: {}ns".format(out_text, raw_time))
        return res, time_scaled
