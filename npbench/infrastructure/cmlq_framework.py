from collections import defaultdict

import numpy as np

from npbench.infrastructure import Framework, build_sql_insert


class CMLQFramework(Framework):
    def __init__(self, fwname):
        super(CMLQFramework, self).__init__(fwname)

    def save_statistics(self, conn, bench, experiment_id):
        cache_stats = np.core.multiarray.get_cmlq_stats()
        instr_ptr_to_id = defaultdict(lambda: None)
        for stat in cache_stats:
            instr_ptr = stat["instr_ptr"]
            del stat["instr_ptr"]
            stat['experiment_id'] = experiment_id
            stat['benchmark'] = bench.info['short_name']
            sql = build_sql_insert("cache", stat)
            cur = conn.cursor()
            cur.execute(sql, tuple(stat.values()))
            instr_ptr_to_id[instr_ptr] = cur.lastrowid
            conn.commit()

        implementations = self.implementations(bench)
        assert len(implementations) == 1

        module_functions = map(lambda f: f.__code__, self.module_functions(bench))

        for function in get_cmlq_functions():
            bench_function = function in module_functions
            fun_stats = get_cmlq_stats(function).values()
            for stat in fun_stats:
                cache_id = instr_ptr_to_id[stat['instr_ptr']]
                del stat['instr_ptr']
                stat['cache_id'] = cache_id
                stat['function'] = function.co_name
                stat['benchmark'] = bench.info['short_name']
                stat['experiment_id'] = experiment_id

                # TODO: this part is not fully reliable, we had to manually fix a few entries
                #   where the framework thought specialization did not happen in the benchmark although it did
                stat['in_bench'] = bench_function
                stat['filename'] = function.co_filename
                sql = build_sql_insert("statistics", stat)
                cur = conn.cursor()
                cur.execute(sql, tuple(stat.values()))
                conn.commit()
