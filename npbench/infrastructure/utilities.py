# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import argparse
import sqlite3
import timeit
from numbers import Number
from typing import Union

import numpy as np


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def time_to_ms(raw: float) -> int:
    return int(round(raw * 1000))


def relative_error(ref: Union[Number, np.ndarray], val: Union[Number, np.ndarray]) -> float:
    return np.linalg.norm(ref - val) / np.linalg.norm(ref)


# Taken from shttps://www.sqlitetutorial.net/sqlite-python/create-tables/
def create_connection(db_file) -> sqlite3.Connection:
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return conn


# Taken from https://www.sqlitetutorial.net/sqlite-python/create-tables/
def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)


def create_result(conn, query, result):
    """
    Create a new result into the results table
    :param conn:
    :param project:
    :return: project id
    """
    cur = conn.cursor()
    cur.execute(query, result)
    conn.commit()
    return cur.lastrowid


sql_create_results_table = """
CREATE TABLE IF NOT EXISTS results (
    id integer PRIMARY KEY,
    timestamp integer NOT NULL,
    benchmark text NOT NULL,
    kind text,
    domain text,
    dwarf text,
    preset text NOT NULL,
    mode text NOT NULL,
    framework text NOT NULL,
    version text NOT NULL,
    details text,
    validated integer,
    time real,
    experiment_id text,
    host text
);
"""

sql_insert_into_results_table = """
INSERT INTO results(
    timestamp, benchmark, kind, domain, dwarf, preset, mode,
    framework, version, details, validated, time, experiment_id
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

sql_create_lcounts_table = """
CREATE TABLE IF NOT EXISTS lcounts (
    id integer PRIMARY KEY,
    timestamp integer NOT NULL,
    benchmark text NOT NULL,
    kind text,
    domain text,
    dwarf text,
    mode text NOT NULL,
    framework text NOT NULL,
    version text NOT NULL,
    details text,
    count integer,
    npdiff integer
);
"""

sql_insert_into_lcounts_table = """
INSERT INTO lcounts(
    timestamp, benchmark, kind, domain, dwarf, mode,
    framework, version, details, count, npdiff
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

sql_create_statistics_table = """
CREATE TABLE IF NOT EXISTS statistics (
    id integer PRIMARY KEY,
    experiment_id integer,
    opcode integer,
    opcode_name TEXT,
    base_opcode integer,
    base_opcode_name TEXT,
    benchmark text NOT NULL,
    function text NOT NULL,
    offset INT NOT NULL,
    exec_count INT NOT NULL,
    exec_ms INT NOT NULL,
    specialization_attempts INT NOT NULL,
    cache_id INT,
    in_bench BOOLEAN NOT NULL,
    filename TEXT NOT NULL,
    FOREIGN KEY(cache_id) REFERENCES cache(id))
"""

sql_create_cache_table = """
CREATE TABLE IF NOT EXISTS cache (
    id integer PRIMARY KEY,
    experiment_id integer,
    opname TEXT NOT NULL,
    exponent_type_misses INT NOT NULL,
    iterator_cache_hits INT NOT NULL,
    iterator_cache_misses INT NOT NULL,
    iterator_case INT NOT NULL,
    left_type_misses INT NOT NULL,
    right_type_misses INT NOT NULL,
    ndims_misses INT NOT NULL,
    op_exec_count INT NOT NULL,
    refcnt_misses INT NOT NULL,
    result_cache_hits INT NOT NULL,
    result_cache_misses INT NOT NULL,
    shape_misses INT NOT NULL,
    state TEXT NOT NULL,
    temp_elision_hits INT NOT NULL,
    trivial_cache_hits INT NOT NULL,
    trivial_cache_misses INT NOT NULL,
    trivial_case INT NOT NULL,
    ufunc_type_misses INT NOT NULL,
    last_state TEXT NOT NULL,
    function_end_clear INT NOT NULL,
    trivial_cache_init INT NOT NULL,
    iterator_cache_init INT NOT NULL,
    benchmark text NOT NULL
    )
"""




def validate(ref, val, framework="Unknown", rtol=1e-5, atol=1e-8, norm_error=1e-5):
    if not isinstance(ref, (tuple, list)):
        ref = [ref]
    if not isinstance(val, (tuple, list)):
        val = [val]
    valid = True
    for r, v in zip(ref, val):
        if not np.allclose(r, v, rtol=rtol, atol=atol):
            try:
                import cupy
                if isinstance(v, cupy.ndarray):
                    relerror = relative_error(r, cupy.asnumpy(v))
                else:
                    relerror = relative_error(r, v)
            except Exception:
                relerror = relative_error(r, v)
            if relerror < norm_error:
                continue
            valid = False
            print("Relative error: {}".format(relerror))
            # return False
    if not valid:
        print("{} did not validate!".format(framework))
    return valid


def build_sql_insert(table, key_vals):
    columns = ', '.join(key_vals.keys())
    placeholders = ', '.join('?' * len(key_vals))
    sql = f'INSERT INTO {table} ({columns}) VALUES ({placeholders})'
    return sql
