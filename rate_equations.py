#! /usr/bin/python3

import glob
import sys
import collections
import numpy as np
from contextlib import suppress
from itertools import product
from typing import List
import functools
import multiprocessing as mp

CPU = 10

to_parametrize = [("S1", "S0", 8), ("S1", "T1", 8), ("T1", "S0", 8)]


PARAMETRIZE = False
TEST_RATES = True


class Data:
    def __init__(self, value, from_state, to_state):
        self.value = value
        self.from_state = from_state
        self.to_state = to_state


class DataParam:
    resp_min = 1e10

    def __init__(
        self, rad_em, nrad_em, em_state, abs_state, data_list, output_fname, sens_exp
    ):
        self.rad_em = rad_em
        self.nrad_em = nrad_em
        self.em_state = em_state
        self.abs_state = abs_state
        self.data_list = data_list
        self.output_fname = output_fname
        self.sens_exp = sens_exp


MapStates = collections.namedtuple("MapStates", "index_ value")

DataList = List[Data]
MapStatesList = List[MapStates]


def objective_function(x, data):
    parametrize = [
        (d[0], d[1], float(f"1e{int(round(x[i], 0))}"))
        for i, d in enumerate(to_parametrize)
    ]

    sens_calc = quantum_yield(
        data.rad_em,
        data.nrad_em,
        data.em_state,
        data.abs_state,
        data.data_list,
        data.output_fname,
        parametrize,
    )

    resp = (abs(sens_calc - data.sens_exp) / data.sens_exp) * 100
    return resp, data.sens_exp, sens_calc


def mapped_states_sort_key(t: MapStates):
    return tuple(map(int, t.index_.split("_")))


def has_duplicated_states(map_states: MapStatesList):
    for i in range(len(map_states)):
        for j in range(i + 1, len(map_states) - 1):
            if map_states[i].index_ == map_states[j].index_:
                return True

    return False


def get_states_list(data: DataList, abs_state):
    states = []
    for state in data:
        states.append(state.from_state)
        states.append(state.to_state)

    states = list(set(states))
    try:
        s0_idx = states.index(abs_state)
    except:
        raise

    states[0], states[s0_idx] = states[s0_idx], states[0]

    return states


def exit_if_it_has_duplicated_states(mapped_states: MapStatesList):
    if has_duplicated_states(mapped_states):
        print(
            "There is repeated transition in the input_data.txt file.\n"
            "Please correct this issue before running the script."
        )
        sys.exit()


def get_map_states_list(states: list, data: DataList):
    mapped_states = [
        MapStates(f"{states.index(d.from_state)}_{states.index(d.to_state)}", d.value)
        for d in data
    ]

    exit_if_it_has_duplicated_states(mapped_states)
    mapped_states.sort(key=mapped_states_sort_key)

    index_list = [state.index_.split("_") for state in mapped_states]
    index_list = [int(a) * len(states) + int(b) for a, b in index_list]

    return index_list, mapped_states


def get_rates_matrix(states: list, index_list, mapped_states: MapStatesList):
    states_number = len(states)
    wet = np.zeros(shape=(states_number * states_number))

    for i in range(len(wet)):
        with suppress(ValueError):
            wet[i] = mapped_states[index_list.index(i)].value

    rates = np.empty(shape=(states_number, states_number))
    states_index = range(states_number)

    for k, (i, j) in enumerate(product(states_index, states_index)):
        rates[i, j] = float(wet[k])

    return rates


def solve_steady_state(rad_em, nrad_em, idx_em_state, states_number, rates):
    trans_rates = rates.transpose()
    for i, j in zip(range(states_number), range(states_number)):
        trans_rates[i, j] = -np.sum(rates[i, :])

    trans_rates[idx_em_state, idx_em_state] = (
        trans_rates[idx_em_state, idx_em_state] - rad_em - nrad_em
    )

    trans_rates[0, :states_number] = 1.0

    b = np.zeros(shape=states_number)
    b[0] = 1.0

    populations = np.linalg.solve(trans_rates, b)

    return populations


def handle_data_list(data_list):
    def parse_data_str(line):
        d = line.split()
        return float(d[2]), d[0], d[1]

    data_ = []
    for l in data_list:
        data_.append(Data(*parse_data_str(l)))

    return data_


def create_output_file(
    outfile_name, data_list, populations, rad_em, nrad_em, em_state, abs_state, q, ef
):
    with open(outfile_name, "w") as output:
        output.write(f"{'-'*30}\n")
        output.write(
            f"arad: {rad_em:>8.2f}\n"
            f"anrad: {nrad_em:>8.2f}\n"
            f"emitting state: {em_state}\n"
            f"absorbing state: {abs_state}\n\n"
        )

        for data in data_list:
            output.write(f"{data.from_state:<6} {data.to_state:<6} {data.value:.4e}\n")

        output.write(f"\n{'-' * 30}\n")
        for pop in populations:
            output.write(f"{pop[0]:<6} {pop[1]:.6f}\n")

        output.write(f"\n{'-' * 30}\n")
        output.write(
            f"q = {q:.2f}%\nef = {ef:.2f}%\nsens = {(q/ef)*100:.2f}%"
        )


def read_input_data(fname):
    with open(fname) as input:
        next(input)
        rad_em = float(next(input).split(":")[1])
        nrad_em = float(next(input).split(":")[1])
        em_state = next(input).split(":")[1].strip()
        abs_state = next(input).split(":")[1].strip()
        next(input)

        data_list = []
        for line in input:
            if not line.strip() or "------------" in line:
                break

            if "#" in line:
                continue

            data_list.append(line)

        return rad_em, nrad_em, em_state, abs_state, data_list


def quantum_yield(
    rad_em, nrad_em, em_state, abs_state, data_list, outfile_name, parametrize=[]
):
    data_list = handle_data_list(data_list)

    for p in parametrize:
        for d in data_list:
            if d.from_state == p[0] and d.to_state == p[1]:
                d.value = p[2]

    states = get_states_list(data_list, abs_state)

    index_in_list, map_states_list = get_map_states_list(states, data_list)

    rates = get_rates_matrix(states, index_in_list, map_states_list)

    try:
        idx_em_state = states.index(em_state)
    except:
        print(f"State {em_state} could not be found in {states}")
        raise

    populations = solve_steady_state(rad_em, nrad_em, idx_em_state, len(states), rates)

    absorbing_rate = [data.value for data in data_list if data.from_state == abs_state]
    idx_abs_state = states.index(abs_state)
    q = (
        100
        * (populations[idx_em_state] * rad_em)
        / (populations[idx_abs_state] * absorbing_rate[0])
    )
    ef = (rad_em / (rad_em + nrad_em)) * 100

    create_output_file(
        outfile_name,
        data_list,
        list(zip(states, populations)),
        rad_em,
        nrad_em,
        em_state,
        abs_state,
        q,
        ef,
    )
    return (q / ef) * 100


def run_objective_function(x_data):
    x, data = x_data
    resp, sens_exp, sens_calc = objective_function(x, data)
    return (x, sens_exp, sens_calc, resp)


def run_parallel(x_data):
    with mp.Pool(CPU) as pool:
        result = pool.map(run_objective_function, x_data)

    return result


if __name__ == "__main__":
    input_data = []
    for fname in glob.glob("*.rates"):
        input_data.append(
            {
                "fname": fname,
                "sens_exp": 100,
                "output_fname": fname.replace(".rates", ".yield"),
            }
        )

    for inp in input_data:
        fname = inp.get("fname")
        sens_exp = inp.get("sens_exp")
        output_fname = inp.get("output_fname")
        print(fname, sens_exp, output_fname)
        try:
            rad_em, nrad_em, em_state, abs_state, data_list = read_input_data(fname)
        except:
            print("It couldn't read the file containing the rates")
            sys.exit()

        quantum_yield(rad_em, nrad_em, em_state, abs_state, data_list, output_fname)

        print("----------------------------------")

        data = DataParam(
            rad_em, nrad_em, em_state, abs_state, data_list, output_fname, sens_exp
        )
       
        values = [range(11) for _ in range(len(to_parametrize))]
        x_data = []
        for value in product(*values):
            x_data.append(([v for v in value], data))

        buffer = run_parallel(x_data)

        buffer.sort(key=lambda l: l[-1])
        with open(fname.replace(".rates", ".simul"), "w") as output:
            trans = [
                f"{to_parametrize[i][0]}_{to_parametrize[i][1]} "
                for i in range(len(to_parametrize))
            ]
            for t in trans:
                output.write(t)
            output.write("\n")

            for b in buffer:
                for v in b[0]:
                    output.write(f"1e{v}   ")
                output.write(f"{b[-3]:>7.2f}  {b[-2]:.2f}  {b[-1]:.4f}\n")

        x = [int(f"{v}".replace("1e", "")) for v in buffer[0][0]]
        
        print(x)
        objective_function(x, data)
