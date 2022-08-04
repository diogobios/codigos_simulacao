#! /usr/bin/python3

import os
import math
import collections
import numpy as np
import operator

# LN_CHOSEN_STATES = ["5D0", "5D1", "5D2", "5D3", "5L6", "5L7", "5G2", "5G3", "5G5", "5G6", "5D4"]
LN_CHOSEN_STATES = ["5D4", "5D1", "5D0"]

is_create_rates_file = True

ExcitationTransition = collections.namedtuple(
    "ExcitationTransition", "initial final energy U_ele S_ele"
)

O_ED = {2: 0.0100e-20, 4: 0.0100e-20, 6: 0.0100e-20}

PI = 3.14159265359
RACAH_Ln = {2: -1.3660, 4: 1.1280, 6: -1.2774}
BOHR_RADIUS = 0.52917721092e-8
RAD_Eu = {
    2: 0.9175 * BOHR_RADIUS ** 2,
    4: 2.0200 * BOHR_RADIUS ** 4,
    6: 9.0390 * BOHR_RADIUS ** 6,
    8: 110.03237 * BOHR_RADIUS ** 8,
}

SIGMA_Eu = {0: 0.981, 1: 0.800, 2: 0.600, 4: 0.139, 6: 0.100}
HBAR = 1.054571726e-27
LIGHT_VEL = 2.997924580e10
ECHARGE = 4.8032045e-10
BOLTZMANN_CONST = 1.3806488e-16
POPULATIONS_Eu = {"7F0": 0.64, "7F1": 0.33}

TRANSITIONS_Eu = [
    ExcitationTransition("7F0", "5D0", 17293, {2: 0.0032, 4: 0.0, 6: 0.0}, 0),
    ExcitationTransition("7F0", "5D1", 19027, {2: 0.0, 4: 0.0, 6: 0.0}, 2.73e-2),
    ExcitationTransition("7F0", "5L6", 25325, {2: 0.0, 4: 0.0, 6: 0.0153}, 0),
    ExcitationTransition("7F0", "5G6", 26752, {2: 0.0, 4: 0.0, 6: 0.0037}, 0),
    ExcitationTransition("7F0", "5D4", 27586, {2: 0.0, 4: 0.0011, 6: 0.0}, 0),
    ExcitationTransition("7F1", "5D0", 16921, {2: 0.0, 4: 0.0, 6: 0.0}, 0.117),
    ExcitationTransition("7F1", "5D1", 18655, {2: 0.0025, 4: 0.0, 6: 0.0}, 2.81e-5),
    ExcitationTransition("7F1", "5D2", 21111, {2: 0.0, 4: 0.0, 6: 0.0}, 4.58e-3),
    ExcitationTransition("7F1", "5D3", 23983, {2: 0.0004, 4: 0.0012, 6: 0.0}, 0),
    ExcitationTransition("7F1", "5L6", 24953, {2: 0.0, 4: 0.0, 6: 0.0091}, 0),
    ExcitationTransition("7F1", "5L7", 25985, {2: 0.0, 4: 0.0, 6: 0.0181}, 0),
    ExcitationTransition("7F1", "5G2", 26020, {2: 0.0, 4: 0.0, 6: 0.0}, 1.28e-2),
    ExcitationTransition("7F1", "5G3", 26250, {2: 0.0002, 4: 0.0012, 6: 0.0}, 0),
    ExcitationTransition("7F1", "5G6", 26380, {2: 0.0, 4: 0.0, 6: 0.004}, 0),
    ExcitationTransition("7F1", "5G5", 26391, {2: 0.0, 4: 0.0004, 6: 0.0097}, 0),
]

angular_moment = ["S", "P", "D", "F", "G", "H", "I", "K", "L", "M", "N", "O", "Q"]

# Tab.1_ChemPhysLetters, 757(2020), 137884
PARAMS_rol = {
    "Eu-N": [3.102, -3.328, -0.3310],
    "Eu-O": [0.340, -1.107, -0.074],
    "Eu-F": [7.123, -10.4428, 1.678],
    "Eu-P": [-0.171, 0.073, -0.369],
    "Eu-S": [1.330, -1.789, 0.105],
    "Eu-Cl": [1.579, -1.996, 0.128],
}


def get_quantum_number(source):
    S = (float(source[0]) - 1) / 2
    L = angular_moment.index(source[1])
    J = float(source[2:])
    return S, L, J


def get_unit_tensor_op(initial_state, final_state):
    for transition in TRANSITIONS_Eu:
        if transition.initial == initial_state and transition.final == final_state:
            return transition.U_ele

    return 0.0


def get_spin_ele(initial_state, final_state):
    for transition in TRANSITIONS_Eu:
        if transition.initial == initial_state and transition.final == final_state:
            return transition.S_ele

    return 0.0


def get_energy_ln(initial_state, final_state):
    for transition in TRANSITIONS_Eu:
        if transition.initial == initial_state and transition.final == final_state:
            return transition.energy

    return 0.0


class EnergyTransfer:
    def __init__(
            self,
            multiplicity,
    ):
        self.G = 1 if "sing" in multiplicity else 3
        self.state_ligand = "S" if "sing" in multiplicity else "T"
        self.mult = multiplicity
        if self.state_ligand == "S":
            self.ligand_dipole_strength = 1e-35
        else:
            self.ligand_dipole_strength = 1e-40

        self.dipole_spin_coupled = 1e-36

        self.n = 3.5

    # Eq.13_ChemPhysLetters, 757(2020), 137884
    def calc_rol0_AR(self, r, ln, atom):
        a, b, c = PARAMS_rol[f"{ln}-{atom}"]
        return math.exp(a + b * r + c * r * r)

    # Spectral overlap factor
    def FC(self, delta):
        ln2 = 0.693147181

        # Ligand bandwidth at half-height
        hhw = 3200  # cm^-1
        gamma = 2 * PI * LIGHT_VEL * hhw  # cm^-1 -> erg

        return (
                1
                / (HBAR * gamma)
                * ((ln2 / PI) ** 0.5)
                * math.exp(-((delta / hhw) ** 2) * ln2)
        )

    # Energy Transfer Rate Calculated by the Multipolar Mechanism
    def calc_wet_ic(self, energy_ln, J, U, population=1.0):
        # Eq.17_Coord Chem Rev, 196(2000), 165
        def gamma(lamb):
            return (
                    (lamb + 1)
                    * (RAD_Eu.get(lamb) ** 2)
                    * ((1 - SIGMA_Eu.get(lamb)) ** 2)
                    * (RACAH_Ln.get(lamb) ** 2 / ((self.rl ** (lamb + 2)) ** 2))
            )

        delta = self.energy_ligand - energy_ln

        # Eq.14_Coord Chem Rev, 196(2000),165
        hamiltonian_dm = (
                ((ECHARGE ** 2) * self.ligand_dipole_strength)
                / ((2 * J + 1) * self.G)
                * (gamma(2) * U.get(2) + gamma(4) * U.get(4) + gamma(6) * U.get(6))
        )

        # Eq.30_J Non-Cryst Solids, 354(2008), 4770
        sigma = SIGMA_Eu.get(1)

        # Eq.15_Coord Chem Rev, 196(2000), 165
        hamiltonian_dd = (
                (2 * (ECHARGE ** 2) * self.ligand_dipole_strength * ((1 - sigma) ** 2))
                / ((2 * J + 1) * self.G * (self.rl ** 6))
                * (O_ED.get(2) * U.get(2) + O_ED.get(4) * U.get(4) + O_ED.get(6) * U.get(6))
        )

        wet_ic = (
                ((2 * PI) / HBAR)
                * self.FC(delta)
                * (hamiltonian_dm + hamiltonian_dd)
                * population
        )
        back_wet_ic = self.calc_back_energy_transfer(wet_ic, delta)
        return wet_ic, back_wet_ic

    def rol(self):
        # Eq.33_J Non-Cryst Solids, 354(2008), 4770

        # self.rol0 = 0.05
        self.rol0 = self.calc_rol0_AR(self.r_min * 1e8, "Eu", "O")

        return self.rol0 * ((self.r_min / self.rl) ** self.n)

    def hamiltonian_ex(self, J, spin_ele):
        # Eq.40_Chap310-Handbook on the Phys and Chem of Rare Earths. Elsevier (2019), 55
        return (1.3333333333 * self.dipole_spin_coupled * spin_ele) * (
                ((ECHARGE ** 2) * (self.rol() ** 2)) / ((2 * J + 1) * (self.rl ** 4))
        )

    # Energy Transfer Rate Calculated by the Exchange Mechanism
    def calc_wet_ex(self, energy_eu, J, spin_ele, population=1.0):
        delta = self.energy_ligand - energy_eu

        wet_ex = (
                ((2 * PI) / HBAR)
                * self.hamiltonian_ex(J, spin_ele)
                * self.FC(delta)
                * population
        )
        back_wet = self.calc_back_energy_transfer(wet_ex, delta)
        return wet_ex, back_wet

    def calc_back_energy_transfer(self, wet, delta):
        temperature = 298.2  # 300
        return wet * math.exp(
            -(HBAR * 2 * PI * LIGHT_VEL * delta) / (BOLTZMANN_CONST * temperature)
        )


def create_rates_file(wet_data, output):
    wet_data_1 = []
    for i in range(len(wet_data) - 1):
        for j in range(i + 1, len(wet_data)):
            ln_i = wet_data[i][0].split("->")[1]
            ln_j = wet_data[j][0].split("->")[1]
            ligand = wet_data[i][0].split()[0]

            if ln_i == ln_j:
                wet_data_1.append(
                    (
                        f"{ligand} {ln_i}",
                        wet_data[i][1] + wet_data[j][1],
                        wet_data[i][2] + wet_data[j][2],
                    )
                )

    wet_data_2 = []
    for d in wet_data:
        state = f'{d[0].split()[0]} {d[0].split("->")[1]}'
        is_equal = False
        for d_1 in wet_data_1:
            if d_1[0] == state:
                is_equal = True
                break
        if not is_equal:
            wet_data_2.append((state, d[1], d[2]))

    wet_data = wet_data_1 + wet_data_2

    for d in wet_data:
        output.write(
            f"{d[0].split()[0]}      {d[0].split()[1]}    {d[1]:.4e} \n{d[0].split()[1]}    {d[0].split()[0]}      {d[2]:.4e} \n"
        )


def calc_energy_transfer(energy_transfer, output):
    wet_data=[]
    for transition in TRANSITIONS_Eu:
        initial_state = transition.initial
        final_state = transition.final
        energy_ln = get_energy_ln(initial_state, final_state)
        spin_ele = get_spin_ele(initial_state, final_state)

        U = get_unit_tensor_op(initial_state, final_state)

        channel = f"{energy_transfer.state_ligand} {initial_state}->{final_state}"
        
        if not final_state in LN_CHOSEN_STATES:
            continue
        
        if initial_state == "7F0":
            continue

        S, L, J = get_quantum_number(final_state)

        try:
            population = POPULATIONS_Eu.get(initial_state)
        except:
            population = 1.0

        wet_ic, b_wet_ic = energy_transfer.calc_wet_ic(energy_ln, J, U, population)
        wet_ex, b_wet_ex = energy_transfer.calc_wet_ex(
            energy_ln, J, spin_ele, population
        )

        if "7F0" in transition.initial and "5D0" in transition.final:
            j_mixing = 0.05 ** 2
            wet_ic = wet_ic * j_mixing
            b_wet_ic = b_wet_ic * j_mixing

        wet_data.append(
            (
                channel,
                wet_ex + wet_ic,
                b_wet_ic + b_wet_ex,
            )
        )
        
        if not is_create_rates_file:
            output.write(
                f"{energy_transfer.rl*1e8:.2f} {energy_transfer.energy_ligand} "
                f"{channel.replace(' ', '_')} {wet_ex:>10.6e}  {wet_ic:>10.6e} {b_wet_ic + b_wet_ex:>10.6e}\n"
            )

    if is_create_rates_file:
        create_rates_file(wet_data, output)


if __name__ == "__main__":
    state = input("energy state of the ligand (type sing or trip): ")
    energy_transfer = EnergyTransfer(state)
    rl_or_energ = input("variation (type rl or energy) or none:")
    energy_transfer.r_min = 2.330e-8
 
    min_, max_ = 16500.00, 24100.00
    if "sing" in energy_transfer.mult:
        min_, max_ = 22500.00, 34600.00

    text_top = (
        "------------------------------\n"
        "arad:  1000\n"
        "anrad: 1000\n"
        "emitting state: 5D0\n"
        "absorbing state: S0\n\n"
        "S0     S1     1e4\n"
        "S1     S0     1e6\n"
        "S1     T      1e6\n"
        "T      S0     1e6\n"
    )

    ln_states = []
    transitions = []
    for t_ in TRANSITIONS_Eu:
        if not t_.final in LN_CHOSEN_STATES:
            continue
        transitions.append(t_)
        ln_states.append((t_.final, t_.energy))

    ln_states.sort(reverse=True, key=operator.itemgetter(1))
    ln_states = [l[0] for l in ln_states]
    ln_states = list(dict.fromkeys(ln_states))

    for j in range(len(ln_states) - 1):
        text_top += f"{ln_states[j]:<7}{ln_states[j + 1]:<6} 1e6\n"
    
    if rl_or_energ == "none":
        with open(f"output_5D4-5D1-5D0.txt", "w") as output:
            output.write(text_top)
            energy_transfer.rl = 4.85 * 1e-8
            energy_transfer.energy_ligand = 22100
            calc_energy_transfer(energy_transfer, output)  
            output.write("------------------------------")
    
    if rl_or_energ == "rl":    
        with open("output.txt", "w") as output:
            output.write("")
        
        for rl in np.arange(2.5, 5.5, 0.5):
            print(" ")
            energy_transfer.rl = rl * 1e-8  # cm
            energy_transfer.energy_ligand = 30000.00 if "sing" in energy_transfer.mult else 20000  # cm^-1
            calc_energy_transfer(energy_transfer, output)
    elif rl_or_energ == "energy":
        with open("output.txt", "w") as output:
            output.write("")
        for energy in np.arange(min_, max_, 200.00):
            print(" ")
            energy_transfer.rl = 3.5 * 1e-8  # cm
            energy_transfer.energy_ligand = energy
            calc_energy_transfer(energy_transfer, output)
    else:
        if is_create_rates_file:
            for rl in np.arange(2.5, 5.5, 0.5):
                for energy in np.arange(min_, max_, 200.00):                
                    with open(f"out_{state}_{rl:.2f}_{energy}.rates", "w") as output:
                        output.write(text_top)
                        energy_transfer.rl = rl * 1e-8
                        energy_transfer.energy_ligand = energy
                        calc_energy_transfer(energy_transfer, output)
                        output.write("------------------------------")
        else:
            with open(f"output.txt", "w") as output:
                for rl in np.arange(2.5, 5.5, 0.1):
                    for energy in np.arange(min_, max_, 100.00):
                        energy_transfer.rl = rl * 1e-8
                        energy_transfer.energy_ligand = energy
                        calc_energy_transfer(energy_transfer, output)

    print("\nrol0 = ", energy_transfer.rol0)

    os.system("pause")
