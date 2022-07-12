from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
import numpy as np
import matplotlib.pyplot as plt
from qiskit.providers.aer.noise import NoiseModel
import qiskit.quantum_info as qi


def main():
    print(PRINT_MESSAGE)

    aer_sim = Aer.get_backend('aer_simulator')

    quantum_circuit = generate_circuit()
    draw_circuit(quantum_circuit)
    counts = run_simulation(aer_sim, quantum_circuit)
    report_results(counts)


def report_results(counts):
    fig = plot_histogram(counts)
    ax = fig.axes[0]
    ax.set_xticklabels(ax.get_xticklabels(), fontsize='7')
    fig.savefig('measurements.png')
    print(counts)


def run_simulation(aer_sim, quantum_circuit):
    counts = execute(quantum_circuit, backend=aer_sim, noise_model=get_noise(0.1), shots=10000).result().get_counts()
    return counts


def draw_circuit(quantum_circuit):
    quantum_circuit.draw('mpl', scale=2, style={'backgroundcolor': '#EEEEEE'})
    plt.savefig("circuit.png")


def generate_circuit():
    all_qubits = [0, 1, 2, 3, 4, 5, 6]
    quantum_error_1 = QuantumCircuit(7, name='error')
    quantum_error_2 = QuantumCircuit(7, name='error2')
    ident3 = qi.Operator(np.identity(2 ** 7))
    quantum_error_1.unitary(ident3, all_qubits, label='error')
    quantum_error_2.unitary(ident3, all_qubits, label='error2')
    quantum_circuit = QuantumCircuit(7)
    initialize_first_qubit_to_state(quantum_circuit)
    quantum_circuit.append(Encoding7(), all_qubits)
    quantum_circuit.barrier()
    ####
    # Attaching unitary identity gates (faulty gates)
    ####
    quantum_circuit.append(quantum_error_1, all_qubits)
    quantum_circuit.append(quantum_error_2, all_qubits)
    quantum_circuit.barrier()
    ancillas_x = QuantumRegister(3, 'ancillas_x')
    ancillas_z = QuantumRegister(3, 'ancillas_z')
    quantum_circuit.add_register(ancillas_x)
    quantum_circuit.add_register(ancillas_z)
    classical_register_x = ClassicalRegister(3, 'synd_X')
    classical_register_z = ClassicalRegister(3, 'synd_Z')
    quantum_circuit.add_register(classical_register_x)
    quantum_circuit.add_register(classical_register_z)
    for i in range(7, 10):
        quantum_circuit.h(i)
    append_stabilizers(quantum_circuit)
    for i in range(7, 10):
        quantum_circuit.h(i)
    quantum_circuit.barrier()
    measure_ancillas(ancillas_x, ancillas_z, classical_register_x, classical_register_z, quantum_circuit)
    # Correction
    quantum_circuit.barrier()
    correct_errors(classical_register_x, classical_register_z, quantum_circuit)
    decode_logical_qubit(all_qubits, quantum_circuit)
    original_qubit_outcome = ClassicalRegister(1, 'outcome')
    quantum_circuit.add_register(original_qubit_outcome)
    quantum_circuit.measure(all_qubits[0], original_qubit_outcome)
    return quantum_circuit


def decode_logical_qubit(all_qubits, quantum_circuit):
    quantum_circuit.append(Encoding7().inverse(), all_qubits)


def correct_errors(classical_register_x, classical_register_z, quantum_circuit):
    for i in range(0, 7):
        quantum_circuit.z(i).c_if(classical_register_z, i + 1)
        quantum_circuit.x(i).c_if(classical_register_x, i + 1)


def measure_ancillas(ancillas_x, ancillas_z, classical_register_x, classical_register_z, quantum_circuit):
    quantum_circuit.measure(ancillas_x[0], classical_register_x[0])
    quantum_circuit.measure(ancillas_x[1], classical_register_x[1])
    quantum_circuit.measure(ancillas_x[2], classical_register_x[2])
    quantum_circuit.measure(ancillas_z[0], classical_register_z[0])
    quantum_circuit.measure(ancillas_z[1], classical_register_z[1])
    quantum_circuit.measure(ancillas_z[2], classical_register_z[2])


def append_stabilizers(quantum_circuit):
    stab1Z = QuantumCircuit(5, name='M1')  # IIIZZZZ
    for i in range(0, stab1Z.num_qubits - 1):
        stab1Z.cx(i, 4)
    stab2Z = QuantumCircuit(5, name='M2')  # IZZIIZZ
    for i in range(0, stab2Z.num_qubits - 1):
        stab2Z.cx(i, 4)
    stab3Z = QuantumCircuit(5, name='M3')  # ZIZIZIZ
    for i in range(0, stab3Z.num_qubits - 1):
        stab3Z.cx(i, 4)
    stab1X = QuantumCircuit(5, name='M4')  # IIIXXXX
    for i in range(0, stab1X.num_qubits - 1):
        stab1X.cx(4, i)
    stab2X = QuantumCircuit(5, name='M5')  # IXXIIXX
    for i in range(0, stab2X.num_qubits - 1):
        stab2X.cx(4, i)
    stab3X = QuantumCircuit(5, name='M6')  # XIXIXIX
    for i in range(0, stab3X.num_qubits - 1):
        stab3X.cx(4, i)
    quantum_circuit.append(stab1Z, [3, 4, 5, 6, 9])
    quantum_circuit.append(stab2Z, [1, 2, 5, 6, 8])
    quantum_circuit.append(stab3Z, [0, 2, 4, 6, 7])  ##Stab Z goes in ancillas_x and then classical_register_x
    quantum_circuit.append(stab3X, [0, 2, 4, 6, 10])
    quantum_circuit.append(stab2X, [1, 2, 5, 6, 11])
    quantum_circuit.append(stab1X, [3, 4, 5, 6, 12])  ##Stab X goes in ancillas_z and then classical_register_z


def initialize_first_qubit_to_state(qc_3qx):
    initial_state = [1 / np.sqrt(2), 1 / np.sqrt(2)]
    # initial_state = [0,1]
    # qc_3qx.initialize(initial_state, 0)
    # # Initialize the 0th qubit in the state `initial_state`


###
# Encoding
###
def Encoding7():
    q_encoding = QuantumCircuit(7, name='Enc')
    # Hadamards
    q_encoding.h(6)
    q_encoding.h(5)
    q_encoding.h(4)
    q_encoding.cx(0, 1)
    q_encoding.cx(0, 2)
    q_encoding.cx(6, 3)
    q_encoding.cx(6, 1)
    q_encoding.cx(6, 0)
    q_encoding.cx(5, 3)
    q_encoding.cx(5, 2)
    q_encoding.cx(5, 0)
    q_encoding.cx(4, 3)
    q_encoding.cx(4, 2)
    q_encoding.cx(4, 1)
    # q_encoding.draw('mpl',  filename='Encoding_Seven')
    # plt.show()
    return q_encoding


#####
# Measurement function (Apply it when you want to look at the results)
#####
# def measurement7(circ, nqubit, noise_model=NoiseModel(['unitary']), shots=100000):
#     cr3 = ClassicalRegister(nqubit, 'outcomes')
#     circ.add_register(cr3)
#     circ.measure(all_qubits, cr3)
#     counts = execute(qc_3qx, backend=aer_sim, noise_model=noise_model, shots=shots).result().get_counts()
#     return counts


# #####
# #Create the error model
# #####
def get_noise(p_error):
    # Bit flip error
    bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
    bit_flip1 = bit_flip.tensor(bit_flip)
    bit_flip2 = bit_flip.tensor(bit_flip1)
    bit_flip3 = bit_flip.tensor(bit_flip2)
    bit_flip4 = bit_flip.tensor(bit_flip3)
    bit_flip5 = bit_flip.tensor(bit_flip4)
    bit_flip6 = bit_flip.tensor(bit_flip5)

    # Phase flip error
    phase_flip = pauli_error([('Z', p_error), ('I', 1 - p_error)])
    phase_flip1 = phase_flip.tensor(phase_flip)
    phase_flip2 = phase_flip.tensor(phase_flip1)
    phase_flip3 = phase_flip.tensor(phase_flip2)
    phase_flip4 = phase_flip.tensor(phase_flip3)
    phase_flip5 = phase_flip.tensor(phase_flip4)
    phase_flip6 = phase_flip.tensor(phase_flip5)

    # ######
    # #Set error basis
    # ######
    noise_model = NoiseModel(['unitary'])
    noise_model.add_all_qubit_quantum_error(bit_flip6, 'error')
    noise_model.add_all_qubit_quantum_error(phase_flip6, 'error2')
    return noise_model


PRINT_MESSAGE = """ notes:
       _               _          _ _  __  __    __             _                                 _ _ _ 
      | |             | |        | (_)/ _|/ _|  / _|           (_)                               | | | |
   ___| |__   ___  ___| | __   __| |_| |_| |_  | |_ ___  _ __   _ _ __ ___   __ _  __ _  ___  ___| | | |
  / __| '_ \ / _ \/ __| |/ /  / _` | |  _|  _| |  _/ _ \| '__| | | '_ ` _ \ / _` |/ _` |/ _ \/ __| | | |
 | (__| | | |  __/ (__|   <  | (_| | | | | |   | || (_) | |    | | | | | | | (_| | (_| |  __/\__ \_|_|_|
  \___|_| |_|\___|\___|_|\_\  \__,_|_|_| |_|   |_| \___/|_|    |_|_| |_| |_|\__,_|\__, |\___||___(_|_|_)
                                                                                   __/ |                
                                                                                  |___/                 
- remember to use barrier between gates to avoid optimization combining them to one gate

"""

if __name__ == "__main__":
    main()
