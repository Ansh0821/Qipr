from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
# import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import style

style.use("default")

image = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


# Function for plotting the image using matplotlib
def plot_image(img, title: str):
    plt.title(title)
    plt.xticks(range(img.shape[0]))
    plt.yticks(range(img.shape[1]))
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap="viridis")
    plt.show()


plot_image(image, "Original Image")


# Convert the raw pixel values to probability amplitudes
def amplitude_encode(img_data):

    # Calculate the RMS value
    rms = np.sqrt(np.sum(np.sum(img_data**2, axis=1)))

    # Create normalized image
    image_norm = []
    for arr in img_data:
        for ele in arr:
            if rms == 0:
                image_norm.append(0)
            else:
                image_norm.append(ele / rms)

    # Return the normalized image as a numpy array
    return np.array(image_norm)


# Get the amplitude ancoded pixel values
# Horizontal: Original image
image_norm_h = amplitude_encode(image)

# Vertical: Transpose of Original image
image_norm_v = amplitude_encode(image.T)

# Initialize some global variable for number of qubits
data_qb = 6
anc_qb = 1
total_qb = data_qb + anc_qb

# Initialize the amplitude permutation unitary
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)

# Replace the 'display' function calls with Matplotlib's 'plt.show()'
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
qc_h.draw(output='mpl', fold=-1)
plt.title('Horizontal Circuit')

plt.subplot(1, 2, 2)
qc_v.draw(output='mpl', fold=-1)
plt.title('Vertical Circuit')

plt.tight_layout()
plt.show()


# Create the circuit for horizontal scan
# qc_h = QuantumCircuit(total_qb)
# qc_h.initialize(image_norm_h, range(1, total_qb))
# qc_h.h(0)
# qc_h.unitary(D2n_1, range(total_qb))
# qc_h.h(0)
# qc_h.draw("mpl", fold=-1)

# # Create the circuit for vertical scan
# qc_v = QuantumCircuit(total_qb)
# qc_v.initialize(image_norm_v, range(1, total_qb))
# qc_v.h(0)
# qc_v.unitary(D2n_1, range(total_qb))
# qc_v.h(0)
# qc_v.draw("mpl", fold=-1)

# Combine both circuits into a single list
circ_list = [qc_h, qc_v]


# Simulating the cirucits
# back = Aer.get_backend('statevector_simulator')
# results = execute(circ_list, backend=back).result()
# sv_h = results.get_statevector(qc_h)
# sv_v = results.get_statevector(qc_v)

# from qiskit.visualization import array_to_latex
# print('Horizontal scan statevector:')
# #print(np.array(sv_h))
# display(array_to_latex(np.array(sv_h)[:30], max_size=30))
# print()
# print('Vertical scan statevector:')
# display(array_to_latex(np.array(sv_v)[:30], max_size=30))


# Create empty circuit
# example_circuit = QuantumCircuit(2)
# example_circuit.measure_all()

# QiskitRuntimeService(channel="ibm_quantum", token="6c6be2949a6b0cf6d5d1117183bcbebbdbab15656d657d6f58f10aacb396cb2cae95250ecdc7870fac522e1ac21b1971241251d65caed436fac1bcb93a615f37")
# service = QiskitRuntimeService()
# backend = service.backend("ibmq_qasm_simulator")

# sampler = Sampler(backend)
# job = sampler.run([(circ_list,)])
# print(f"job id: {job.job_id()}")
# result = job.result()
# print(result)
