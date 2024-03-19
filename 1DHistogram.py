from qiskit import Aer, QuantumCircuit, execute
from PIL import Image
import numpy as np
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram

# Function to normalize and prepare the image
def prepare_image(image_path):
    # Load the image and convert it to grayscale
    image = Image.open(image_path).convert('L')
    # Resize the image to a desired size (e.g., 32x32 pixels)
    image = image.resize((32, 32))
    # Convert the image to a numpy array and normalize pixel values
    normalized_image = np.array(image, dtype=np.float32) / 255.0
    return normalized_image

# Quantum circuit for FRQI
def frqi_circuit(parameters, qubit):
    num_qubits = len(parameters)
    circuit = QuantumCircuit(num_qubits, num_qubits)
    for i, param in enumerate(parameters):
        circuit.ry(param, i)
    for i in range(num_qubits - 1):
        circuit.cx(i, i + 1)
    return circuit

# PSO optimization
def pso_optimization(image_data):
    num_qubits = len(image_data)
    def objective_function(params):
        circuit = frqi_circuit(params, num_qubits)
        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=1000)
        result = job.result()
        counts = result.get_counts(circuit)
        entropy = calculate_entropy(counts)
        return -entropy
    initial_guess = np.random.uniform(0, 2 * np.pi, size=num_qubits)
    result = minimize(objective_function, initial_guess, method='PSO')
    return result.x

# Calculate Shannon entropy from counts
def calculate_entropy(counts):
    total_counts = sum(counts.values())
    probabilities = np.array(list(counts.values())) / total_counts
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Calculate Quantum Renyi Entropy from counts
def calculate_qre(counts, alpha):
    total_counts = sum(counts.values())
    probabilities = np.array(list(counts.values())) / total_counts
    qre = np.log(np.sum(probabilities**alpha)) / (1 - alpha)
    return qre

if __name__ == "__main__":
    # Load and prepare the image
    image_path = "your_image.jpg"
    image_data = prepare_image(image_path)
    
    # PSO optimization for FRQI
    optimized_parameters = pso_optimization(image_data)
    
    # Generate and execute the FRQI circuit
    frqi_circuit = frqi_circuit(optimized_parameters, len(image_data))
    backend = Aer.get_backend('qasm_simulator')
    job = execute(frqi_circuit, backend, shots=1000)
    result = job.result()
    counts = result.get_counts(frqi_circuit)
    
    # Calculate Quantum Renyi Entropy
    alpha = 2  # You can choose any value for alpha
    qre = calculate_qre(counts, alpha)
    print(f"Quantum Renyi Entropy (alpha={alpha}): {qre}")
    
    # Plot histogram of measurement results
    plot_histogram(counts)
