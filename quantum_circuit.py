import pennylane as qml
from pennylane import numpy as np 
import torch
#We import pennylane for quantum computing functionalities
#We import pennylane's numpy for compatibility with quantum operations

n_qubits = 2
n_layers = 3

#we gonna create quantum simulator device
dev= qml.device("default.qubit", wires= n_qubits)

@qml.qnode(dev, interface= "torch")
def quantum_circuit(inputs, weights):
    qml.template.AngleEmbedding(inputs, wires= range(n_qubits))
    qml.template.StronglyEntanglingLayers(weights, wires= range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shape = qml.templates.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
weights = torch.randn(weight_shape, requires_grad=True)

print(f"Weight shape: {weight_shape}")  # (3, 2) for 3 layers, 2 qubits