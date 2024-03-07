import pennylane as qml
from pennylane import numpy as np

# alpha_noise : parametros gobal que controla el ruido de amplitude damping

# Circuito con 1 qubit:
dev = qml.device("default.mixed", wires=1)
@qml.qnode(dev, interface=None)
def qcircuit_1_qubit_mixed(params, x, bias=None, entanglement=False, alpha_noise = 0.0):
  '''A variational quantum circuit representing the Universal classifier.
  Args:
      params (array[float]): array of parameters
      x (array[float]): single input vector
  Returns:
      float: qml state
  '''
  for i,p in enumerate(params):
    arg = np.multiply(p,x) + bias[i]
    arg1, arg2, arg3 = arg
    qml.Rot(arg3,arg1,arg2 , wires=0) # RZ * RY * RZ -> data
    qml.AmplitudeDamping(alpha_noise, wires=0)
    qml.Snapshot(f"ket_1_qubits_{i}")
  return qml.state()
# ====================================================================

# Circuito con 2 qubit:
dev = qml.device("default.mixed", wires=2)
@qml.qnode(dev, interface=None)
def qcircuit_2_qubit_mixed(params, x, bias = None, entanglement = False, alpha_noise = 0.0):
  #global alpha_noise
  """A variational quantum circuit representing the Universal classifier.
  Args:
      params (array[float]): array of parameters
      x (array[float]): single input vector
  Returns:
      float: qml state
  """
  n_layer = len(params) // 2
  for i in range(n_layer):
    arg_1 = np.multiply(params[i],x) + bias[i]
    arg_2 = np.multiply(params[i + n_layer],x) + bias[i + n_layer]
    qml.Rot(*arg_1 , wires=0) # RZ * RY * RZ -> data
    qml.AmplitudeDamping(alpha_noise, wires=0)
    qml.Rot(*arg_2 , wires=1) # RZ * RY * RZ -> data
    qml.AmplitudeDamping(alpha_noise, wires=1)
    qml.Snapshot(f"ket_2_qubits_{i}")
    if entanglement == True:
      qml.CZ(wires=[0,1])
      qml.Snapshot(f"ket_2_qubits_entanglement_{i}")
  return qml.state()
# =============================================================================

# Circuito con 4 qubit:
dev = qml.device("default.mixed", wires=4)
@qml.qnode(dev, interface=None)
def qcircuit_4_qubit_mixed(params, x, bias = None, entanglement = False, alpha_noise = 0.0):
  """A variational quantum circuit representing the Universal classifier.
  Args:
      params (array[float]): array of parameters
      x (array[float]): single input vector
  Returns:
      float: qml state
  """
  n_layer = len(params) // 4
  for i in range(n_layer):
    arg_1 = np.multiply(params[i],x) + bias[i]
    arg_2 = np.multiply(params[i + n_layer],x) + bias[i + n_layer]
    arg_3 = np.multiply(params[i + 2*n_layer],x) + bias[i + 2*n_layer]
    arg_4 = np.multiply(params[i + 3*n_layer],x) + bias[i + 3*n_layer]
    qml.Rot(*arg_1 , wires=0) # RZ * RY * RZ -> data
    qml.AmplitudeDamping(alpha_noise, wires=0)
    qml.Rot(*arg_2 , wires=1) # RZ * RY * RZ -> data
    qml.AmplitudeDamping(alpha_noise, wires=1)
    qml.Rot(*arg_3 , wires=2) # RZ * RY * RZ -> data
    qml.AmplitudeDamping(alpha_noise, wires=2)
    qml.Rot(*arg_4 , wires=3) # RZ * RY * RZ -> data
    qml.AmplitudeDamping(alpha_noise, wires=3)
    qml.Snapshot(f"ket_4_qubits_{i}")
    if entanglement == True:
      if i % 2 == 0 and i < n_layer-1:
        # par:
        qml.CZ(wires=[0,1])
        qml.CZ(wires=[2,3])
        qml.Snapshot(f"ket_4_qubits_entanglement_par{i}")
      elif i < n_layer-1:
        # impar:
        qml.CZ(wires=[1,2])
        qml.CZ(wires=[0,3])
        qml.Snapshot(f"ket_4_qubits_entanglement_impar{i}")
  return qml.state()