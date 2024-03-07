"""
! pip install pennylane
! pip install qutip
!pip install tqdm

Need libraries:
=====================================================================
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
from qutip import Bloch
import qutip

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from tqdm import tqdm
from sklearn.metrics import classification_report

import json
import pandas as pd

from collections import defaultdict
import matplotlib as mpl
from matplotlib import cm

from qutip import basis, sigmax,sigmay,sigmaz, tensor, Cubic_Spline, mesolve, Qobj, qeye, destroy
from qutip import fidelity as fidelity_qutip


"""# 2. Funciones propias"""

def fidelity(state0, state1):
  F  = qml.math.fidelity(state0, state1)
  return F

# Make a dataset of points inside and outside of a circle
def circle(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    """
    Generates a dataset of points with 1/0 labels inside a given radius.

    Args:
        samples (int): number of samples to generate
        center (tuple): center of the circle
        radius (float: radius of the circle

    Returns:
        Xvals (array[tuple]): coordinates of points
        yvals (array[int]): classification labels
    """
    Xvals, yvals = [], []

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = 0
        if np.linalg.norm(x - center) < radius:
            y = 1
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)

# Commented out IPython magic to ensure Python compatibility.
def make_IRIS(n_comp):
  # Cargar el conjunto de datos Iris
  iris = load_iris()
  # Acceder a las características y las etiquetas
  X = iris.data  # Características
  # Crear una instancia de PCA y especificar el número de componentes deseados (2 en este caso)
  pca = PCA(n_components=n_comp, random_state=None)
  # Aplicar PCA a las características
  X_pca = pca.fit(X).transform(X)
  y = iris.target  # Etiquetas
  # Percentage of variance explained for each components
  print(
      "explained variance ratio (first two components): %s"
#       % str(pca.explained_variance_ratio_)
  )
  return X_pca, y

def circle_v2(samples, centers=[[0.0, 0.0], [0.0, 0.0]], radii=[np.sqrt(0.8),np.sqrt(0.8 - 2/np.pi)]):
    """
    Generates a dataset of points with three class labels based on two radii.

    Args:
        samples (int): number of samples to generate
        centers (list of tuples): centers of the circles
        radii (list of floats): radii of the circles

    Returns:
        Xvals (array[tuple]): coordinates of points
        yvals (array[int]): classification labels
    """
    Xvals, yvals = [], []

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = 0

        for j, center in enumerate(centers):
            if np.linalg.norm(x - center) < radii[j]:
                y = j + 1

        Xvals.append(x)
        yvals.append(y)

    return np.array(Xvals, requires_grad=False), np.array(yvals, requires_grad=False)

def make_IRIS_v2(n_comp, n_classes):
    # Cargar el conjunto de datos Iris
    iris = load_iris()
    # Acceder a las características y las etiquetas
    X = iris.data  # Características
    # Crear una instancia de PCA y especificar el número de componentes deseados
    pca = PCA(n_components=n_comp, random_state=None)
    # Aplicar PCA a las características
    X_pca = pca.fit_transform(X)
    y = iris.target  # Etiquetas
    if n_classes == 1:
        # Filtrar muestras y etiquetas de una clase específica
        class_index = 0  # Índice de la clase deseada
        mask = (y == class_index)
        X_pca = X_pca[mask]
        y = y[mask]
        return X_pca, y
    elif n_classes == 2:
        # Filtrar muestras y etiquetas de dos clases específicas
        class_indices = [0, 1]  # Índices de las clases deseadas
        mask = np.isin(y, class_indices)
        X_pca = X_pca[mask]
        y = y[mask]
        return X_pca, y
    else:
      return X_pca,y

def representatives(classes, qubits_lab):
  """
  This function creates the label states for the classification task
  INPUT:
      -classes: number of classes of our problem
      -qubits_lab: how many qubits will store the labels
  OUTPUT:
      -reprs: the label states
  """
  reprs = np.zeros((classes, 2**qubits_lab), dtype = 'complex', requires_grad=False)
  if qubits_lab == 1:
      if classes == 0:
          raise ValueError('Nonsense classifier')
      if classes == 1:
          raise ValueError('Nonsense classifier')
      if classes == 2:
          reprs[0] = np.array([1, 0])
          reprs[1] = np.array([0, 1])
      if classes == 3:
          reprs[0] = np.array([1, 0])
          reprs[1] = np.array([1 / 2, np.sqrt(3) / 2])
          reprs[2] = np.array([1 / 2, -np.sqrt(3) / 2])
      if classes == 4:
          reprs[0] = np.array([1, 0])
          reprs[1] = np.array([1 / np.sqrt(3), np.sqrt(2 / 3)])
          reprs[2] = np.array([1 / np.sqrt(3), np.exp(1j * 2 * np.pi / 3) * np.sqrt(2 / 3)])
          reprs[3] = np.array([1 / np.sqrt(3), np.exp(-1j * 2 * np.pi / 3) * np.sqrt(2 / 3)])
      if classes == 6:
          reprs[0] = np.array([1, 0])
          reprs[1] = np.array([0, 1])
          reprs[2] = 1 / np.sqrt(2) * np.array([1, 1])
          reprs[3] = 1 / np.sqrt(2) * np.array([1, -1])
          reprs[4] = 1 / np.sqrt(2) * np.array([1, 1j])
          reprs[5] = 1 / np.sqrt(2) * np.array([1, -1j])

  if qubits_lab == 2:
      if classes == 0:
          raise ValueError('Nonsense classifier')
      if classes == 1:
          raise ValueError('Nonsense classifier')
      if classes == 2:
          reprs[0] = np.array([1, 0, 0, 0])
          reprs[1] = np.array([0, 0, 0, 1])
      if classes == 3:
          reprs[0] = np.array([1, 0, 0, 0])
          reprs[1] = np.array([0, 1, 0, 0])
          reprs[2] = np.array([0, 0, 1, 0])
      if classes == 4:
          reprs[0] = np.array([1, 0, 0, 0])
          reprs[1] = np.array([0, 1, 0, 0])
          reprs[2] = np.array([0, 0, 1, 0])
          reprs[3] = np.array([0, 0, 0, 1])
  if qubits_lab == 4:
      if classes == 2:
        reprs[0] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        reprs[1] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      if classes == 3:
        reprs[0] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        reprs[1] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        reprs[2] = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

  return reprs

def to_spherical(state):
    r0 = np.abs(state[0])
    ϕ0 = np.angle(state[0])
    r1 = np.abs(state[1])
    ϕ1 = np.angle(state[1])
    r = np.sqrt(r0 ** 2 + r1 ** 2)
    θ = 2 * np.arccos(r0 / r)
    ϕ = ϕ1 - ϕ0
    return [r, θ, ϕ]

def to_cartesian(polar):
    r = polar[0]
    θ = polar[1]
    ϕ = polar[2]
    x = r * np.sin(θ) * np.cos(ϕ)
    y = r * np.sin(θ) * np.sin(ϕ)
    z = r * np.cos(θ)
    return [x, y, z]

def visualization_1_qubit(f_q_circuit, X, Y, params, ax, angles=[-90,180], sz=1, bias=None, entanglement=False):
  nf, nc = X.shape
  print(type(Y))
  clases = list(set(Y))

  if nc == 1:
    X = np.hstack((X, np.zeros((nf, 2), requires_grad=False)))
  elif nc == 2:
    X = np.hstack((X, np.zeros((nf, 1), requires_grad=False)))
  dict_coord = {}
  # Agregar datos al diccionario
  def agregar_dato(key, dato):
      dict_coord.setdefault(key, []).append(dato)
  for i in range(len(X)):
    state = f_q_circuit(params, X[i], bias, entanglement)
    alpha, beta = state
    my_state = [complex(alpha), complex(beta)]
    polar = to_spherical(my_state)
    x, y, z = to_cartesian(polar)
    agregar_dato(f'cl_{int(Y[i])}_x', x)
    agregar_dato(f'cl_{int(Y[i])}_y', y)
    agregar_dato(f'cl_{int(Y[i])}_z', z)
  bloch_sphere = Bloch(view=(angles))
  bloch_sphere.axes = ax
  bloch_sphere.point_size = sz
  color_to_cycle = ["#FF0000", "#0000FF", "#006400"]
  my_colors = [ color_to_cycle[cl] for cl in clases ]
  bloch_sphere.point_color = my_colors
  for cl in clases:
    X_key = f'cl_{cl}_x'
    Y_key = f'cl_{cl}_y'
    Z_key = f'cl_{cl}_z'
    pnts = [dict_coord[X_key], dict_coord[Y_key], dict_coord[Z_key]]
    bloch_sphere.add_points(pnts)
    bloch_sphere.render()
  bloch_sphere.show()

# Visualizacion Train, test, true
def plot_data(x, y, fig=None, ax=None):
    """
    Plot data with red/blue values for a binary classification.

    Args:
        x (array[tuple]): array of data points as tuples
        y (array[int]): array of data points as tuples
    """
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    reds = y == 0
    blues = y == 1
    ax.scatter(x[reds, 0], x[reds, 1], c="red", s=20, edgecolor="k")
    ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20, edgecolor="k")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

# Visualizacion Train, test, true
def plot_data_v2(x, y, fig=None, ax=None):
    """
    Plot data with red/blue values for a binary classification.

    Args:
        x (array[tuple]): array of data points as tuples
        y (array[int]): array of data points as tuples
    """
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    reds = y == 0
    blues = y == 1
    green = y == 2
    ax.scatter(x[reds, 0], x[reds, 1], c="red", s=20, edgecolor="k")
    ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20, edgecolor="k")
    ax.scatter(x[green, 0], x[green, 1], c="green", s=20, edgecolor="k")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

def accuracy_score(y_true, y_pred):
    """Accuracy score.

    Args:
        y_true (array[float]): 1-d array of targets
        y_predicted (array[float]): 1-d array of predictions
        state_labels (array[float]): 1-d array of state representations for labels

    Returns:
        score (float): the fraction of correctly classified samples
    """
    score = y_true == y_pred
    return score.sum() / len(y_true)

def iterate_minibatches(inputs, targets, batch_size):
    """
    A generator for batches of the input data
    Args:
        inputs (array[float]): input data
        targets (array[float]): targets
    Returns:
        inputs (array[float]): one batch of input data of length `batch_size`
        targets (array[float]): one batch of targets of length `batch_size`
    """
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], targets[idxs]

def test(model, params, x, y, state_labels=None, bias=None, entanglement = False):
  """
  Tests on a given set of data. This function predicts in function of max fidelity.
  Args:
      params (array[float]): array of parameters
      x (array[float]): 2-d array of input vectors
      y (array[float]): 1-d array of targets
      state_labels (array[float]): 1-d array of state representations for labels
  Returns:
      predicted (array([int]): predicted labels for test data
      output_states (array[float]): output quantum states from the circuit
  """
  fidelity_values = []
  dm_labels = [s for s in state_labels]
  #print(f'len labels = {len(dm_labels)}')
  predicted = []
  for i in range(len(x)):
    #fidel_function = lambda y: qcircuit(params, x[i], y)
    fidelities = []
    for dm in dm_labels:
      state_output = model(params, x[i], bias=bias, entanglement = entanglement)
      # fidelity_cost(model, params, x, y, state_labels=None)
      f = fidelity(qml.math.dm_from_state_vector(state_output), qml.math.dm_from_state_vector(dm))
      fidelities.append(f)
    best_fidel = np.argmax(fidelities)
    #print(fidelities, best_fidel, y[i])
    predicted.append(best_fidel)
    fidelity_values.append(fidelities)
  return np.array(predicted), np.array(fidelity_values)


def test_intra_states(model, params, x, y, state_labels=None, bias=None, entanglement = False, alpha_noise = 0.0):
  # Esta funcion permite obtener los estados cuenticos de un circuito cuentico
  # y devuelve un diccionario donde los estados se guardan para ser utilizados
  # en test de fidelidas posteriores.
  dict_states_by_sample = []
  for i in range(len(x)):
    if i == 0:
      print(x[i])
    diccionario_estados = qml.snapshots(model)(params,x[i], bias=bias, entanglement = entanglement, alpha_noise = alpha_noise )
    dict_states_by_sample.append(diccionario_estados)
  return dict_states_by_sample

def w_r_json(write_or_read, path, file = []):
  # Esta función permite escribir o leer diccionarios como json files.

  if write_or_read == 'w':
  # Si se desea escribir se guarda la parte real e imaginaria en una sub lista para
  # cada uno de los estados cuanticos:
    list_new = []
    for dict_i in file:
      json_errors = {k: [v.numpy().real.tolist(), v.numpy().imag.tolist()] for k, v in dict_i.items()}
      list_new.append(json_errors)
    # Escribir la lista de diccionarios en un archivo JSON
    with open(path, 'w') as archivo_json:
        json.dump(list_new, archivo_json)

  elif write_or_read == 'r':
  # Si se desea leer un archivo json se recupera el archivo coomo una lista de listas
  # y se devuelve un formato tipo diccionario pero con el formato en numero complejos
  # nuevamente.
    with open(path, 'r') as archivo_json:
        lista_recuperada = json.load(archivo_json)
    list_new = []
    for dict_i in lista_recuperada:
      json_errors = {k: np.array(v[0]) + 1j * np.array(v[1])  for k, v in dict_i.items()}
      list_new.append(json_errors)
    return lista_recuperada, list_new
  
# Define output labels as quantum state vectors
def density_matrix(state):
    """Calculates the density matrix representation of a state.

    Args:
        state (array[complex]): array representing a quantum state vector

    Returns:
        dm: (array[complex]): array representing the density matrix
    """
    nf, = state.shape
    state = np.reshape(state, (nf, 1))
    #print(f"State = {state}")
    #print(f"conj state = {np.conj(state).T}")
    return state @ (np.conj(state).T) # Outer product

def density_matrix_1(state):
    """Calculates the density matrix representation of a state.

    Args:
        state (array[complex]): array representing a quantum state vector

    Returns:
        dm: (array[complex]): array representing the density matrix
    """

    nf,nc = state.shape
    if nc == 1:
      state = np.reshape(state, (nf, 1))
      #print(f"State = {state}")
      #print(f"conj state = {np.conj(state).T}")
      return state @ (np.conj(state).T) # Outer product
    else:
      return state


def _cost(model, params, x, y, state_labels=None, bias = None, f_cost = None, entanglement = False, alpha_noise = 0.0):
    """Cost function to be minimized.
    Args:
        params (array[float]): array of parameters
        x (array[float]): 2-d array of input vectors
        y (array[float]): 1-d array of targets
        state_labels (array[float]): array of state representations for labels
    Returns:
        float: loss value to be minimized
    """
    # Compute prediction for each input in data batch
    loss = 0.0
    # dm_labels = state_labels
    #dm_labels = [s for s in state_labels]
    for i in range(len(x)):
      state_output = model(params, x[i], bias = bias, entanglement = entanglement, alpha_noise = alpha_noise)
      f =  f_cost(state_output, state_labels[y[i]])
      loss = loss + f
    return loss / len(x)

def _test(model, params, x, y, state_labels=None, bias=None, entanglement = False, alpha_noise = 0.0):
  """
  Tests on a given set of data.
  Args:
      params (array[float]): array of parameters
      x (array[float]): 2-d array of input vectors
      y (array[float]): 1-d array of targets
      state_labels (array[float]): 1-d array of state representations for labels
  Returns:
      predicted (array([int]): predicted labels for test data
      output_states (array[float]): output quantum states from the circuit
  """
  fidelity_values = []
  #dm_labels = [s for s in state_labels]
  #dm_labels = [density_matrix(s) for s in state_labels]
  #print(f'len labels = {len(dm_labels)}')
  predicted = []
  for i in range(len(x)):
    #fidel_function = lambda y: qcircuit(params, x[i], y)
    fidelities = []
    for dm in state_labels:
      state_output = model(params, x[i], bias=bias, entanglement = entanglement, alpha_noise = alpha_noise)
      # fidelity_cost(model, params, x, y, state_labels=None)
      f = fidelity(state_output, dm)
      fidelities.append(f)
    best_fidel = np.argmax(fidelities)
    #print(fidelities, best_fidel, y[i])
    predicted.append(best_fidel)
    fidelity_values.append(fidelities)
  return np.array(predicted), np.array(fidelity_values)


# Funcion para predecir mediante fidelidad cuando un estado de salida
# pertenece a una clase.
def _test_pulse_model(state_output, state_labels):
  dm_labels = [density_matrix(s) for s in state_labels]
  fidelities = []
  for dm in dm_labels:
    #state_output = model(params, x[i], bias=bias, entanglement = entanglement, alpha_noise = alpha_noise)
    # fidelity_cost(model, params, x, y, state_labels=None)
    f = fidelity(state_output, dm)
    fidelities.append(f)
  best_fidel = np.argmax(fidelities)
  return best_fidel




def traducir_a_positivo(angulo):
    while angulo < 0:
        angulo += 2 * np.pi
    return angulo


# Pulso cosenoidal

def pulse_x_cos(t, args):
    t_init = args["t_init"]
    t_final = args["t_final"]
    w = args["w"]
    pulse = np.heaviside((t-t_init), 0.0) * np.heaviside(-(t - t_final), 0.0)
    return np.cos(w*t)*pulse

# Pulsos para la clase

def pulse_x(t, args):
  t_init = args["t_init"]
  t_final = args["t_final"]
  y = np.heaviside((t-t_init), 0.0) * np.heaviside(-(t - t_final), 0.0)
  return y

def pulse_z(t, args):
  t_init = args["t_init"]
  t_final = args["t_final"]
  y = np.heaviside((t-t_init), 0.0) * np.heaviside(-(t - t_final), 0.0)
  return y

def pulse_x_with_noise(t, args):
  t_init = args["t_init"]
  t_final = args["t_final"]
  std_noise = args["std_noise"]
  noise = np.random.normal(loc=0, scale=std_noise, size = len(t))
  y = np.heaviside((t-t_init), 0.0) * np.heaviside(-(t - t_final), 0.0) + noise
  return y

def pulse_z_with_noise(t, args):
  t_init = args["t_init"]
  t_final = args["t_final"]
  std_noise = args["std_noise"]
  noise = np.random.normal(loc=0, scale=std_noise, size = len(t))
  y = np.heaviside((t-t_init), 0.0) * np.heaviside(-(t - t_final), 0.0) + noise
  return y


class report_to_excel:
  def __init__(self, report, path_to_save):
    self.report = report
    self.path_to_save = path_to_save

  def report_to_dict(self):
    # Inicializa un diccionario para almacenar los resultados
    resultados_diccionario = {}

    # Agrega la métrica de accuracy al diccionario
    resultados_diccionario['accuracy'] = self.report['accuracy']

    # Itera a través de las claves del informe y guárdalas en el diccionario
    for clase, metricas in self.report.items():
        # Ignora la clave 'accuracy' que ya hemos agregado por separado
        if clase != 'accuracy':
            # Crea un diccionario para almacenar las métricas por clase
            metricas_clase = {}

            # Itera a través de las métricas para la clase actual
            for metrica, valor in metricas.items():
                # Crea una clave compuesta usando el nombre de la clase y el nombre de la métrica
                clave = f"{clase}_{metrica}"
                # Guarda el valor en el diccionario
                metricas_clase[clave] = valor

            # Guarda las métricas por clase en el diccionario general
            resultados_diccionario[clase] = metricas_clase

    return resultados_diccionario

  def dict_to_excel(self, N_qubits, tf_noise, std_noide, B0, B1, num_layers, entanglement, f_loss, T1, T2, alpha_noise):

    name_excel_file = self.path_to_save
    mi_diccionario = self.report_to_dict()
    # Supongamos que ya tienes el diccionario llamado 'mi_diccionario'
    # Puedes utilizar la función from_dict de pandas para convertir el diccionario en un DataFrame
    df_nuevo = pd.DataFrame.from_dict(mi_diccionario, orient='index')

    # Transpone el DataFrame para intercambiar filas y columnas
    df_nuevo_transpuesto = df_nuevo.transpose()

    # Expande los subdiccionarios y crea nuevas columnas para cada clave de los subdiccionarios
    df_nuevo_expandido = pd.json_normalize(df_nuevo_transpuesto.to_dict(), sep='_')

    # Colocar idenficador de prueba
    # self.idd = f"N_layers = {num_layers} + Noise = {alpha_noise} + f_cost = {f_loss.__name__}"
    df_nuevo_expandido["f_loss"] = f_loss.__name__
    df_nuevo_expandido["N_layers"] = num_layers
    df_nuevo_expandido["Noise"] = alpha_noise
    df_nuevo_expandido["N_qubits"] = N_qubits
    df_nuevo_expandido["tf_noise"] =  tf_noise
    df_nuevo_expandido["std_noide"] = std_noide
    df_nuevo_expandido["B0"] = B0
    df_nuevo_expandido["B1"] = B1
    df_nuevo_expandido["num_layers"] = num_layers
    df_nuevo_expandido["entanglement"] = entanglement
    df_nuevo_expandido["T1"] = T1
    df_nuevo_expandido["T2"] = T2

    # Nombre del archivo de Excel existente
    nombre_archivo_excel = name_excel_file

    try:
        # Intenta cargar el archivo Excel existente
        df_existente = pd.read_excel(nombre_archivo_excel)
    except FileNotFoundError:
        # Si el archivo no existe, crea un DataFrame vacío
        df_existente = pd.DataFrame()

    # Concatena el DataFrame existente con el nuevo DataFrame
    df_final = pd.concat([df_existente, df_nuevo_expandido], ignore_index=True)

    # Guarda el DataFrame combinado en el archivo de Excel
    df_final.to_excel(nombre_archivo_excel, index=False)