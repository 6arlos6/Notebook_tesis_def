
from .main_fun import  representatives, iterate_minibatches, _cost, accuracy_score
from .main_fun import  test_intra_states, _test, w_r_json, density_matrix
import pandas as pd
import json
import pennylane as qml
from pennylane import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class Modelo_DRU:

  def __init__(self, modelo, f_loss, num_layers = 10, learning_rate = 0.1,
               epochs = 10, batch_size = 32, n_clases = 3, n_qubits = 1,
               random_state = 42, save_process = True, entanglement = False,
               path_save_w0 = "", path_save_states_0="", file_name_w = "test_w",
               file_name_state = "test_states", val_prc = 0.3, features = 2,
               alpha_noise = 0.0, excel_file_experimente = 'resultados_clasificacion.xlsx',
               save_w_states = False, verbose_test = False, save_excel_result = False):

        self.modelo = modelo
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_clases = n_clases

        self.state_labels = representatives(n_clases, n_qubits)
        self.state_labels = [density_matrix(s) for s in self.state_labels]

        self.ramdom_state = random_state
        np.random.seed(self.ramdom_state)
        # initialize random weights
        self.n_qubits = n_qubits
        self.params = np.random.uniform(size=(self.n_qubits*self.num_layers, 3), requires_grad=True)
        self.bias = np.random.uniform(size=(self.n_qubits*self.num_layers, 3), requires_grad=True)
        # save states
        # funcion de costo:
        self.f_loss = f_loss
        self.save_process = save_process
        self.entanglement = entanglement
        self.alpha_noise = alpha_noise
        # save list:
        self.acc_train = []
        self.acc_test = []
        self.loss_list_train = []
        self.loss_list_test = []
        # porcentaje validadion
        self.prc_val = val_prc
        # paths to save:
        self.file_name_state = file_name_state
        self.file_name_w = file_name_w
        self.path_save_states_0 = path_save_states_0
        self.path_save_w0 = path_save_w0
        # self.features
        self.features = features
        # excel_file
        self.excel_file_experimente = excel_file_experimente
        # save w, state
        self.save_w_states = save_w_states
        self.verbose_test = verbose_test
        self.save_excel_result = save_excel_result


  def fit(self, X, y):
    # dividir train and validacion:
    # train and val
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                     X, y, test_size = self.prc_val, random_state=self.ramdom_state)
    # aumentar con zeros:
    if self.features < 3:
      self.X_train = np.hstack((self.X_train, np.zeros((self.X_train.shape[0], 1), requires_grad=False)))
      self.X_val = np.hstack((self.X_val, np.zeros((self.X_val.shape[0], 1), requires_grad=False)))
    # guardar pesos y estados antes del entrenamiento
    if self.save_w_states == True:
      self.write_params(self.path_save_w0 + self.file_name_w + "_before_parameters.json")
      self.save_states(self.X_train, self.y_train, self.path_save_states_0 + self.file_name_state + "_before_states.json")
    # Entrenamiento por epocas:
    self.opt = qml.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999)
    for it in tqdm(range(self.epochs), desc="Epoch"):
      for Xbatch, ybatch in iterate_minibatches(self.X_train, self.y_train, batch_size=self.batch_size):
        self.params, self.bias = self.opt.step(
            lambda p, b: _cost(self.modelo,
                               p, Xbatch, ybatch, self.state_labels, b,
                               f_cost = self.f_loss, entanglement = self.entanglement, alpha_noise = self.alpha_noise),
            self.params, self.bias)
        if self.save_process == True:
          predicted_train = self.test(self.X_train, self.y_train)
          accuracy_train = self.score(self.y_train, predicted_train)
          predicted_test = self.test(self.X_val, self.y_val)
          accuracy_val = self.score(self.y_val, predicted_test)
          loss_train =  _cost(self.modelo, self.params, self.X_train, self.y_train, self.state_labels, self.bias,
                        f_cost = self.f_loss, entanglement = self.entanglement, alpha_noise = self.alpha_noise)
          loss_test =  _cost(self.modelo, self.params, self.X_val, self.y_val, self.state_labels, self.bias,
                        f_cost = self.f_loss, entanglement = self.entanglement, alpha_noise = self.alpha_noise)
          self.acc_train.append(accuracy_train)
          self.acc_test.append(accuracy_val)
          self.loss_list_train.append(loss_train)
          self.loss_list_test.append(loss_test)

    # Imprimir reporte final
    predicted_test = self.test(self.X_val, self.y_val)

    if self.verbose_test == True:
      print(classification_report(self.y_val, predicted_test))

    # Escribir resultado en excel:
    report_dict = classification_report(self.y_val, predicted_test, output_dict=True)
    mi_diccionario = self.report_to_dict(report_dict)

    if len(self.acc_train) == 0:
      predicted_train = self.test(self.X_train, self.y_train)
      accuracy_train = self.score(self.y_train, predicted_train)
      predicted_test = self.test(self.X_val, self.y_val)
      accuracy_val = self.score(self.y_val, predicted_test)
      loss_train =  _cost(self.modelo, self.params, self.X_train, self.y_train, self.state_labels, self.bias,
                    f_cost = self.f_loss, entanglement = self.entanglement, alpha_noise = self.alpha_noise)
      loss_test =  _cost(self.modelo, self.params, self.X_val, self.y_val, self.state_labels, self.bias,
                    f_cost = self.f_loss, entanglement = self.entanglement, alpha_noise = self.alpha_noise)
      self.acc_train.append(accuracy_train)
      self.acc_test.append(accuracy_val)
      self.loss_list_train.append(loss_train)
      self.loss_list_test.append(loss_test)

    acc_train_end = self.acc_train[-1]
    loss_train_end = self.loss_list_train[-1]
    loss_test_end = self.loss_list_test[-1]
    # save excel:
    if self.save_excel_result == True:
      self.dict_to_excel(mi_diccionario,
                        acc_train_end, loss_train_end, loss_test_end,
                        self.excel_file_experimente)

    # guardar pesos y estados despues del entrenamiento
    if self.save_w_states == True:
      self.save_states(self.X_train, self.y_train, self.path_save_states_0 + self.file_name_state+ "_after_states.json")
      self.write_params(self.path_save_w0 + self.file_name_w + "_after_parameters.json")

    return self.params, self.bias


  def test(self, X, y):
    predicted_train, fidel_train = _test(self.modelo, self.params, X, y,
                                         self.state_labels, self.bias,
                                         self.entanglement, alpha_noise = self.alpha_noise)
    return  predicted_train

  def score(self, y, predicted):
    accuracy_train = accuracy_score(y, predicted)
    return accuracy_train

  def save_states(self, X, y, path):
    states_4_q_enten_1 = test_intra_states(self.modelo, self.params, X, y,
                                           self.state_labels, self.bias,
                                           self.entanglement, alpha_noise = self.alpha_noise)
    w_r_json('w', path, file = states_4_q_enten_1)

  def write_params(self, path):
    # Crear un diccionario con los parámetros y sesgos
    params_dict = {"params": self.params.tolist(), "bias": self.bias.tolist()}
    # Escribir el diccionario en un archivo JSON
    with open(path, "w") as json_file:
        json.dump(params_dict, json_file)

  def report_to_dict(self, report):
    # Inicializa un diccionario para almacenar los resultados
    resultados_diccionario = {}

    # Agrega la métrica de accuracy al diccionario
    resultados_diccionario['accuracy'] = report['accuracy']

    # Itera a través de las claves del informe y guárdalas en el diccionario
    for clase, metricas in report.items():
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

  def dict_to_excel(self, mi_diccionario, acc_train_end, loss_train_end, loss_test_end, name_excel_file):
    # Supongamos que ya tienes el diccionario llamado 'mi_diccionario'
    # Puedes utilizar la función from_dict de pandas para convertir el diccionario en un DataFrame
    df_nuevo = pd.DataFrame.from_dict(mi_diccionario, orient='index')

    # Transpone el DataFrame para intercambiar filas y columnas
    df_nuevo_transpuesto = df_nuevo.transpose()

    # Expande los subdiccionarios y crea nuevas columnas para cada clave de los subdiccionarios
    df_nuevo_expandido = pd.json_normalize(df_nuevo_transpuesto.to_dict(), sep='_')

    # Colocar idenficador de prueba
    # self.idd = f"N_layers = {num_layers} + Noise = {alpha_noise} + f_cost = {f_loss.__name__}"
    df_nuevo_expandido["f_loss"] = self.f_loss.__name__
    df_nuevo_expandido["N_layers"] = self.num_layers
    df_nuevo_expandido["Noise"] = self.alpha_noise
    df_nuevo_expandido["end_train_accuracy"] = acc_train_end
    df_nuevo_expandido["end_train_loss"] =  loss_train_end
    df_nuevo_expandido["end_test_loss"] = loss_test_end

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