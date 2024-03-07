
import pennylane as qml
from pennylane import numpy as np


def fidelity(state0, state1):
  F  = qml.math.fidelity(state0, state1)
  return F


def fidelity_cost(dm_pred, dm_true):
  f = fidelity(dm_pred, dm_true)
  return 1 - f



# Distancia de traza:
verbose_trace_dist = 0

Proceso = '''
Entrada prediccion:
=======================
{}
Entrada true:
=======================
{}
=======================
=======================
Resta:
=======================
{}
Resta trasnpuesta conjugada:
=======================
{}
Producto:
=======================
{}
SQRT:
=======================
{}
Traza:
========================
{}
'''

def Trace_Distance(dm_pred, dm_true):
  global verbose_trace_dist
  diff = dm_pred - dm_true
  diff_t = np.transpose(diff)
  diff_t = np.conjugate(diff_t)
  diff_m = np.dot(diff_t, diff)
  #sqrt_diff = fractional_matrix_power(diff_t @ diff,0.5)
  eigenvalues, eigenvectors = np.linalg.eig(diff_m)
  inverted_sqrt_eigenvalues = np.sqrt(eigenvalues)
  sqrt_diff = np.dot(np.dot(eigenvectors, np.diag(inverted_sqrt_eigenvalues)), np.linalg.inv(eigenvectors))
  trac_sqrt = 0.5*np.real(np.trace(sqrt_diff))
  my_salida = trac_sqrt
  if verbose_trace_dist == 1:
    print(Proceso.format(dm_pred,
                         dm_true,
                         diff,
                         diff_t,
                         diff_m,
                         sqrt_diff,
                         np.trace(sqrt_diff)))
    verbose_trace_dist = 0
  return trac_sqrt

# ================================================


# Distancia de traza:
verbose_trace_dist = 0

Proceso = '''
Entrada prediccion:
=======================
{}
Entrada true:
=======================
{}
=======================
=======================
Resta:
=======================
{}
Resta trasnpuesta conjugada:
=======================
{}
Producto:
=======================
{}
SQRT:
=======================
{}
Traza:
========================
{}
'''

def Trace_Distance_v3(dm_pred, dm_true):

  global verbose_trace_dist

  condicion_1 = np.count_nonzero(dm_true) == 1
  condicion_2 = np.array_equal(np.diag(np.diag(dm_true)), dm_true)
  if condicion_1 and condicion_2:
    indice = np.where(np.diag(dm_true) == 1)[0][0]
    const = 0.999
    e = dm_true[indice,indice] - const
    nf,nc = dm_true.shape
    res = e/nf
    dm_true[np.diag_indices_from(dm_true)] = res
    dm_true[indice,indice] = const
  diff = dm_pred - dm_true
  diff_t = np.transpose(diff)
  diff_t = np.conjugate(diff_t)
  diff_m = np.dot(diff_t, diff)

  eigenvalues, eigenvectors = np.linalg.eig(diff_m)
  inverted_sqrt_eigenvalues = np.sqrt(eigenvalues)
  sqrt_diff = np.dot(np.dot(eigenvectors, np.diag(inverted_sqrt_eigenvalues)), np.linalg.inv(eigenvectors))
  trac_sqrt = 0.5*np.real(np.trace(sqrt_diff))
  my_salida = trac_sqrt
  if verbose_trace_dist == 1:
    print(Proceso.format(dm_pred,
                         dm_true,
                         diff,
                         diff_t,
                         diff_m,
                         sqrt_diff,
                         np.trace(sqrt_diff)))
    verbose_trace_dist = 0
  return trac_sqrt

# ========================================================

# Entropia de von newman
verbose_trace_dist = 0
Proceso_2 = '''
Entrada prediccion:
=======================
{}
Entrada true:
=======================
{}
=======================
=======================
Log prediccion:
=======================
{}
Log True:
=======================
{}
Producto:
=======================
{}
Divergencia:
=======================
{}
'''

def Von_Neumman_Divergence(dm_pred, dm_true):
  global verbose_trace_dist
  #log_p = logm(dm_pred)
  if dm_true[0,0] == 1:
    #print("Hola entre")
    #print(dm_true)
    p = 0.001
    X = np.array([[0,1],[1,0]])
    dm_true = (1 - p)*dm_true + p*np.dot(np.dot(X,dm_true),X)
  eigenvalues, eigenvectors = np.linalg.eig(dm_pred)
  inverted_sqrt_eigenvalues = np.log(eigenvalues)
  log_p = np.dot(np.dot(eigenvectors, np.diag(inverted_sqrt_eigenvalues)), np.linalg.inv(eigenvectors))
  #log_rho = logm(dm_true)
  eigenvalues, eigenvectors = np.linalg.eig(dm_true)
  inverted_sqrt_eigenvalues = np.log(eigenvalues)
  log_rho = np.dot(np.dot(eigenvectors, np.diag(inverted_sqrt_eigenvalues)), np.linalg.inv(eigenvectors))
  diff = log_p - log_rho
  prod = np.dot(dm_pred, diff)
  vkld = np.real(np.trace(prod))

  if verbose_trace_dist == 1:
    print(Proceso_2.format(dm_pred,
                         dm_true,
                         log_p,
                         log_rho,
                         prod,
                         vkld))
  return vkld

# =============================================================
# Entropia de von newman
verbose_trace_dist = 1
Proceso_2 = '''
Entrada prediccion:
=======================
{}
Entrada true:
=======================
{}
=======================
=======================
Log prediccion:
=======================
{}
Log True:
=======================
{}
Producto:
=======================
{}
Divergencia:
=======================
{}
'''

def Von_Neumman_Divergence_v2(dm_pred, dm_true):
  # Imprimir un paso:
  global verbose_trace_dist
  # Condiciones para operar matrices:
  condicion_1 = np.count_nonzero(dm_true) == 1
  condicion_2 = np.array_equal(np.diag(np.diag(dm_true)), dm_true)
  if condicion_1 and condicion_2:
    #dm_true = dm_true.astype(np.float64)
    indice = np.where(np.diag(dm_true) == 1)[0][0]
    const = 0.999
    e = dm_true[indice,indice] - const
    nf,nc = dm_true.shape
    res = e/nf
    dm_true[np.diag_indices_from(dm_true)] = res
    dm_true[indice,indice] = const
  # continuar:
  eigenvalues, eigenvectors = np.linalg.eig(dm_pred)
  inverted_sqrt_eigenvalues = np.log(eigenvalues)
  log_p = np.dot(np.dot(eigenvectors, np.diag(inverted_sqrt_eigenvalues)), np.linalg.inv(eigenvectors))
  #log_rho = logm(dm_true)
  eigenvalues, eigenvectors = np.linalg.eig(dm_true)
  inverted_sqrt_eigenvalues = np.log(eigenvalues)
  log_rho = np.dot(np.dot(eigenvectors, np.diag(inverted_sqrt_eigenvalues)), np.linalg.inv(eigenvectors))
  diff = log_p - log_rho
  prod = np.dot(dm_pred, diff)
  vkld = np.real(np.trace(prod))

  if verbose_trace_dist == 1:
    print(Proceso_2.format(dm_pred,
                         dm_true,
                         log_p,
                         log_rho,
                         prod,
                         vkld))
    print("=========================")
    print(f"nf = {nf}")
  return vkld
# ===============================================================

# Entropia de renyi

verbose_trace_dist = 0

Proceso_3 = '''
Entrada prediccion:
=======================
{}
Entrada true:
=======================
{}
=======================
=======================
Potencia:
=======================
{}
dm_true_elevada:
=======================
{}
Arg 1 (RHO * P * RHO):
=======================
{}
Traza 1:
=======================
{}
Arg 2 LOG(*):
=======================
{}
log:
=======================
{}
Divergencia:
=======================
{}
'''

def matrix_pow(p, power):
  eigenvalues, eigenvectors = np.linalg.eig(p)
  inverted_sqrt_eigenvalues = np.sign(eigenvalues) * (np.abs(eigenvalues)) ** (power)
  p_powered = np.dot(np.dot(eigenvectors, np.diag(inverted_sqrt_eigenvalues)), np.linalg.inv(eigenvectors))
  return p_powered

def Renyi_Divergence(dm_pred, dm_true):

  global verbose_trace_dist, alpha_R
  if dm_true[0,0] == 1:
    #print("Hola entre")
    #print(dm_true)
    p = 0.001
    X = np.array([[0,1],[1,0]])
    dm_true = (1 - p)*dm_true + p*np.dot(np.dot(X,dm_true),X)
  # sigma:
  power_a = (1-alpha_R)/(2*alpha_R)
  dm_true_powered = matrix_pow(dm_true, power_a)
  # product:
  arg_1 = np.dot(np.dot(dm_true_powered, dm_pred), dm_true_powered)
  # trace:
  tra = np.real(np.trace( matrix_pow(arg_1, alpha_R) ))
  # arg of log:
  arg_2 = (1/(np.real(np.trace(dm_pred)))) * tra
  # log:
  arg_3 = np.log(arg_2)
  # Divergece:
  D = (1/(alpha_R - 1)) * arg_3

  if verbose_trace_dist == 1:
    print(Proceso_3.format(dm_pred,
                         dm_true,
                         power_a,
                         dm_true_powered,
                         arg_1,
                         tra,
                         arg_2,
                         arg_3,
                        D))
    verbose_trace_dist = 0
  return D

# ===================================================================

# Entropia de renyi

verbose_trace_dist = 0

Proceso_3 = '''
Entrada prediccion:
=======================
{}
Entrada true:
=======================
{}
=======================
=======================
Potencia:
=======================
{}
dm_true_elevada:
=======================
{}
Arg 1 (RHO * P * RHO):
=======================
{}
Traza 1:
=======================
{}
Arg 2 LOG(*):
=======================
{}
log:
=======================
{}
Divergencia:
=======================
{}
'''

def matrix_pow(p, power):
  eigenvalues, eigenvectors = np.linalg.eig(p)
  inverted_sqrt_eigenvalues = np.sign(eigenvalues) * (np.abs(eigenvalues)) ** (power)
  p_powered = np.dot(np.dot(eigenvectors, np.diag(inverted_sqrt_eigenvalues)), np.linalg.inv(eigenvectors))
  return p_powered

def Renyi_Divergence_v2(dm_pred, dm_true):
  global verbose_trace_dist, alpha_R
  # check si la matriz tiene solo un 1 en su diagonal
  condicion_1 = np.count_nonzero(dm_true) == 1
  condicion_2 = np.array_equal(np.diag(np.diag(dm_true)), dm_true)
  if condicion_1 and condicion_2:
    #dm_true = dm_true.astype(np.float64)
    indice = np.where(np.diag(dm_true) == 1)[0][0]
    const = 0.999
    e = dm_true[indice,indice] - const
    nf,nc = dm_true.shape
    res = e/nf
    dm_true[np.diag_indices_from(dm_true)] = res
    dm_true[indice,indice] = const
  # sigma:
  power_a = (1-alpha_R)/(2*alpha_R)
  dm_true_powered = matrix_pow(dm_true, power_a)
  # product:
  arg_1 = np.dot(np.dot(dm_true_powered, dm_pred), dm_true_powered)
  # trace:
  tra = np.real(np.trace( matrix_pow(arg_1, alpha_R) ))
  # arg of log:
  arg_2 = (1/(np.real(np.trace(dm_pred)))) * tra
  # log:
  arg_3 = np.log(arg_2)
  # Divergece:
  D = (1/(alpha_R - 1)) * arg_3

  if verbose_trace_dist == 1:
    print(Proceso_3.format(dm_pred,
                         dm_true,
                         power_a,
                         dm_true_powered,
                         arg_1,
                         tra,
                         arg_2,
                         arg_3,
                        D))
    verbose_trace_dist = 0
  return D




