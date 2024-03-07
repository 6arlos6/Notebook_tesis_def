
from .main_fun import traducir_a_positivo
import pennylane as qml
from pennylane import numpy as np
from collections import defaultdict
from qutip import basis, sigmax,sigmay,sigmaz, tensor, Cubic_Spline, mesolve, Qobj, qeye, destroy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qutip import Bloch
import matplotlib as mpl
from .main_fun import pulse_x, pulse_z, pulse_x_with_noise, pulse_z_with_noise
from matplotlib import cm

class Quantum_Spin_Proces:

  def __init__(self, h = 1, gir = 1.760e11, B0 = 10e-3, nf = 1, N_qubits = 1, J = 1, tf_noise = False,
                 noise_std = 0.01, B1_offset = 0, n_points_pulse_Ri = 100,
                 n_points_pulse_2Qbits = 100, n_swap = 1, T1 = 1e3, T2 = 1e3,
                 tf_quantum_noise = False):
        self.gir = gir
        self.B0 = B0
        self.B1 = B1_offset
        self.Dt = -7
        self.h = h
        self.nf = nf
        self.N_qubits = N_qubits
        self.J = J
        self.tf_noise = tf_noise
        self.noise_std = noise_std
        self.B1_offset = B1_offset
        self.n_points_pulse_Ri = n_points_pulse_Ri
        self.n_points_pulse_2Qbits = n_points_pulse_2Qbits
        self.n_swap = n_swap
        self.global_time = 0
        self.dict_states = {}
        self.pulse_type = defaultdict(list)
        self.T1 = T1
        self.T2 = T2
        self.tf_quantum_noise = tf_quantum_noise

  def Rz(self, alpha, ket_0, q_obj = 0, tf_expect = True):
      # Estados iniciales y qubit objetivo:
      self.q_obj = q_obj
      self.ket_0 = ket_0
      # parametros de compuerta:
      self.ω_x = 0
      self.ω_z = self.gir * self.B0
      alpha  = traducir_a_positivo(alpha)
      self.delt_t = (alpha)/self.ω_z
      self.B1 = 0
      self.O_x = self.gir*(self.B1/2)
      # solucion:
      out = self.Hamiltonian_solve(tf_expect)
      return out

  def Rx(self, alpha, ket_0, q_obj = 0, tf_expect = True):
      # Estados iniciales y qubit objetivo:
      self.q_obj = q_obj
      self.ket_0 = ket_0
      # parametros de compuerta:
      self.ω_x = self.gir * self.B0
      self.ω_z = self.gir * self.B0
      self.delt_t = (2*np.pi*self.nf)/self.ω_x
      self.B1 = (alpha * 2)/(self.gir * self.delt_t)
      self.O_x = self.gir*(self.B1/2)
      # solucion:
      out = self.Hamiltonian_solve(tf_expect)
      return out

  def Ry(self, alpha, ket_0, q_obj = 0, tf_expect = True):

      out_1 = self.Rx(np.pi/2, ket_0, q_obj=q_obj, tf_expect = False)
      end_state_1 = out_1.states[-1]
      out_2 = self.Rz(alpha, end_state_1,q_obj=q_obj, tf_expect = False)
      end_state_2 = out_2.states[-1]
      out_3 = self.Rx(-np.pi/2, end_state_2,q_obj=q_obj, tf_expect = False)
      end_state_3 = out_3.states[-1]
      if tf_expect == True:
        out_1_exp = self.Rx(np.pi/2, ket_0, q_obj=q_obj, tf_expect = True)
        out_2_exp = self.Rz(alpha, end_state_1,q_obj=q_obj, tf_expect = True)
        out_3_exp = self.Rx(-np.pi/2, end_state_2,q_obj=q_obj, tf_expect = True)
        out = [out_1_exp, out_2_exp, out_3_exp]
      else:
        out = out_3
      return out

  def H(self, ket_0, alpha = np.pi/2, q_obj = 0, tf_expect = True):
      out_1 = self.Rz(np.pi/2, ket_0, q_obj=q_obj, tf_expect = False)
      end_state_1 = out_1.states[-1]
      out_2 = self.Rx(alpha, end_state_1,q_obj=q_obj, tf_expect = False)
      end_state_2 = out_2.states[-1]
      out_3 = self.Rz(np.pi/2, end_state_2,q_obj=q_obj, tf_expect = False)
      end_state_3 = out_3.states[-1]
      if tf_expect == True:
        out_1_exp = self.Rz(np.pi/2, ket_0, q_obj=q_obj, tf_expect = True)
        out_2_exp = self.Rx(alpha, end_state_1,q_obj=q_obj, tf_expect = True)
        out_3_exp = self.Rz(np.pi/2, end_state_2,q_obj=q_obj, tf_expect = True)
        out = [out_1_exp, out_2_exp, out_3_exp]
      else:
        out = out_3
      return out

  def SWAP(self, ket_0, measure_op, q_obj = [0,1], tf_expectt = True):
      self.q_obj = q_obj
      self.ket_0 = ket_0
      self.Dt = np.pi/(self.J*self.n_swap)
      print(f"time_swap = {self.Dt}")
      self.measure = measure_op
      self.qobj = Qobj(self.measure, dims=[[2,2],[2,2]])
      self.out = self.Hamiltonian_solve_excharge(tf_expectt)
      return self.out

  def sqrt_SWAP(self, ket_0, measure_op, q_obj = [0,1], tf_expectt = True):
      self.q_obj = q_obj
      self.ket_0 = ket_0
      self.Dt = (np.pi/(2*self.J*self.n_swap))
      print(f"time_sqrt_swap = {self.Dt}")
      self.measure = measure_op
      self.qobj = Qobj(self.measure, dims=[[2,2],[2,2]])
      self.out = self.Hamiltonian_solve_excharge(tf_expectt)
      return self.out

  def CNOT(self, ket_0, measure_op, q_obj = [0,1], tf_expectt = False):
      self.q_obj = q_obj
      q_control , q_target = q_obj
      state_1 = self.Ry(np.pi/2, ket_0, q_obj = q_target, tf_expect = False).states[-1]
      state_2 = self.sqrt_SWAP(state_1, [], q_obj = q_obj, tf_expectt = False).states[-1]
      state_3 = self.Rz(np.pi, state_2, q_obj = q_control, tf_expect = False).states[-1]
      state_4 = self.sqrt_SWAP(state_3, [], q_obj= q_obj, tf_expectt = False).states[-1]
      state_5 = self.Rz(-np.pi/2, state_4, q_obj = q_control, tf_expect = False).states[-1]
      state_6 = self.Rz(-np.pi/2, state_5, q_obj = q_target, tf_expect = False).states[-1]
      state_7 = self.Ry(-np.pi/2, state_6, q_obj = q_target, tf_expect = False).states[-1]
      return state_7

  def CZ(self, ket_0, measure_op, q_obj = [0,1], tf_expectt = False):
      self.q_obj = q_obj
      q_control , q_target = q_obj
      state_2 = self.sqrt_SWAP(ket_0, [], q_obj = q_obj, tf_expectt = False).states[-1]
      state_3 = self.Rz(-np.pi, state_2, q_obj = q_control, tf_expect = False).states[-1]
      state_4 = self.sqrt_SWAP(state_3, [], q_obj = q_obj, tf_expectt = False).states[-1]
      state_5 = self.Rz(np.pi/2, state_4, q_obj = q_control, tf_expect = False).states[-1]
      state_6 = self.Rz(-np.pi/2, state_5, q_obj = q_target, tf_expect = False).states[-1]
      return state_6

  def Hamiltonian_solve(self, tf_expect = True):
      # Constantes del Hamiltoniano:
      h0_constant = - (self.h/2) * (self.ω_x)
      h1_constant =   (self.h/2) * (self.ω_z)
      h2_constant =   (self.h/2) * (self.O_x)
      if self.N_qubits > 1:
        apply_qbit_z = []
        apply_qbit_x = []
        for i in range(self.N_qubits):
          if i == self.q_obj:
            apply_qbit_z.append(sigmaz())
            apply_qbit_x.append(sigmax())
          else:
            apply_qbit_z.append(qeye(2))
            apply_qbit_x.append(qeye(2))
      else:
        apply_qbit_z = [sigmaz()]
        apply_qbit_x = [sigmax()]
      H0 = h0_constant * tensor(*apply_qbit_z)
      H1 = h1_constant * tensor(*apply_qbit_z)
      H2 = h2_constant * tensor(*apply_qbit_x)

      # Correccion de desviacion estandar:
      if self.B1 != 0:
        dv = self.noise_std/abs(self.B1)
      else:
        dv = 0
      self.args = { "t_init": 0, "t_final": self.delt_t, "std_noise": dv}
      self.tlist  = np.linspace(0, self.delt_t, self.n_points_pulse_Ri)
      # Hamiltonian
      if self.tf_noise == False:
        H = [H0, [H1, pulse_z], [H2, pulse_x]]
        # Guardar pulso:
        t_actual = self.global_time
        t_final = self.global_time + self.delt_t
        self.pulse_type[self.q_obj].append({
            "Type_pulse": "Unitary",
            "B0": self.B0,
            "B1": self.B1,
            "Delt_t": self.delt_t,
            "t_i": t_actual,
            "t_f": t_final,
            "Noise": ""
        })
        self.global_time += self.delt_t
      else:
        #print("Entre con ruido!")
        noise_x = pulse_x_with_noise(self.tlist, self.args)
        S_x = Cubic_Spline(self.tlist[0], self.tlist[-1], noise_x)
        #noise_z = pulse_z_with_noise(self.tlist, self.args)
        #S_z = Cubic_Spline(self.tlist[0], self.tlist[-1], noise_z)
        H = [H0, [H1, pulse_z], [H2, S_x]]
        # Guardar pulso
        t_actual = self.global_time
        t_final = self.global_time + self.delt_t
        self.pulse_type[self.q_obj].append(
                         {
                          "Type_pulse": "Unitary",
                          "B0": self.B0,
                          "B1": self.B1,
                          "Delt_t": self.delt_t,
                          "t_i": t_actual,
                          "t_f": t_final,
                          "Noise": {
                              "S_z": 0,
                              "S_x": S_x
                          }
                        })
        self.global_time += self.delt_t
      if tf_expect:
        if self.N_qubits > 1:
          apply_qbit_e_ops = []
          for i in range(self.N_qubits):
            if i == self.q_obj:
              apply_qbit_e_ops.append(sigmaz())
            else:
              apply_qbit_e_ops.append(qeye(2))
          e_ops = [tensor(*apply_qbit_e_ops)]
        else:
          e_ops = [sigmax(), sigmay(), sigmaz()]
      else:
        e_ops = []

      if self.tf_quantum_noise:
        a = destroy(2)
        #print("Entro aca")
        T2_star = 1/((1/self.T2) - (1/(2*self.T1)))
        print(T2_star)
        c1 = a/(np.sqrt(self.T1))
        c2 = a.dag()*a*np.sqrt(2/T2_star)
        c_ops = [c1, c2]
      else:
        c_ops = []

      self.output_rwa = mesolve(H, self.ket_0, self.tlist, c_ops, e_ops, self.args)
      return self.output_rwa

  def Hamiltonian_solve_excharge(self, tf_expect = True):
      delt_t = self.Dt
      Si, Sj = self.q_obj
      if self.N_qubits > 1:
        apply_qbit_z = []
        apply_qbit_x = []
        apply_qbit_y = []
        for i in range(self.N_qubits):
          if i == Si or i == Sj:
            apply_qbit_z.append(sigmaz())
            apply_qbit_x.append(sigmax())
            apply_qbit_y.append(sigmay())
          else:
            apply_qbit_z.append(qeye(2))
            apply_qbit_x.append(qeye(2))
            apply_qbit_y.append(qeye(2))

      H = ((self.J * self.h**2)/4)*(tensor(*apply_qbit_x) + tensor(*apply_qbit_y) + tensor(*apply_qbit_z))

      #H = (self.J/4)*(tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()))
      qobj = Qobj(self.measure, dims=[[2,2],[2,2]])
      h_t = [H, pulse_x]

      # Correccion de desviacion estandar:
      if self.B1 != 0:
        dv = self.noise_std/abs(self.B1)
      else:
        dv = 0
      self.args = { "t_init": 0, "t_final": delt_t, "std_noise": dv}
      #self.args = {"t_init":0, "t_final": delt_t, "noise": True}
      if tf_expect:
        e_ops = [qobj]
      else:
        e_ops = []

      if self.tf_quantum_noise:
        a = tensor(destroy(2),destroy(2))
        #print("Entro aca")
        T2_star = 1/((1/self.T2) - (1/(2*self.T1)))
        print(T2_star)
        c1 = a/(np.sqrt(self.T1))
        c2 = a.dag()*a*np.sqrt(2/T2_star)
        c_ops = [c1, c2]
      else:
        c_ops = []

      self.tlist  = np.linspace(0, delt_t, self.n_points_pulse_2Qbits)
      # Save pulses
      t_actual = self.global_time
      t_final = self.global_time + delt_t
      self.pulse_type[f'I_{self.q_obj[0]}-{self.q_obj[1]}'].append(
                         {
                            "Type_pulse": "Two_Qubits",
                            "Q_bits_target":self.q_obj,
                            "J": self.J,
                            "Delt_t": self.Dt,
                            "t_i": t_actual,
                            "t_f": t_final,
                            "Noise": ""
                          })
      self.global_time += self.Dt
      self.output = mesolve(h_t, self.ket_0, self.tlist, c_ops, e_ops, self.args)
      return self.output

  def plot_expect(self, out, ry_tf = False):
      ## create Bloch sphere instance ##
      if ry_tf == False:
        fig = plt.figure(constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        b=Bloch()
        b.axes = ax1
        b.fig = fig
        ## normalize colors to times in tlist ##
        nrm = mpl.colors.Normalize(0, self.delt_t)
        colors = cm.jet(nrm(self.tlist))
        ## add data points from expectation values ##
        b.add_points([out.expect[0],out.expect[1],out.expect[2]],'m')
        ## customize sphere properties ##
        b.point_color=list(colors)
        b.point_marker=['o']
        b.point_size=[8]
        b.view=[-9,11]
        b.zlpos=[1.1,-1.2]
        b.zlabel=['$\left|0\\right>_{f}$','$\left|1\\right>_{f}$']
        ## plot sphere ##
        b.render()
        ## Add color bar ##
        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=nrm)
        sm.set_array([])  # You need to set a dummy array for the right scaling
        cbar = plt.colorbar(sm, ax = ax1, orientation='vertical', shrink=0.5)
        cbar.set_label('Time [s]')
        plt.show()
      else:
        fig = plt.figure(constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        b=Bloch()
        b.axes = ax1
        b.fig = fig
        ## normalize colors to times in tlist ##
        nrm = mpl.colors.Normalize(0, self.delt_t)
        colors = cm.jet(nrm(self.tlist))
        ## add data points from expectation values ##
        for i in range(len(out)):
          b.add_points([out[i].expect[0], out[i].expect[1], out[i].expect[2]],'m')
        ## customize sphere properties ##
        b.point_color=list(colors)
        b.point_marker=['o']
        b.point_size=[8]
        b.view=[-9,11]
        b.zlpos=[1.1,-1.2]
        b.zlabel=['$\left|0\\right>_{f}$','$\left|1\\right>_{f}$']
        ## plot sphere ##
        b.render()
        ## Add color bar ##
        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=nrm)
        sm.set_array([])  # You need to set a dummy array for the right scaling
        cbar = plt.colorbar(sm, ax = ax1, orientation='vertical', shrink=0.5)
        cbar.set_label('Time [s]')
        plt.show()

  def plot_excharges(self, out, index, ry_tf = False):

      labels_axis = ["X","Y","Z"]
      # Grafica del valor esperado:
      if ry_tf == False:
        plt.figure(figsize=(6, 2))
        # Subplot para el valor esperado
        plt.subplot(1, 1, 1)  # 2 filas, 1 columna, primer subplot
        plt.plot(self.tlist, out.expect[index])
        plt.title(f'Valor Esperado \n Eje {labels_axis[index]}')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor Esperado')
        plt.grid(True)
      else:
        time = list(self.tlist) + list(self.tlist) + list(self.tlist)
        Ntime = len(time)
        tt = np.linspace(0,3e-9,Ntime)
        y = list(out[0].expect[index]) + list(out[1].expect[index]) + list(out[2].expect[index])
        plt.figure(figsize=(6, 2))
        # Subplot para el valor esperado
        plt.subplot(1, 1, 1)  # 2 filas, 1 columna, primer subplot
        plt.plot(tt, y)
        plt.title(f'Valor Esperado \ Eje {labels_axis[index]}')
        plt.xlabel('Tiempo')
        plt.ylabel('Valor Esperado')
        plt.grid(True)
      plt.show()
      print("\n")
      # Grafica del pulso:
      if self.tf_noise == False:
        if self.Dt == -7:
          # Subplot para el pulso
          plt.figure(figsize=(4, 3))
          plt.subplot(2, 1, 1)  # 2 filas, 1 columna, segundo subplot
          plt.plot(self.tlist, self.B0 * pulse_z(self.tlist, self.args))
          plt.title('Pulso B0')
          plt.xlabel('Tiempo')
          plt.ylabel('Amplitud')
          plt.grid(True)
          # Subplot para el pulso
          plt.subplot(2, 1, 2)  # 2 filas, 1 columna, segundo subplot
          plt.plot(self.tlist, self.B1 * pulse_x(self.tlist, self.args))
          plt.title('Pulso B1')
          plt.xlabel('Tiempo')
          plt.ylabel('Amplitud')
          plt.grid(True)
        else:
          plt.figure(figsize=(4, 3))
          plt.subplot(2, 1, 1)  # 2 filas, 1 columna, segundo subplot
          plt.plot(self.tlist, self.J * pulse_z(self.tlist, self.args))
          plt.title('Pulso B0')
          plt.xlabel('Tiempo')
          plt.ylabel('Amplitud')
          plt.grid(True)

      else:
        if self.Dt == -7:
          plt.figure(figsize=(4, 3))
          plt.subplot(2, 1, 1)  # 2 filas, 1 columna, segundo subplot
          plt.plot(self.tlist, self.B0 * pulse_z_with_noise(self.tlist, self.args))
          plt.title('Pulso B0')
          plt.xlabel('Tiempo')
          plt.ylabel('Amplitud')
          plt.grid(True)
          # Subplot para el pulso
          plt.subplot(2, 1, 2)  # 2 filas, 1 columna, segundo subplot
          plt.plot(self.tlist, self.B1 * pulse_x_with_noise(self.tlist, self.args))
          plt.title('Pulso B1')
          plt.xlabel('Tiempo')
          plt.ylabel('Amplitud')
          plt.grid(True)
        else:
          plt.figure(figsize=(4, 3))
          plt.subplot(2, 1, 1)  # 2 filas, 1 columna, segundo subplot
          plt.plot(self.tlist, self.J * pulse_z(self.tlist, self.args))
          plt.title('Pulso B0')
          plt.xlabel('Tiempo')
          plt.ylabel('Amplitud')
          plt.grid(True)
      # Ajustar el espacio entre los subgráficos para evitar solapamiento
      plt.tight_layout()
      # Mostrar el gráfico
      plt.show()