import numpy as np
import matplotlib.pyplot as plt
import pylab
import random

#Este ejercicio se hizo teniendo como guia el Bayesian parameter estimation de ComputoCienciasUniandes

#Carga los datos
datos = np.loadtxt("CircuitoRC.txt")
t_obs = datos[:,0]
q_obs = datos[:,1]

def likelihood(q_obs, q_model):
	chi_cuadrado = (1.0/2.0)*sum(((q_obs-q_model)/100)**2)
	return np.exp(-chi_cuadrado)

V_0=10

def modelo(time, R, C):
	return V_0*C*(1-np.exp(-time/(R*C)))

R_walk = np.empty((0))
C_walk = np.empty((0))
l_walk = np.empty((0))

R_walk = np.append(R_walk, np.random.random())
C_walk = np.append(C_walk, np.random.random())

q_init = modelo(t_obs, R_walk[0], C_walk[0])
l_walk = np.append(l_walk, likelihood(q_obs, q_init))

n_iterations = 10000

for i in range(n_iterations):
	R_prime = np.random.normal(R_walk[i], 0.1)
	C_prime = np.random.normal(C_walk[i], 0.1)

	q_init = modelo(t_obs, R_walk[i], C_walk[i])
	q_prime = modelo(t_obs, R_prime, C_prime)

	l_prime = likelihood(q_obs, q_prime)
	l_init = likelihood(q_obs, q_init)

	alpha = l_prime/l_init
	if(alpha>=1.0):
		R_walk = np.append(R_walk, R_prime)
		C_walk = np.append(C_walk, C_prime)
		l_walk = np.append(l_walk, l_prime)
	else:
		beta = np.random.random()
		if(beta<=alpha):
			R_walk = np.append(R_walk, R_prime) 	
                	C_walk = np.append(C_walk, C_prime)
                	l_walk = np.append(l_walk, l_prime)

		else:
			R_walk = np.append(R_walk, R_walk[i])
	        	C_walk = np.append(C_walk, C_walk[i])
                	l_walk = np.append(l_walk, l_init)

max_likelihood_id = np.argmax(l_walk)
best_R = R_walk[max_likelihood_id]
best_C = C_walk[max_likelihood_id]
best_q = modelo(t_obs, best_R, best_C)

plt.figure()
plt.scatter(R_walk, -np.log(l_walk))
plt.xlabel("Resistencia (Ohm)")
plt.ylabel("likelihood")
plt.title("Resistencia en funcion de likelihood")
plt.savefig("likelihood_R.pdf")
plt.close()

plt.figure()
plt.scatter(C_walk, -np.log(l_walk))
plt.xlabel("Capacitancia (F)")
plt.ylabel("likelihood")
plt.title("Capacitancia en funcion de likelihood")
plt.savefig("likelihood_C.pdf")
plt.close()

plt.figure()
plt.hist(R_walk, 20, normed=True)
plt.xlabel("Resistencia (Ohm)")
plt.ylabel("Frecuencia")
plt.title("Muestreo de la resistencia")
plt.savefig("histograma_R.pdf")
plt.close()

plt.figure()
plt.hist(C_walk, 20, normed=True)
plt.xlabel("Capacitancia (F)")
plt.ylabel("Frecuencia")
plt.title("MUestreo de la capacitancia")
plt.savefig("histograma_C.pdf")
plt.close()

R = "El mejor valor de R es  %f" %best_R
C = "El mejor valor de C es %f" %best_C

plt.figure()
plt.scatter(t_obs, q_obs, label="Datos")
plt.plot(t_obs, best_q, c="r", label="Modelo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Carga (c)")
plt.title("Carga en funcion del tiempo")
plt.text(-25,110,R)
plt.text(-25,105,C)
plt.legend(loc =4)
plt.savefig("Carga.pdf")
plt.close()


