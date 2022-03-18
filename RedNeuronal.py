import numpy as np
import random

# Implementamos la funcion sigmoide y su derivada

def sigmoide(x):
    return 1/(1+ np.exp(-x))

def derivada_sigmoide(x):
    return x * (1 - x)

# La clase de la red neuronal. 
# Sigue la estructura basica de init, feedforward, backprop y predict

class RedNeuronal():

	#x es la entrada y y es la salida esperada
	def __init__(self,x,y,random=False):


		self.entrada = x
		self.y = y
		self.salida = np.zeros(y.shape) # Se inicializa a 0 para futuras actualizaciones
		self.contar = 0 # Esto ayuda a saber las fases de nuesra inteligencia

        #Inicializamos los datos de forma random
		self.datos1 = np.random.rand(self.entrada.shape[1],self.y.shape[0])
		self.datos2 = np.random.rand(self.y.shape[0],1)
		
	
    #Esta función feedforward sólo recibe la entrada y la hace pasar por todas las capas hasta la de salida.
	#Entonces, si lo elegimos, imprimirá cada paso de la ronda de pruebas actual y su pérdida
	def feedforward(self,msg=False,step=100):
		#En este caso, suponemos que los sesgos de las capas son 0
		self.capa_1 = sigmoide(np.dot(self.entrada,self.datos1))
		self.salida = sigmoide(np.dot(self.capa_1, self.datos2))
		self.contar += 1

		# He configurado un código para ver cuántas pruebas ha hecho y
		# el error actual. Sólo imprimirá cada paso (está configurado a 100 por defecto)
		if msg:
			if(self.contar%100==0):
				print("Error en la ronda de prueba: "+str(self.contar))
				print((self.y - self.salida)**2)


	
    #El método backprop utiliza la regla de la cadena para encontrar la derivada de la función de pérdida con respecto a
	#pesos2 y pesos1, por lo que puede actualizar todos los pesos de los nn
	def backprop(self):
		#Utiliza la función 'derivada_sigmoide' del principio
		d_datos2 = np.dot(self.capa_1.T,(2*(self.y-self.salida)*
							derivada_sigmoide(self.salida)))

		d_datos1 = np.dot(self.entrada.T,(np.dot(2*(self.y-self.salida)*
							derivada_sigmoide(self.salida),self.datos2.T)*
							derivada_sigmoide(self.capa_1)))

		# actualizo
		self.datos1 += d_datos1
		self.datos2 += d_datos2

	# Imprime los resultados que predice de una entrada 'entrada' con los valores actuales
	# de self.datos1 y self.datos2


	
    #Para comprobar la predicción simplemente alimentamos la entrada a través del nn
	def predict(self,entrada):
		self.capa_1 = sigmoide(np.dot(entrada,self.datos1))
		self.salida = sigmoide(np.dot(self.capa_1, self.datos2))
		print("Despues de ",self.contar," rondas de prueba, la predicción es:")
		print(self.salida)