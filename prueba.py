import RedNeuronal
import numpy as np

X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],[1],[1],[0]])

NN = RedNeuronal.RedNeuronal(X,y)
# Exemple de prediccio amb un entrenament insuficient
# Correr codi varis cops i veure com varia.

NN.predict(X)
print("")

for i in range(500):
	NN.feedforward()
	NN.backprop()

NN.predict(X)
print("")

for i in range(500):
	NN.feedforward()
	NN.backprop()

NN.predict(X)
print("")

for i in range(500):
	NN.feedforward()
	NN.backprop()

NN.predict(X)
print("")


for i in range(500):
	NN.feedforward()
	NN.backprop()

NN.predict(X)
print("")

for i in range(10000):
	NN.feedforward()
	NN.backprop()

NN.predict(X)
print("")

print ("Predicción para la entrada [0,1,0]:")
NN.predict(np.array([0,1,0]))
print("")
print ("Predicción para la entrada [0,1,1]:")
NN.predict(np.array([0,1,1]))
print("")
