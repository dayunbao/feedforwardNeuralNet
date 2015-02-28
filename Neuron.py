import random
import math
import numpy as np

class Neuron:
	
	def __init__(self, numNeurons, numWeights):
		self.inputVals = np.zeros(numNeurons)
		self.weights = np.zeros((numNeurons, numWeights))
		self.target = 0
		self.activateVals = np.zeros(numNeurons)
		self.delta = np.zeros(numNeurons)
		self.biasWeights = np.zeros(numNeurons)
		self.deltaInputVal = np.zeros(numNeurons)
		self.weightChange = np.zeros((numNeurons,numWeights))
		self.biasChange = np.zeros(numNeurons)
	
	'''
	def __str__(self):
		s = str("Weights: %f Bias Weights: %f" % (self.weights, self.biasWeights)
		return s

	def __repr__(self):
		s = str("Input: %f Weights: %s Activation Value %f Target: %d" % (self.inputVal, self.weights,self.activateVal, self.target))
		return s
	'''

	def generateWeights(self):
		for x in np.nditer(self.weights, op_flags=['readwrite']):
			x[...] = random.uniform(-0.03, 0.03)

	def generateBiasWeights(self):
		for x in np.nditer(self.biasWeights, op_flags=['readwrite']):
			x[...] = random.uniform(-0.03, 0.03)

	def sigmoid(self, x):
		if x < -40:
			return -1.0
		else:
			return ((2.0 / (1.0 + math.exp(-1 * (x)))) - 1.0)

	def activate(self, neurons):
		#Manual dot product
		#for x in range(len(self.inputVals)):
			#for y in range(len(neurons.weights)):
				#self.inputVals[x] += (neurons.inputVals[y] * neurons.weights[y,x])
		#self.target = neurons.target
		#Numpy dot product
		self.inputVals = np.dot(neurons.inputVals, neurons.weights)
		self.inputVals += self.biasWeights
		#for x in range(len(self.inputVals)):
			#self.activateVals[x] = self.sigmoid(self.inputVals[x])
		vecSigmoid = np.vectorize(self.sigmoid, otypes=[float])
		self.activateVals = vecSigmoid(self.inputVals)

