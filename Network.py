from Neuron import Neuron
import math, shelve, cPickle, gzip
import numpy as np

class Network:
	#constructor
	def __init__(self, inputNeurons, hiddenNeurons, outputNeurons, learningRate):
		self.inputNeurons = inputNeurons
		self.hiddenNeurons = hiddenNeurons
		self.outputNeurons = outputNeurons
		self.learningRate = learningRate
		self.actualOutput = np.zeros(10)
		self.targetOutput = np.zeros(10)
		self.trainingVals = np.zeros(50000)
		self.trainingClasses = np.zeros(50000)
		self.validateVals = np.zeros(10000)
		self.validateClasses = np.zeros(10000)
		self.testVals = np.zeros(10000)
		self.testClasses = np.zeros(10000)

	#**********backpropagation functions**********	
	def train(self, epoch, numInput):
		#matched = 0
		self.loadMNIST()
		
		self.inputNeurons.generateWeights()
		self.hiddenNeurons.generateWeights()
		self.hiddenNeurons.generateBiasWeights()
		self.outputNeurons.generateBiasWeights()

		for e in range(epoch):
			#print "Matched %d" % matched
			#if matched == numInput:
				#break

			print "Epoch #%d" % e
			#neuronInfo.write("Epoch #%d\n" % e)

			for n in range(numInput):
				self.getInput(n)

				self.hiddenNeurons.activate(self.inputNeurons)
				self.outputNeurons.activate(self.hiddenNeurons)

				#if self.checkOutput():
					#print "Match"
					#matched += 1
					#break	
				#else:
				self.backpropagate()
				self.update()

				self.clearNeurons(self.inputNeurons)
				self.clearNeurons(self.hiddenNeurons)
				self.clearNeurons(self.outputNeurons)

				#print "Input weights %s" % self.inputNeurons.weights
				#print "Hidden weights %s" % self.hiddenNeurons.weights
		
		self.shelveWeights()

	def backpropagate(self):
		self.deltaForOutput()
		self.changeHiddenWeights()
		self.changeBiasWeights(self.outputNeurons)
		
		self.deltaForHidden()
		self.changeInputWeights()
		self.changeBiasWeights(self.hiddenNeurons)

	def deltaForOutput(self):
		self.targetOutput = self.getTargetArr(self.inputNeurons.target)
		for x in range(len(self.outputNeurons.delta)):
			self.outputNeurons.delta[x] = ((self.targetOutput[x] - self.outputNeurons.activateVals[x]) * self.activateDerivative(self.outputNeurons.activateVals[x]))

	def changeHiddenWeights(self):
		for x in range(len(self.hiddenNeurons.activateVals)):
			for y in range(len(self.outputNeurons.delta)):
				self.hiddenNeurons.weightChange[x,y] = (self.learningRate * self.outputNeurons.delta[y] * self.hiddenNeurons.activateVals[x])
	
	def deltaForHidden(self):
		for x in range(len(self.hiddenNeurons.deltaInputVal)):
			for y in range(len(self.outputNeurons.delta)):
				self.hiddenNeurons.deltaInputVal[x] += (self.outputNeurons.delta[y] * self.hiddenNeurons.weights[x,y])
		for x in range(len(self.hiddenNeurons.delta)):
			self.hiddenNeurons.delta[x] = (self.hiddenNeurons.deltaInputVal[x] * self.activateDerivative(self.hiddenNeurons.activateVals[x]))

	def changeInputWeights(self):
		for x in range(len(self.inputNeurons.inputVals)):
			for y in range(len(self.hiddenNeurons.delta)):
				self.inputNeurons.weightChange[x,y] = (self.learningRate * self.hiddenNeurons.delta[y] * self.inputNeurons.inputVals[x])

	def changeBiasWeights(self, neurons):
		for x in range(len(neurons.delta)):
			neurons.biasChange[x] = self.learningRate * neurons.delta[x]
	
	def updateWeightsAndBiases(self, layer1, layer2):
		#for x in range(len(layer1.weights)):
			#for y in range(len(layer2.weightChange)):
				#layer1.weights[x,y] += layer1.weightChange[x,y]
		layer1.weights += layer1.weightChange
		#for x in range(len(layer2.biasWeights)):
			#layer2.biasWeights[x] += layer2.biasChange[x]
		layer2.biasWeights += layer2.biasChange
	
	def update(self):
		self.updateWeightsAndBiases(self.hiddenNeurons, self.outputNeurons)
		self.updateWeightsAndBiases(self.inputNeurons, self.hiddenNeurons)
	
	def checkOutput(self):
		self.actualOutput = np.zeros(self.actualOutput.shape)
		self.targetOutput = np.zeros(self.targetOutput.shape)
		self.actualOutput = self.outputNeurons.activateVals
		#print "Actual output: %s" % self.actualOutput
		#print "InputNeurons target: %f" % self.inputNeurons.target
		self.targetOutput = self.getTargetArr(self.inputNeurons.target)
		#print "Target output: %s\n" % self.targetOutput
		if np.allclose(self.actualOutput, self.targetOutput):
			return True
		else:
			return False
		'''
		for x in range(len(self.actualOutput)):
			for y in range(len(self.targetOutput)):
				if self.targetOutput == 1.0 and self.actualOutput >= 0.90:
					return True
				elif self.targetOutput == -1.0 and self.actualOutput <= -0.90:
					return True
				else:
					return False
		'''

	def getInput(self, numInput):
		#for x in range(len(self.inputNeurons)):
		self.inputNeurons.target = self.trainingClasses[numInput]
		self.inputNeurons.inputVals = self.trainingVals[numInput]

		#for x in range(len(self.hiddenNeurons)):
		#self.hiddenNeurons.target = self.trainingClasses[numInput]
	
		#for x in range(len(self.outputNeurons)):
		#self.outputNeurons.target = self.trainingClasses[numInput]

	def shelveWeights(self):
		saveWeights = shelve.open("weights")
		
		#numpy array of input neuron weights
		saveWeights["inputWeights"] = self.inputNeurons.weights

		#numpy array of hidden neuron bias weights
		saveWeights["hiddenBiasWeights"] = self.hiddenNeurons.biasWeights
		
		#numpy array of hidden neurons weights
		saveWeights["hiddenWeights"] = self.hiddenNeurons.weights

		#numpy array of output bias weights
		saveWeights["outputBiasWeights"] = self.outputNeurons.biasWeights

		saveWeights.close()

	#**********functions for testing the trained neural net**********
	def test(self, epoch, numInput):
		self.loadMNIST()
		self.getWeights()
	
		for n in range(numInput):
			self.getTestInput(n)

			self.hiddenNeurons.activate(self.inputNeurons)
			self.outputNeurons.activate(self.hiddenNeurons)

			self.checkTestOutput()

			self.clearNeurons(self.inputNeurons)
			self.clearNeurons(self.hiddenNeurons)
			self.clearNeurons(self.outputNeurons)

	def getWeights(self):
		weights = shelve.open("weights")

		self.inputNeurons.weights = weights["inputWeights"]
		self.hiddenNeurons.biasWeights = weights["hiddenBiasWeights"]
		self.hiddenNeurons.weights = weights["hiddenWeights"]
		self.outputNeurons.biasWeights = weights["outputBiasWeights"]

		weights.close()
		
		
	def checkTestOutput(self):
		self.actualOutput = 0.0
		self.targetOutput = 0.0
		self.actualOutput = self.outputNeurons[0].activateVal
		print "Actual output: %f" % self.actualOutput
		self.targetOutput = self.inputNeurons[0].target
		print "Target output: %f" % self.targetOutput
		if self.actualOutput == self.targetOutput:
			return "Match!"
	
	def getTestInput(self, numInput):
		for x in range(len(self.inputNeurons)):
			self.inputNeurons[x].target = self.trainingClasses[numInput]
			self.inputNeurons[x].inputVals = self.trainingVals[numInput][x]

		for x in range(len(self.hiddenNeurons)):
			self.hiddenNeurons[x].target = self.trainingClasses[numInput]
	
		for x in range(len(self.outputNeurons)):
			self.outputNeurons[x].target = self.trainingClasses[numInput]
	
	#**********Validate Functions**********
	def validate(self, epoch, numInput):
		self.loadMNIST()
		self.getWeights()

		for e in range(epoch):
			print "Epoch #%d" % e
			for n in range(numInput):
				self.getValidateInput(n)
			
				self.hiddenNeurons.activate(self.inputNeurons)
				self.outputNeurons.activate(self.hiddenNeurons)
			
				self.validateOutput()

				self.clearNeurons(self.inputNeurons)
				self.clearNeurons(self.hiddenNeurons)
				self.clearNeurons(self.outputNeurons)

	def validateOutput(self):
		errorList = []
		
		self.targetOutput = np.zeros(self.targetOutput.shape)

		self.targetOutput = self.getTargetArr(self.inputNeurons.target)

		for x in range(len(self.actualOutput)):
			error = round(math.pow((self.targetOutput[x] - self.outputNeurons.activateVals[x]), 2), 2)
			errorList.append((self.inputNeurons.target, self.targetOutput[x], self.outputNeurons.activateVals[x], error))
		
		print "Error: %s" % errorList

	def getValidateInput(self, numInput):
		self.inputNeurons.target = self.validateClasses[numInput]
		self.inputNeurons.inputVals = self.validateVals[numInput]


	#**********general purpose functions***********
	def loadMNIST(self):
		mnist = gzip.open("mnist.pkl.gz", "rb")
		#trainSet, validSet, testSet are each tuples
		#1st item in tuple is list of images, 2nd is list of class labels
		trainSet, validSet, testSet = cPickle.load(mnist)

		mnist.close()
   
		#trainingImages = trainSet[0]
		self.trainingVals = np.array(trainSet[0])

		#trainingLabels = trainSet[1]
		self.trainingClasses = np.array(trainSet[1])

		self.validateVals = np.array(validSet[0])

		self.validateClasses = np.array(validSet[1])

		#testImages = testSet[0]
		self.testVals = np.array(testSet[0])

		#testLabels = testSet[1]
		self.testClasses = np.array(testSet[1])

	def clearNeurons(self, neurons):
		neurons.inputVals = np.zeros(neurons.inputVals.shape)
		neurons.activateVals = np.zeros(neurons.activateVals.shape)
		neurons.delta = np.zeros(neurons.delta.shape)
		neurons.deltaInputVal = np.zeros(neurons.deltaInputVal.shape)
		neurons.weightChange = np.zeros(neurons.weightChange.shape)
		neurons.biasChange = np.zeros(neurons.biasChange.shape)

	def getTargetArr(self, targetVal):
		target = np.array([[1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0], [-1.0, 1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0], [-1.0,-1.0, 1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0], [-1.0,-1.0,-1.0, 1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0], [-1.0,-1.0,-1.0,-1.0, 1.0,-1.0,-1.0,-1.0,-1.0,-1.0], [-1.0,-1.0,-1.0,-1.0,-1.0, 1.0,-1.0,-1.0,-1.0,-1.0], [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0, 1.0,-1.0,-1.0,-1.0], [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0, 1.0,-1.0,-1.0], [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0, 1.0,-1.0], [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0, 1.0]])
		return target[targetVal]

	def activateDerivative(self, val):
		derivative = (.5 * ( (1 + val ) * (1 - val) ))
		return derivative

