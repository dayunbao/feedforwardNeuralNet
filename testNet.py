from Neuron import Neuron
from Network import Network

inputNeurons = Neuron(784, 30) 
hiddenNeurons = Neuron(30, 10) 
outputNeurons = Neuron(10,0)

testNet = Network(inputNeurons, hiddenNeurons, outputNeurons, 0.0001)

#for x in range(len(inputNeurons)):
#	print "Input Neuron #%d: %s\n " % (x, inputNeurons[x])

numInput = 10000 

epoch = 1

#train, backprop, update
testNet.test(epoch, numInput)

