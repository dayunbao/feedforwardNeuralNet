from Neuron import Neuron
from Network import Network
import random

inputNeurons = Neuron(784, 30)
hiddenNeurons = Neuron(30, 10)
outputNeurons = Neuron(10,0)

testNet = Network(inputNeurons, hiddenNeurons, outputNeurons, 0.001)

numInput = 50000 

epoch = 10

#train, backprop, update
testNet.train(epoch, numInput)

epoch2 = 1
numInput2 = 10000

testNet.validate(epoch2, numInput2)

