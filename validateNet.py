from Neuron import Neuron
from Network import Network
import random

inputNeurons = Neuron(784, 30)
hiddenNeurons = Neuron(30, 10)
outputNeurons = Neuron(10,0)

testNet = Network(inputNeurons, hiddenNeurons, outputNeurons, 0.0001)

numInput = 10000

epoch = 1

#validate
testNet.validate(epoch, numInput)
