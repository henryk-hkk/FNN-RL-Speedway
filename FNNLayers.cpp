#include "FNNLayers.hpp"

FNNInputLayer::FNNInputLayer(size_t neuronCount, AFunc func) : FNNLayer(neuronCount, func) {
	neurons.reserve(neuronCount);
	for (size_t i = 0; i < neuronCount; i++)
		neurons.push_back(std::make_unique<Neuron>());
}
void FNNLayer::init() {
	srand(static_cast<unsigned>(std::time(NULL)));

	neurons.reserve(neuronCount);
	for (size_t i = 0; i < neuronCount; i++)
		neurons.push_back(std::make_unique<Neuron>(mfuncs::getRandomDouble(0, 1)));

	weights.resize(neuronCount);

	for (size_t i = 0; i < neuronCount; i++) {
		weights[i] = std::vector<double>(previous->neuronCount);
		for (size_t j = 0; j < previous->neuronCount; j++) {
			weights[i][j] = mfuncs::getRandomInteger(-10, 10) * 0.1;// set [-1;1]
		}
	}
}
void FNNLayer::processInput() {
	auto prevActivations = previous->activations;

	for (size_t i = 0; i < neuronCount; i++) {//scrolling through the neurons of current layer
		double inputSum{};
		for (size_t j = 0; j < previous->neuronCount; j++)//getting activations and weights of previous layer's neurons
			inputSum += prevActivations[j] * weights[i][j];
		inputSum = getActivationFunction(activationFunc)(inputSum + neurons[i]->bias);
		neurons[i]->activate(inputSum);
		activations[i] = neurons[i]->value;
	}
}
void FNNInputLayer::processInput(std::vector<double> inputArr) {
	for (size_t i = 0; i < neuronCount; i++) {
		neurons[i]->activate(inputArr[i]);
		activations[i] = neurons[i]->value;
	}
}