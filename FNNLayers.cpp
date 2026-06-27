#include "FNNLayers.hpp"

FNNInputLayer::FNNInputLayer(size_t neuronCount, AFunc func) : FNNLayer(neuronCount, func) {
	neurons.reserve(neuronCount);
	for (size_t i = 0; i < neuronCount; i++)
		neurons.push_back(std::make_unique<Neuron>());
}
void FNNLayer::init() {
	neurons.reserve(neuronCount);

	for (size_t i = 0; i < neuronCount; i++) {
		neurons.push_back(std::make_unique<Neuron>(0.0));
	}

	weights.resize(neuronCount);

	if (previous == nullptr || previous->neuronCount == 0) return; // to be safe with input layer

	double std_dev = std::sqrt(2.0 / previous->neuronCount); // standard deviation

	std::normal_distribution<double> dist(0.0, std_dev);

	for (size_t i = 0; i < neuronCount; i++) {
		weights[i] = std::vector<double>(previous->neuronCount);
		for (size_t j = 0; j < previous->neuronCount; j++) {
			weights[i][j] = dist(randomEngine);
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