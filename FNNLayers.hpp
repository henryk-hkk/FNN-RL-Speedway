#pragma once
#include "Neuron.hpp"
#include "AFunc.hpp"
#include <memory>
#include <ctime>

class FNNLayer {
public:
	FNNLayer() = default;
	FNNLayer(size_t neuronCount, AFunc func = AFunc::LINEAR) : neuronCount(neuronCount), activationFunc(func), activations(neuronCount) {}
	FNNLayer(size_t neuronCount, FNNLayer* prev, AFunc func = AFunc::LINEAR) : neuronCount(neuronCount), previous(prev), activationFunc(func), activations(neuronCount) { init(); }

	void init();

	void processInput();
	FNNLayer* previous = nullptr, * next = nullptr;
	size_t neuronCount = 0;
	std::vector<std::vector<double>> weights;//weights[i][j] -> i-index of current neuron, j-indexes of previous layer's neurons
	std::vector<std::unique_ptr<Neuron>> neurons;
	std::vector<double> activations;
	AFunc activationFunc;
	virtual ~FNNLayer() = default;
};

struct FNNInputLayer : FNNLayer {
	FNNInputLayer(size_t neuronCount, AFunc func = AFunc::LINEAR);
	void processInput(std::vector<double> inputArr);
};

struct FNNHiddenLayer : FNNLayer {
	using FNNLayer::FNNLayer;
};

struct FNNOutputLayer : FNNLayer {
	using FNNLayer::FNNLayer;
};