#include "FNN.hpp"

void FNN::setActivationFunction(AFunc func) {
	for (size_t i{}; i < layers.size() - 1; i++) //without the output layer.
		layers[i]->activationFunc = func;
}
void FNN::setLayerActivationFunction(size_t layer, AFunc func) {
	layers[layer]->activationFunc = func;
}
void FNN::setOutputLayerActivationFunction(AFunc func) {
	layers.back()->activationFunc = func;
}
void FNN::addInputLayer(size_t layerSize) {
	layers.clear();
	layers.resize(1);
	layers[0] = (std::make_unique<FNNInputLayer>(layerSize));
}
void FNN::addOutputLayer(size_t layerSize) {
	FNNLayer* lastLayerPtr = layers.back().get();
	layers.push_back(std::make_unique<FNNOutputLayer>(layerSize, lastLayerPtr));
	lastLayerPtr->next = layers.back().get();
}
void FNN::addLayer(size_t layerSize) {
	FNNLayer* lastLayerPtr = layers.back().get();
	layers.push_back(std::make_unique<FNNHiddenLayer>(layerSize, lastLayerPtr));
	lastLayerPtr->next = layers.back().get();
}

void FNN::cloneWeights(const FNN& other) {
	for (size_t l = 0; l < this->layers.size(); l++) {
		this->layers[l]->weights = other.layers[l]->weights;

		for (size_t n = 0; n < this->layers[l]->neuronCount; n++) {
			this->layers[l]->neurons[n]->bias = other.layers[l]->neurons[n]->bias;
		}
	}
}

void FNN::cloneWeights(std::shared_ptr<FNN> other) {
	for (size_t l = 0; l < this->layers.size(); l++) {
		this->layers[l]->weights = other->layers[l]->weights;

		for (size_t n = 0; n < this->layers[l]->neuronCount; n++) {
			this->layers[l]->neurons[n]->bias = other->layers[l]->neurons[n]->bias;
		}
	}
}

void FNN::process(const std::vector<double>& inputArr) {
	FNNInputLayer* inputLayerPtr = dynamic_cast<FNNInputLayer*>(layers[0].get());
	if (!inputLayerPtr) {
		std::cerr << "NOT AN INPUT LAYER";
		return;
	}
	inputLayerPtr->processInput(inputArr);
	for (size_t i = 1; i < layers.size(); i++) {
		layers[i]->processInput();
	}
	if (layers.back()->activationFunc == AFunc::SOFTMAX) {
		auto output = getOutput();
		output = mfuncs::softmax(output);
		for (size_t i{}; i < layers.back()->neuronCount; i++)
			layers.back()->neurons[i]->value = output[i];
	}
}

std::vector<double> FNN::getOutput() const {
	size_t outputLayerSize = layers.back()->neuronCount;
	std::vector<double> output(outputLayerSize);
	for (size_t i = 0; i < outputLayerSize; i++)
		output[i] = layers.back()->neurons[i]->value;
	return output;
}

double FNN::getCost(const std::vector<double>& desiredOutput) const {
	size_t outputLayerSize = layers.back()->neuronCount;
	std::vector<double> output = getOutput();
	double cost{};
	for (size_t i = 0; i < outputLayerSize; i++) {
		cost += pow((desiredOutput[i] - output[i]), 2);
	}
	return cost;
}