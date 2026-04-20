#pragma once
#include "FNNLayers.hpp"
#include <iostream>
class FNN { //Feedforward Neural Network
public:
	std::vector<std::unique_ptr<FNNLayer>> layers;
	void setActivationFunction(AFunc func);
	void setLayerActivationFunction(size_t layer, AFunc func);
	void setOutputLayerActivationFunction(AFunc func);
	void addInputLayer(size_t layerSize);
	void addOutputLayer(size_t layerSize);
	void addLayer(size_t layerSize);
	void process(const std::vector<double>& inputArr);

	std::vector<double> getOutput() const;
	double getCost(const std::vector<double>& desiredOutput) const;
};