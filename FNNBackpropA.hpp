#pragma once
#include "FNN.hpp"
#include "Dataset.hpp"
class FNNBackpropA { //Feedforward Neural Networks Backpropagation Agent
public:
	static void trainFNN(FNN& fnn, Dataset& trainingDataset, unsigned epochs, double learningFactor, bool getConsoleOutput = false);
	static double testFNN(FNN& fnn, Dataset& testDataset);
protected:
	static std::vector<std::vector<double>> getErrorsMean(FNN& fnn, std::vector<std::vector<std::vector<double>>> errorsArr, size_t inputsPerIteration);
	static std::vector<std::vector<double>> getNeuronErrors(FNN& fnn, const std::vector<double>& desiredOutput);
	static void adjustFNNValues(FNN& fnn, std::vector<std::vector<double>> errors, double trainingFactor);
};