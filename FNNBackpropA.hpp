#pragma once
#include "FNN.hpp"
#include "Dataset.hpp"
#include <algorithm>

class FNNBackpropA { //Feedforward Neural Networks Backpropagation Agent
public:
	static void trainFNN(std::shared_ptr<FNN> fnn, Dataset& trainingDataset, unsigned epochs, double learningFactor, bool getConsoleOutput = false);
	static void trainFNNStep(std::shared_ptr<FNN> fnn, const std::vector<double>& input, const std::vector<double>& desiredOutput, double learningFactor, bool getConsoleOutput = false);
	static double testFNN(std::shared_ptr<FNN> fnn, Dataset& testDataset);
protected:
	//static std::vector<std::vector<double>> getErrorsMean(FNN& fnn, std::vector<std::vector<std::vector<double>>> errorsArr, size_t inputsPerIteration);
	static std::vector<std::vector<double>> getNeuronErrors(std::shared_ptr<FNN> fnn, const std::vector<double>& desiredOutput);
	static void adjustFNNValues(std::shared_ptr<FNN> fnn, const std::vector<std::vector<double>>& errors, double learningFactor);
};