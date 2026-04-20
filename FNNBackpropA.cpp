#include "FNNBackpropA.hpp"

void FNNBackpropA::trainFNN(FNN& fnn, Dataset& trainingDataset, unsigned epochs, double learningFactor, bool getConsoleOutput) {
	std::vector<std::vector<std::vector<double>>> errorsArr;
	std::vector<std::vector<double>> errorsMean;
	size_t inputsPerIteration = 1;
	size_t iterations = trainingDataset.inputVector.size() / inputsPerIteration;
	for (unsigned epoch{}; epoch < epochs; epoch++) {
		for (size_t iteration = 0; iteration < iterations; iteration++) {
			errorsArr.clear();
			errorsArr.resize(inputsPerIteration);
			for (unsigned i = 0; i < inputsPerIteration; i++) {
				fnn.process(trainingDataset.inputVector[iteration * inputsPerIteration + i]);
				errorsArr[i] = getNeuronErrors(fnn, trainingDataset.desiredOutputVector[iteration * inputsPerIteration + i]);
			}
			if(inputsPerIteration > 1) errorsMean = getErrorsMean(fnn, errorsArr, inputsPerIteration);
			//because of asymmetrical "shape" of the "errorsArr" vector - each layer can have a different number of neurons, thus a different number of array indexes.
			adjustFNNValues(fnn, errorsMean, learningFactor);
			if (getConsoleOutput) std::cout << "\nTraining iteration: " << (iteration + 1) << ". Iteration cost: " << fnn.getCost(trainingDataset.desiredOutputVector[(iteration + 1) * inputsPerIteration - 1]);
		}
	}
}
double FNNBackpropA::testFNN(FNN& fnn, Dataset& testDataset) {
	size_t correct{};
	for (size_t i{}; i < testDataset.inputVector.size(); i++) {
		fnn.process(testDataset.inputVector[i]);
		if (mfuncs::convertOutputVectorToDigit(fnn.getOutput()) == mfuncs::convertOutputVectorToDigit(testDataset.desiredOutputVector[i])) {
			correct++;
		}
	}
	return static_cast<double>(correct) / static_cast<double>(testDataset.inputVector.size());
}
std::vector<std::vector<double>> FNNBackpropA::getErrorsMean(FNN& fnn, std::vector<std::vector<std::vector<double>>> errorsArr, size_t inputsPerIteration) {
	std::vector<std::vector<double>> errorsMean(fnn.layers.size());
	for (size_t layer = 1; layer < fnn.layers.size(); layer++) {
		errorsMean[layer].resize(fnn.layers[layer]->neuronCount, 0.0);
		for (size_t neuron{}; neuron < fnn.layers[layer]->neuronCount; neuron++) {
			for (int i{}; i < inputsPerIteration; i++) {
				errorsMean[layer][neuron] += errorsArr[i][layer][neuron];
			}
			errorsMean[layer][neuron] /= inputsPerIteration;
		}
	}
	return errorsMean;
}
std::vector<std::vector<double>> FNNBackpropA::getNeuronErrors(FNN& fnn, const std::vector<double>& desiredOutput) {
	std::vector<std::vector<double>> errors;
	Neuron* currentNeuronPtr;
	double currentNeuronValue{}, nextLayerErrorSum{};

	errors.resize(fnn.layers.size());

	for (int layer = (int)fnn.layers.size() - 1; layer > 0; layer--) {//iterating through layers, starting from the last one.
		errors[layer].resize(fnn.layers[layer]->neuronCount);
		for (size_t neuron = 0; neuron < fnn.layers[layer]->neuronCount; neuron++) {//iterating through neurons
			currentNeuronPtr = fnn.layers[layer]->neurons[neuron].get();
			currentNeuronValue = currentNeuronPtr->value;
			if (layer == fnn.layers.size() - 1) //if the neuron is in the output layer
				//errors[layer][neuron] = 2 * (currentNeuronValue - desiredOutput[neuron]) * derivative(currentNeuronValue); - for sigmoid activation
				errors[layer][neuron] = currentNeuronValue - desiredOutput[neuron]; //for softmax
			else {
				nextLayerErrorSum = 0;
				for (size_t nextLayerNeuron = 0; nextLayerNeuron < fnn.layers[layer + 1]->neuronCount; nextLayerNeuron++)
					nextLayerErrorSum += fnn.layers[layer + 1]->weights[nextLayerNeuron][neuron] * errors[layer + 1][nextLayerNeuron];
				errors[layer][neuron] = nextLayerErrorSum * getDerivative(fnn.layers[layer]->activationFunc)(currentNeuronValue);
			}
		}
	}
	return errors;
}
void FNNBackpropA::adjustFNNValues(FNN& fnn, std::vector<std::vector<double>> errors, double learningFactor) {
	for (size_t layer = 1; layer < fnn.layers.size(); layer++) {
		for (size_t neuron = 0; neuron < fnn.layers[layer]->neuronCount; neuron++) {
			fnn.layers[layer]->neurons[neuron]->bias -= learningFactor * errors[layer][neuron];
			for (size_t previousLayerNeuon = 0; previousLayerNeuon < fnn.layers[layer - 1]->neuronCount; previousLayerNeuon++) {
				fnn.layers[layer]->weights[neuron][previousLayerNeuon] -= learningFactor * errors[layer][neuron] * fnn.layers[layer - 1]->neurons[previousLayerNeuon]->value;
			}
		}
	}
}