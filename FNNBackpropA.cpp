#include "FNNBackpropA.hpp"

void FNNBackpropA::trainFNN(std::shared_ptr<FNN> fnn, Dataset& trainingDataset, unsigned epochs, double learningFactor, bool getConsoleOutput) {
	std::vector<std::vector<double>> errorsArr;

	const size_t inputsPerIteration = 1;
	size_t iterations = trainingDataset.inputVector.size() / inputsPerIteration; //Inputs per iteration - deprecated feature.
	for (unsigned epoch{}; epoch < epochs; epoch++) {
		for (size_t iteration = 0; iteration < iterations; iteration++) {
			errorsArr.clear();
			fnn->process(trainingDataset.inputVector[iteration]);
			errorsArr = getNeuronErrors(fnn, trainingDataset.desiredOutputVector[iteration]);

			//because of asymmetrical "shape" of the "errorsArr" vector - each layer can have a different number of neurons, thus a different number of array indexes.
			adjustFNNValues(fnn, errorsArr, learningFactor);
			if (getConsoleOutput) std::cout << "\nTraining iteration: " << (iteration + 1) << ". Iteration cost: " << fnn->getCost(trainingDataset.desiredOutputVector[(iteration + 1) * inputsPerIteration - 1]);
		}
	}
}

void FNNBackpropA::trainFNNStep(std::shared_ptr<FNN> fnn, const std::vector<double>& input, const std::vector<double>& desiredOutput, double learningFactor, bool getConsoleOutput) {
	std::vector<std::vector<double>> errorsArr;
	fnn->process(input);
	errorsArr = getNeuronErrors(fnn, desiredOutput);
	adjustFNNValues(fnn, errorsArr, learningFactor);
	if (getConsoleOutput) std::cout << ". Iteration cost: " << fnn->getCost(desiredOutput) << "\n";
}

double FNNBackpropA::testFNN(std::shared_ptr<FNN> fnn, Dataset& testDataset) {
	size_t correct{};
	for (size_t i{}; i < testDataset.inputVector.size(); i++) {
		fnn->process(testDataset.inputVector[i]);
		if (mfuncs::convertOutputVectorToDigit(fnn->getOutput()) == mfuncs::convertOutputVectorToDigit(testDataset.desiredOutputVector[i])) {
			correct++;
		}
	}
	return static_cast<double>(correct) / static_cast<double>(testDataset.inputVector.size());
}

std::vector<std::vector<double>> FNNBackpropA::getNeuronErrors(std::shared_ptr<FNN> fnn, const std::vector<double>& desiredOutput) {
	std::vector<std::vector<double>> errors;
	Neuron* currentNeuronPtr;
	double currentNeuronValue{}, nextLayerErrorSum{};

	errors.resize(fnn->layers.size());

	for (int layer = (int)fnn->layers.size() - 1; layer > 0; layer--) {//iterating through layers, starting from the last one.
		errors[layer].resize(fnn->layers[layer]->neuronCount);
		for (size_t neuron = 0; neuron < fnn->layers[layer]->neuronCount; neuron++) {//iterating through neurons
			currentNeuronPtr = fnn->layers[layer]->neurons[neuron].get();
			currentNeuronValue = currentNeuronPtr->value;
			if (layer == fnn->layers.size() - 1) { //if the neuron is in the output layer
				double rawError = currentNeuronValue - desiredOutput[neuron];
				errors[layer][neuron] = std::clamp(rawError, -1.0, 1.0); // For RL, to avoid exploding gradient
				//errors[layer][neuron] = rawError;
			}
			else {
				nextLayerErrorSum = 0;
				for (size_t nextLayerNeuron = 0; nextLayerNeuron < fnn->layers[layer + 1]->neuronCount; nextLayerNeuron++)
					nextLayerErrorSum += fnn->layers[layer + 1]->weights[nextLayerNeuron][neuron] * errors[layer + 1][nextLayerNeuron];
				errors[layer][neuron] = nextLayerErrorSum * getDerivative(fnn->layers[layer]->activationFunc)(currentNeuronValue);
			}
		}
	}
	return errors;
}
void FNNBackpropA::adjustFNNValues(std::shared_ptr<FNN> fnn, const std::vector<std::vector<double>>& errors, double learningFactor) {
	for (size_t layer = 1; layer < fnn->layers.size(); layer++) {
		for (size_t neuron = 0; neuron < fnn->layers[layer]->neuronCount; neuron++) {
			fnn->layers[layer]->neurons[neuron]->bias -= learningFactor * errors[layer][neuron];
			for (size_t previousLayerNeuon = 0; previousLayerNeuon < fnn->layers[layer - 1]->neuronCount; previousLayerNeuon++) {
				fnn->layers[layer]->weights[neuron][previousLayerNeuon] -= learningFactor * errors[layer][neuron] * fnn->layers[layer - 1]->neurons[previousLayerNeuon]->value;
			}
		}
	}
}