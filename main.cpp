#include "FNNBackpropA.hpp"
#include "NNFileManager.hpp"

int main() {
	//testing
	//FNN fnn = NNFileManager::loadFNN("C:/Users/zimor/Documents/NN_91.5Acc");
	//Dataset dataset = NNFileManager::getMNISTDatasetFromCSV("C:/Users/zimor/Documents/neural networks/mnist_test.csv");
	//std::cout << "\n" << FNNBackpropA::testFNN(fnn, dataset);

	FNN speedwayFNN;
	speedwayFNN.addInputLayer(16);
	speedwayFNN.addLayer(64);
	speedwayFNN.addLayer(64);
	speedwayFNN.addOutputLayer(2);
	speedwayFNN.setActivationFunction(AFunc::RELU);
	speedwayFNN.setOutputLayerActivationFunction(AFunc::LINEAR);

	NNFileManager::saveFNN(speedwayFNN, "C:/Users/zimor/Documents");
}