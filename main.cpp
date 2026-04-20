#include "FNNBackpropA.hpp"
#include "NNFileManager.hpp"

int main() {
	FNN fnn = NNFileManager::loadFNN("C:/Users/zimor/Documents/NN_91.5Acc");
	Dataset dataset = NNFileManager::getMNISTDatasetFromCSV("C:/Users/zimor/Documents/neural networks/mnist_test.csv");
	std::cout << "\n" << FNNBackpropA::testFNN(fnn, dataset);
}