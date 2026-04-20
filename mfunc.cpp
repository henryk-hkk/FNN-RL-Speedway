#include "mfuncs.hpp"

double mfuncs::max(double a, double b) {
	return ((a > b) ? a : b);
}

double mfuncs::max(const std::vector<double>& vec) {
	double max = vec[0];
	for (size_t i{ 1 }; i < vec.size(); i++)
		max = (vec[i] > max) ? vec[i] : max;
	return max;
}
double mfuncs::linear(double x) {
	return x;
};
double mfuncs::sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}
double mfuncs::dSigmoid(double x) { //x - output of a sigmoid function
	return x * (1 - x);
}
double mfuncs::ReLU(double x) {
	return max(0.0, x);
}
double mfuncs::dReLU(double x) {
	return (x >= 0 ? 1 : 0);
}
double mfuncs::tanh(double x) {
	return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
}
double mfuncs::dTanh(double x) { //x - output of a tanh function
	return 1.0 - std::pow(x, 2);
}
std::vector<double> mfuncs::softmax(const std::vector<double>& vec) {
	double sum{};
	std::vector<double> output(vec.size());
	auto max = mfuncs::max(vec);

	for (size_t i{}; i < vec.size(); i++) {
		output[i] = std::exp(vec[i] - max); // to avoid overflow
		sum += output[i];
	}
	for (size_t i{}; i < vec.size(); i++)
		output[i] /= sum;
	return output;
}

double mfuncs::getRandomDouble(double a, double b) {
	return a + (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * (b - a);
}

int mfuncs::getRandomInteger(int a, int b) {
	return a + (rand() % (b - a + 1));
}

std::vector<double> mfuncs::convertDigitToOutputVector(unsigned digit) {
	std::vector<double> outputVector(10, 0);
	outputVector[digit] = 1;
	return outputVector;
}

unsigned mfuncs::convertOutputVectorToDigit(const std::vector<double>& outputVector) {
	unsigned digit = 0;
	double max = outputVector[0];
	for (size_t i{ 1 }; i < outputVector.size(); i++) {
		if (outputVector[i] > max) {
			max = outputVector[i];
			digit = i;
		}
	}
	return digit;
}