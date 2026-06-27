#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>

constexpr double RELU_LEAKAGE_RATE = 0.05;

static std::mt19937 randomEngine(std::random_device{}());

namespace mfuncs { //Mathematical Functions
	//get max value
	double max(double a, double b);
	double max(const std::vector<double>& vec);
	size_t maxIndex(const std::vector<double>& vec);
	//activation functions and their derivatives
	double linear(double x);		// linear function y = x
	double sigmoid(double x);		// sigmoid function
	double dSigmoid(double x);		// sigmoid function derivative
	double ReLU(double x);			// Rectified Linear Unit function
	double dReLU(double x);			// Rectified Linear Unit function derivative
	double lReLU(double x);			// Leaky ReLU
	double dLReLU(double x);		// Leaky ReLU derivative
	double tanh(double x);
	double dTanh(double x);
	std::vector<double> softmax(const std::vector<double>& vec);
	//random number generation
	double getRandomDouble(double a, double b);
	int getRandomInteger(int a, int b);
	std::vector<int> getNRandomUniqueIntegers(int a, int b, size_t n);
	//vector-digit conversion
	std::vector<double> convertDigitToOutputVector(unsigned digit);
	unsigned convertOutputVectorToDigit(const std::vector<double>& outputVector);
};