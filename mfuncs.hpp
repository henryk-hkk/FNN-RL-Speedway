#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>

namespace mfuncs { //Mathematical Functions
	//get max value
	double max(double a, double b);
	double max(const std::vector<double>& vec);
	//activation functions and their derivatives
	double linear(double x);
	double sigmoid(double x);
	double dSigmoid(double x);
	double ReLU(double x);
	double dReLU(double x);
	double tanh(double x);
	double dTanh(double x);
	std::vector<double> softmax(const std::vector<double>& vec);
	//random number generation
	double getRandomDouble(double a, double b);
	int getRandomInteger(int a, int b);
	//vector-digit conversion
	std::vector<double> convertDigitToOutputVector(unsigned digit);
	unsigned convertOutputVectorToDigit(const std::vector<double>& outputVector);
};