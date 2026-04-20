#pragma once
#include "mfuncs.hpp"

enum class AFunc { //Activation Function
	SIGMOID,
	TANH,
	RELU,
	SOFTMAX,
	LINEAR,
};

static auto getActivationFunction(AFunc func) -> double(*)(double) {
	switch (func) {
	case AFunc::SIGMOID:	return mfuncs::sigmoid;
	case AFunc::TANH:		return mfuncs::tanh;
	case AFunc::RELU:		return mfuncs::ReLU;
	default:				return mfuncs::linear;
	}
}

static auto getDerivative(AFunc func) -> double(*)(double) {
	switch (func) {
	case AFunc::SIGMOID:	return mfuncs::dSigmoid;
	case AFunc::TANH:		return mfuncs::dTanh;
	case AFunc::RELU:		return mfuncs::dReLU;
	default:				return mfuncs::linear;
	}
}