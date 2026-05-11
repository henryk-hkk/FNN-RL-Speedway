#pragma once
struct Neuron {
	double value{ 1 }, bias{};
	void activate(const double& input);

	Neuron(double bias = 0) : bias(bias) {};
};