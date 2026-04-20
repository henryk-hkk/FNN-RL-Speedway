#pragma once
struct Neuron {
	double value{ 1 }, bias{};
	void activate(double input);

	Neuron(double bias = 0) : bias(bias) {};
};