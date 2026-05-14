#pragma once
#include "Transition.hpp"
#include "FNN.hpp"
#include <deque>
#include "FNNBackpropA.hpp"

struct DQNParams { //Deep-Q-Learning parameters
	double gamma;
	double epsilon, epsilonMin, epsilonDecayRate;
	size_t batchSize, stateSize, actionsCount;
	size_t memoryCapacity;
};

class FNNDQNA { //FNN Deep-Q-Learning Agent
private:
	FNN& main;
	FNN& target;
	std::deque<Transition> memory{};

	DQNParams params;

	double rng;

public:
	FNNDQNA(const DQNParams& dqnparams, const FNN& main, const FNN& target);

	size_t act(const State& state);
	void saveTransition(const Transition& transition);
	void replayLearn();
	void updateTargetNN();

};