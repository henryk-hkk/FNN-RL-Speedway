#pragma once
#include "Transition.hpp"
#include "FNN.hpp"
#include <deque>
#include "FNNBackpropA.hpp"

struct DQNHyperParams { //Deep-Q-Learning parameters
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

	DQNHyperParams params;
	double learningFactor = 0.001;

	std::vector<Transition> getMemoryBatch();
	double getTargetValue(const Transition& transition);
	std::vector<double> getCurrentQValues(const std::vector<double>& inputVec);

public:
	FNNDQNA(const DQNHyperParams& dqnparams, FNN& main, FNN& target);

	size_t act(const State& state);
	void saveTransition();
	void replayLearn();
	void updateTargetNN();

};