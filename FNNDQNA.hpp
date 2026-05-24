#pragma once
#include "Transition.hpp"
#include "FNN.hpp"
#include <deque>
#include "FNNBackpropA.hpp"

struct DQNHyperParams { //Deep-Q-Learning parameters
	double gamma;									// Parameter for calculating target Q values
	double epsilon, epsilonMin, epsilonDecayRate;	// Percentage of randomly chosen actions
	size_t batchSize, stateSize, actionsCount;		// Memory size parameters for different elements
	size_t memoryCapacity;							// Max memory size
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
	FNNDQNA(
		const DQNHyperParams& dqnparams,
		FNN& main, 
		FNN& target
	);

	size_t act(const State& state);

	void remember(
		const std::vector<double>& state,
		size_t action, 
		double reward,
		const std::vector<double>& nextState, 
		bool finished
	);
	void replayLearn();
	void updateTargetNN();
};