#pragma once
#include "Transition.hpp"
#include "FNN.hpp"
#include <deque>
#include "FNNBackpropA.hpp"

constexpr size_t SUPERVISED_LEARNING_EPOCH_COUNT = 10;

struct DQNHyperParams { //Deep-Q-Learning parameters
	double gamma;									// Discount - Parameter for calculating target Q values
	double epsilon, epsilonMin, epsilonDecayRate;	// Percentage of randomly chosen actions
	double learningFactor;							// Backpropagation learning factor
	size_t batchSize, actionsCount;					// Memory size parameters for different elements
	size_t memoryCapacity;							// Max memory size
};

class FNNDQNA { //FNN Deep-Q-Learning Agent
private:
	std::shared_ptr<FNN> main;
	std::shared_ptr<FNN> target;
	std::deque<Transition> memory{};

	DQNHyperParams params;

	std::vector<Transition> getMemoryBatch();
	double getTargetValue(const Transition& transition);
	std::vector<double> getCurrentQValues(const std::vector<double>& inputVec);

public:
	FNNDQNA(
		const DQNHyperParams& dqnparams,
		std::shared_ptr<FNN> main, 
		std::shared_ptr<FNN> target
	);
	DQNHyperParams getParams() const;
	void setParams(const DQNHyperParams& params);

	void updateNNs(std::shared_ptr<FNN> main, std::shared_ptr<FNN> target);

	std::shared_ptr<FNN> getTargetNN() const;

	size_t act(const DQNState& state, bool getConsoleOutput = false);
	size_t act(const std::vector<double>& stateVec, bool getConsoleOutput = false);
	void remember(
		const std::vector<double>& state,
		size_t action, 
		double reward,
		const std::vector<double>& nextState, 
		bool finished
	);
	void replayLearn();
	void updateTargetNN();
	void supervisedLearning(const std::vector<DemonstrativeTransition>& demonstrations);
};