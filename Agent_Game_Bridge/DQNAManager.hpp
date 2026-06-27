#pragma once
#include "../FNN.hpp"
#include "../FNNDQNA.hpp"
#include "SpeedwayState.hpp"

constexpr size_t HIDDEN_LAYERS_SIZE = 256;
constexpr size_t STATES_TO_PROCESS = 8;

struct DQNAssets {
	std::shared_ptr<FNN> main;
	std::shared_ptr<FNN> target;
	std::shared_ptr<FNNDQNA> dqnAgent;
	
	size_t getAction(const std::shared_ptr<DQNState> state);
	void update(const FNN& fnn, bool clearEpsilon = false);
	void clearEpsilon();
};

class DQNAManager { //Deep-Q-Learning Agent Manager
public:
	static DQNAssets initAgent(); // initializes DQN agent
};