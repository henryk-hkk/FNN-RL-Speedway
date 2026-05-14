#pragma once
#include "State.hpp"

struct Transition {
	const State& originalState;
	size_t action;
	double reward;
	const State& nextState;
	bool finished;
};