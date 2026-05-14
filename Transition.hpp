#pragma once
#include "State.hpp"

struct Transition {
	const State& originalState, nextState;
	size_t action;
	float reward;
	bool finished;
};