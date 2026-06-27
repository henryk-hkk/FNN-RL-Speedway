#pragma once
#include <vector>

struct DQNState {
	DQNState() = default;
	virtual ~DQNState() = default;
	virtual std::vector<double> serialise() const;		//Returns serialized State data in a form of std::vector<double>
}; 