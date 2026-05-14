#pragma once
#include <vector>

struct State {
	std::vector<double> distances;			//Set of distances from the player to the boards at different angles
	std::vector<double> absAngle;			//Sin and cos of the absolute player angle (angle relative to the screen)
	double forwardVel;						//Forward player velocity
	double sidewaysVel;						//Sideways player velocity (sliding/skidding/drifting)
	double checkpointDistance;				//Distance from the player to the checkpoint
	std::vector<double> checkpointAngle;	//Sin and cos of the angle at which the checkpoint-player distance is the smallest
	bool isHitting;							//Indicator of the player hitting the boards
	std::vector<double> vector() const;		//Returns serialized State data in a form of std::vector<double>
	State(std::vector<double> distances, double absAngle, double forwardVel, double sidewaysVel, double checkpointXY, double playerXY, bool isHitting);
};