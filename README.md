 # FNN-RL-Speedway

## Overview

This project is a custom-built, low-level implementation of a Feedforward Neural Network written entirely from scratch in modern C++. The network now successfully serves as the core decision-making agent within a custom 2D Speedway simulation, driven and trained using Reinforcement Learning.

## Core Engineering Focus

Instead of relying on high-level ML libraries or vectorized matrix operations, this project is built around a granular, node-based Object-Oriented architecture. This more academic approach allows for a deeper understanding of memory management, data structures, and the raw mathematical mechanics of backpropagation at the individual neuron level:

* **Node-Based OOP Architecture:** Granular design utilizing interconnected structures (e.g., `Layer`, `Neuron`), processing forward passes and error gradients iteratively rather than through matrix abstraction.

* **Modern C++ & Memory Management:** Strict utilization of the STL and smart pointers to manage network topology, ensuring robust lifecycle management and preventing memory leaks during weight updates.

* **Numerical Stability & Algorithm Design:** Custom implementation of backpropagation logic and activation functions (including numerically stable Softmax), handling mathematical edge cases and precision natively in C++.

* **Custom RL Integration:** Seamless binding of the custom FNN to the 2D vehicle physics engine, handling state observation, action selection, and reward propagation entirely from scratch.

## Validation & Training Phases

### Phase 1: Mathematical Validation (MNIST)
To verify the mathematical correctness of the core architecture and the backpropagation algorithm, the model was initially trained and tested on the standard **MNIST dataset**.

* **Network Architecture:** 784 -> 16 -> 16 -> 10
* **Activation Functions:** Sigmoid (Hidden), Softmax (Output)
* **Loss Function:** Mean Squared Error
* **Accuracy Achieved:** 90% (on average)

### Phase 2: Speedway RL Integration
Following core validation, the network was successfully integrated into the custom 2D Speedway environment using Reinforcement Learning (Deep Q-Network / Q-learning principles):

* **State Space:** Dynamically processes environmental variables, such as vehicle velocity, orientation, and distances to track boundaries.
* **Action Space:** Translates network outputs into action choices.
* **Reward Function:** Agent learning is based on previous experiences and state rewards.

## Future Roadmap

With the core FNN and RL environment successfully integrated, future development will focus on optimization and environment expansion:

* Implementing multi-agent simulations to train the AI for competitive racing and overtaking maneuvers.
* Prototyping more advanced RL algorithms (e.g., Policy Gradients or PPO) within the custom framework.
* Introducing multithreading to accelerate training episodes and gradient descent calculations.
