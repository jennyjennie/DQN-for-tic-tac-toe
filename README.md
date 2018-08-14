# DQN-Tic-Tac-Toe
This project aims to use DQN to build unbeatable Tic Tac Toe AI. 

Research interest of the project:
+ Use domain knowledge to make DQN learn faster

We adpot two ways to make DQN learn faster:
1. Add more determinative decisions(the state that decides the winning and losing) in Experience replay
> Allow DQN to reuse the data and train more times
2. Extra reward for two-way winning board 
> Allow DQN to learn and use it to win the game

The code I upload inculdes:
1. the code of Muti Layer Perceptron
2. the code of DQN we use in the progect
3. the model of player1's and 2's DQN when it plays against 10 people

Criteria for DQN being unbeatable:
+ DQN plays tic tac toe against randomly moving program for 10k rounds, and it can't lose a game
+ We also invite 10 people to play against well trained DQN to ensure the DQN is unbeatable


Tic Tac Toe can be seen as a kind of Markov Decision Process, and every step of the game is expressed as (s, a, r ,s'), therefore we use 
DQN to solve the problem.

+ s: the current state of the board

+ a: 9 locations to put pieces

+ r: Reward function R(s'), the reward is only related to the new state

+ s': the state after the opponent puts the piece

In this project, Multi Layer Perceptron is used to approximate Q function, and we use Q(s) instead of Q(s, a) to output all q.

MLP:
+ Input: 18 neurons, input board's vector
+ Hidden: 36 neurons, tanh(x)
+ Output: 9 neurons, f(x), output all Q vaules 
 







