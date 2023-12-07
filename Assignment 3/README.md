# Simulated Annealing
 
The theory for Simulated Annealing has been discussed in the lectures. In this assignment we ask you to implement a Simulated Annealing solver for one out of the following problems:

- The Traveling Salesperson Problem, try to solve city configurations presented in "TSP-Configuration.zip".
- Minimum energy configuration of point charges confined to the interior of a circle (see attached paper by Wille and Vennik).
- Reverse-Engineering a Predator-Prey System.

 
In all cases, you should start with a relatively small problem, so that you can experiment with the SA parameters. Next you should try to solve (much) larger problems and try to find out how well your solution scales for these problems. The idea is that you run experiments with the cooling schedule, the length of the Markov chains, the initial temperature, etc. Keep in mind that the optimal parameters for the simulated annealing will depend on the nature and size of your problem. Be sure to run multiple experiments with the same parameter settings to gain error margins and to study the convergence behaviour of your model. Report your findings applying everything relevant you learned in the lectures on Stochastic Simulation. Try to keep your report compact (often one of the biggest challenge), but make sure that it displays your clear understanding on how simulated annealing works.
