#ifndef SIMULATE_H
#define SIMULATE_H

#include "../structures.h"

/**
 * @brief Solves the helicopter routing problem using Simulated Annealing with restarts
 * @param problem The problem data containing villages, helicopters, constraints, etc.
 * @return Solution object containing the plan for all helicopters
 */
Solution solveWithSimulatedAnnealing(const ProblemData& problem);

#endif // SIMULATE_H
