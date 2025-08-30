#include "solver.h"
#include <iostream>
#include <chrono>


//Algorithm
#include "algo/simulate.h"
#include "algo/genetic.h"

using namespace std;
// You can add any helper functions or classes you need here.

/**
 * @brief The main function to implement your search/optimization algorithm.
 * Uses Simulated Annealing with restarts to solve the helicopter routing problem.
 */
Solution solve(const ProblemData& problem) {
    cout << "Starting solver with Simulated Annealing..." << endl;
    
    // Use the simulated annealing algorithm from the algo folder
    Solution solution = solveWithGeneticAlgorithm(problem);
    
    cout << "Solver finished." << endl;
    return solution;
}