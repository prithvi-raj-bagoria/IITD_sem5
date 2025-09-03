#include "genetic.h"
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <set>

class GeneticAlgorithm {
private:
    const ProblemData& problem;
    mt19937 rng;  // Random number generator
    uniform_real_distribution<double> uniform_dist;  // For generating random 0-1 values
    chrono::steady_clock::time_point start_time;
    double time_limit_seconds;

    // Genetic algorithm parameters - tuned for good performance
    static constexpr int POPULATION_SIZE = 50;           // Number of solutions in population
    static constexpr double MUTATION_RATE = 0.15;       // Probability of mutation
    static constexpr double CROSSOVER_RATE = 0.8;       // Probability of crossover
    static constexpr int ELITE_SIZE = 10;               // Number of best solutions to keep
    static constexpr int TOURNAMENT_SIZE = 5;           // Tournament selection size
    static constexpr int MAX_GENERATIONS_NO_IMPROVE = 100;  // Restart after no improvement
    static constexpr int RESTART_INTERVAL = 200;        // Maximum generations before restart
    
public:
    GeneticAlgorithm(const ProblemData& prob) 
        : problem(prob), rng(chrono::steady_clock::now().time_since_epoch().count()),
          uniform_dist(0.0, 1.0) {
        time_limit_seconds = problem.time_limit_minutes * 60.0;  // Convert minutes to seconds
        start_time = chrono::steady_clock::now();
    }

    Solution solve() {
        // Initialize population with diverse solutions
        vector<Solution> population = initializePopulation();
        vector<double> fitness = evaluatePopulation(population);
        
        // Track best solution found so far
        int best_idx = getBestSolutionIndex(fitness);
        Solution best_solution = population[best_idx];
        double best_fitness = fitness[best_idx];
        
        int generation = 0;
        int generations_no_improve = 0;
        int restarts = 0;
        
        cout << "Starting Genetic Algorithm with time limit: " << time_limit_seconds << "s" << endl;
        cout << "Population size: " << POPULATION_SIZE << ", Initial best fitness: " << best_fitness << endl;
        
        // Main evolution loop
        while (!isTimeUp()) {
            generation++;
            
            // Create new generation
            vector<Solution> new_population = createNewGeneration(population, fitness);
            vector<double> new_fitness = evaluatePopulation(new_population);
            
            // Update population
            population = new_population;
            fitness = new_fitness;
            
            // Check for improvement
            int current_best_idx = getBestSolutionIndex(fitness);
            if (fitness[current_best_idx] > best_fitness) {
                best_solution = population[current_best_idx];
                best_fitness = fitness[current_best_idx];
                generations_no_improve = 0;
                cout << "New best solution found: " << best_fitness << " at generation " << generation << endl;
            } else {
                generations_no_improve++;
            }
            
            // Check for restart conditions
            bool should_restart = false;
            
            // Too many generations without improvement?
            if (generations_no_improve > MAX_GENERATIONS_NO_IMPROVE) {
                should_restart = true;
            }
            
            // Regular restart interval?
            if (generation % RESTART_INTERVAL == 0) {
                should_restart = true;
            }
            
            // Population diversity too low?
            if (calculatePopulationDiversity(population) < 0.1) {
                should_restart = true;
            }
            
            // Perform restart
            if (should_restart && !isTimeUp()) {
                restarts++;
                if (restarts % 3 == 0) {  // Don't spam output
                    cout << "Restart #" << restarts << " at generation " << generation 
                         << ", best fitness: " << best_fitness << endl;
                }
                
                // Restart with new diverse population but keep some elite solutions
                population = restartPopulation(best_solution, population, fitness);
                fitness = evaluatePopulation(population);
                generations_no_improve = 0;
            }
            
            // Progress report every 50 generations
            if (generation % 50 == 0) {
                double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start_time).count();
                double avg_fitness = calculateAverageFitness(fitness);
                cout << "Generation " << generation << ", Best: " << best_fitness 
                     << ", Avg: " << avg_fitness << ", Time: " << elapsed << "s" << endl;
            }
        }
        
        // Final report
        double final_elapsed = chrono::duration<double>(chrono::steady_clock::now() - start_time).count();
        cout << "Final best fitness: " << best_fitness << " after " << generation 
             << " generations and " << restarts << " restarts" << endl;
        cout << "Total time used: " << final_elapsed << "s out of " << time_limit_seconds << "s" << endl;
        
        return best_solution;
    }

private:
    // Check if we're running out of time
    bool isTimeUp() {
        auto current_time = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(current_time - start_time).count();
        return elapsed >= time_limit_seconds * 0.99;  // Use 99% of time
    }
    
    // Create initial population with diverse solutions
    vector<Solution> initializePopulation() {
        vector<Solution> population;
        population.reserve(POPULATION_SIZE);
        
        // Create diverse initial solutions using different strategies
        for (int i = 0; i < POPULATION_SIZE; i++) {
            Solution solution;
            
            if (i < POPULATION_SIZE * 0.3) {
                // 30% greedy solutions
                solution = generateGreedySolution();
            } else if (i < POPULATION_SIZE * 0.6) {
                // 30% random solutions
                solution = generateRandomSolution();
            } else {
                // 40% hybrid solutions (mix of greedy and random)
                solution = generateHybridSolution();
            }
            
            population.push_back(solution);
        }
        
        return population;
    }
    
    // Generate a greedy solution (similar to simulated annealing initial solution)
    Solution generateGreedySolution() {
        Solution solution;
        
        // For each helicopter, score villages by value/distance ratio
        vector<vector<pair<double, int>>> helicopter_village_scores(problem.helicopters.size());
        
        for (size_t h = 0; h < problem.helicopters.size(); h++) {
            const Helicopter& helicopter = problem.helicopters[h];
            Point home_city = problem.cities[helicopter.home_city_id - 1];
            
            for (size_t v = 0; v < problem.villages.size(); v++) {
                const Village& village = problem.villages[v];
                double dist = distance(home_city, village.coords);
                double round_trip_dist = 2.0 * dist;
                
                if (round_trip_dist <= helicopter.distance_capacity) {
                    // Calculate potential value
                    double potential_value = min(9 * village.population, 
                        (int)(helicopter.weight_capacity / problem.packages[1].weight)) * problem.packages[1].value;
                    potential_value += min(village.population, 
                        (int)(helicopter.weight_capacity / problem.packages[2].weight)) * problem.packages[2].value;
                    
                    double trip_cost = helicopter.fixed_cost + helicopter.alpha * round_trip_dist;
                    double score = (potential_value - trip_cost) / round_trip_dist;
                    
                    helicopter_village_scores[h].push_back({score, village.id});
                }
            }
            
            sort(helicopter_village_scores[h].begin(), helicopter_village_scores[h].end(), 
                 greater<pair<double, int>>());
        }
        
        // Create helicopter plans
        for (size_t h = 0; h < problem.helicopters.size(); h++) {
            HelicopterPlan plan;
            plan.helicopter_id = (int)h + 1;
            
            createGreedyPlan(problem.helicopters[h], helicopter_village_scores[h], plan);
            solution.push_back(plan);
        }
        
        return solution;
    }
    
    // Generate a completely random solution
    Solution generateRandomSolution() {
        Solution solution;
        
        // For each helicopter, randomly assign villages
        for (size_t h = 0; h < problem.helicopters.size(); h++) {
            const Helicopter& helicopter = problem.helicopters[h];
            HelicopterPlan plan;
            plan.helicopter_id = (int)h + 1;
            
            // Randomly decide number of trips (1-5)
            int num_trips = 1 + (rng() % 5);
            double total_distance = 0.0;
            
            for (int t = 0; t < num_trips && total_distance < problem.d_max * 0.8; t++) {
                Trip trip = createRandomTrip(helicopter);
                if (!trip.drops.empty()) {
                    double trip_dist = calculateTripDistance(helicopter, trip);
                    if (total_distance + trip_dist <= problem.d_max) {
                        plan.trips.push_back(trip);
                        total_distance += trip_dist;
                    }
                }
            }
            
            solution.push_back(plan);
        }
        
        return solution;
    }
    
    // Generate hybrid solution (mix of greedy and random)
    Solution generateHybridSolution() {
        // Start with greedy, then add random elements
        Solution solution = generateGreedySolution();
        
        // Randomly modify some trips
        for (auto& helicopter_plan : solution) {
            if (uniform_dist(rng) < 0.5 && !helicopter_plan.trips.empty()) {
                // Randomly modify a trip
                int trip_idx = rng() % helicopter_plan.trips.size();
                const Helicopter& helicopter = problem.helicopters[helicopter_plan.helicopter_id - 1];
                
                // Maybe add a random village to this trip
                if (uniform_dist(rng) < 0.3) {
                    addRandomVillageToTrip(helicopter, helicopter_plan.trips[trip_idx]);
                }
            }
        }
        
        return solution;
    }
    
    // Create a greedy plan for a helicopter
    void createGreedyPlan(const Helicopter& helicopter, 
                         const vector<pair<double, int>>& village_scores,
                         HelicopterPlan& plan) {
        Point home_city = problem.cities[helicopter.home_city_id - 1];
        double total_distance_used = 0.0;
        set<int> visited_villages;
        
        for (const auto& [score, village_id] : village_scores) {
            if (score <= 0 || visited_villages.count(village_id)) continue;
            
            const Village& village = problem.villages[village_id - 1];
            double round_trip_dist = 2.0 * distance(home_city, village.coords);
            
            if (total_distance_used + round_trip_dist > problem.d_max) continue;
            if (round_trip_dist > helicopter.distance_capacity) continue;
            
            Trip trip = createOptimalTripForVillage(helicopter, village);
            if (!trip.drops.empty()) {
                plan.trips.push_back(trip);
                visited_villages.insert(village_id);
                total_distance_used += round_trip_dist;
            }
        }
    }
    
    // Create random trip for helicopter
    Trip createRandomTrip(const Helicopter& helicopter) {
        Trip trip;
        Point home_city = problem.cities[helicopter.home_city_id - 1];
        
        // Pick 1-2 random villages within range
        vector<int> candidate_villages;
        for (size_t v = 0; v < problem.villages.size(); v++) {
            const Village& village = problem.villages[v];
            double dist = distance(home_city, village.coords);
            if (2.0 * dist <= helicopter.distance_capacity) {
                candidate_villages.push_back(village.id);
            }
        }
        
        if (candidate_villages.empty()) return trip;
        
        shuffle(candidate_villages.begin(), candidate_villages.end(), rng);
        
        // Add 1-2 villages and calculate proper package allocation
        int num_villages = 1 + (rng() % min(2, (int)candidate_villages.size()));
        for (int i = 0; i < num_villages; i++) {
            int village_id = candidate_villages[i];
            Drop drop;
            drop.village_id = village_id;
            // Set drops to 0 initially - will be calculated properly later
            drop.dry_food = 0;
            drop.perishable_food = 0;
            drop.other_supplies = 0;
            trip.drops.push_back(drop);
        }
        
        // Calculate proper package pickups and redistribute to drops
        if (!trip.drops.empty()) {
            recalculatePackagePickups(helicopter, trip);
        }
        
        return trip;
    }
    
    // Evaluate fitness of entire population
    vector<double> evaluatePopulation(const vector<Solution>& population) {
        vector<double> fitness;
        fitness.reserve(population.size());
        
        for (const auto& solution : population) {
            fitness.push_back(evaluateSolution(solution));
        }
        
        return fitness;
    }
    
    // Get index of best solution in population
    int getBestSolutionIndex(const vector<double>& fitness) {
        return max_element(fitness.begin(), fitness.end()) - fitness.begin();
    }
    
    // Create new generation using selection, crossover, and mutation
    vector<Solution> createNewGeneration(const vector<Solution>& population, 
                                       const vector<double>& fitness) {
        vector<Solution> new_population;
        new_population.reserve(POPULATION_SIZE);
        
        // Keep elite solutions (best ones)
        vector<int> elite_indices = getEliteIndices(fitness);
        for (int idx : elite_indices) {
            new_population.push_back(population[idx]);
        }
        
        // Generate rest through crossover and mutation
        while (new_population.size() < POPULATION_SIZE) {
            // Tournament selection for parents
            int parent1_idx = tournamentSelection(fitness);
            int parent2_idx = tournamentSelection(fitness);
            
            // Crossover
            if (uniform_dist(rng) < CROSSOVER_RATE) {
                pair<Solution, Solution> children = crossover(population[parent1_idx], 
                                                           population[parent2_idx]);
                
                // Mutate children
                if (uniform_dist(rng) < MUTATION_RATE) {
                    mutate(children.first);
                }
                if (uniform_dist(rng) < MUTATION_RATE) {
                    mutate(children.second);
                }
                
                new_population.push_back(children.first);
                if (new_population.size() < POPULATION_SIZE) {
                    new_population.push_back(children.second);
                }
            } else {
                // Just mutate parent
                Solution child = population[parent1_idx];
                if (uniform_dist(rng) < MUTATION_RATE) {
                    mutate(child);
                }
                new_population.push_back(child);
            }
        }
        
        return new_population;
    }
    
    // Tournament selection - pick best from random group
    int tournamentSelection(const vector<double>& fitness) {
        int best_idx = rng() % fitness.size();
        double best_fitness = fitness[best_idx];
        
        for (int i = 1; i < TOURNAMENT_SIZE; i++) {
            int candidate_idx = rng() % fitness.size();
            if (fitness[candidate_idx] > best_fitness) {
                best_idx = candidate_idx;
                best_fitness = fitness[candidate_idx];
            }
        }
        
        return best_idx;
    }
    
    // Get indices of elite solutions
    vector<int> getEliteIndices(const vector<double>& fitness) {
        vector<pair<double, int>> fitness_pairs;
        for (size_t i = 0; i < fitness.size(); i++) {
            fitness_pairs.push_back({fitness[i], (int)i});
        }
        
        sort(fitness_pairs.begin(), fitness_pairs.end(), greater<pair<double, int>>());
        
        vector<int> elite_indices;
        for (int i = 0; i < min(ELITE_SIZE, (int)fitness_pairs.size()); i++) {
            elite_indices.push_back(fitness_pairs[i].second);
        }
        
        return elite_indices;
    }
    
    // Crossover two parent solutions to create children
    pair<Solution, Solution> crossover(const Solution& parent1, const Solution& parent2) {
        Solution child1 = parent1;
        Solution child2 = parent2;
        
        // Random helicopter-wise crossover
        for (size_t h = 0; h < min(parent1.size(), parent2.size()); h++) {
            if (uniform_dist(rng) < 0.5) {
                // Swap helicopter plans
                if (h < child1.size() && h < child2.size()) {
                    swap(child1[h], child2[h]);
                }
            }
        }
        
        // Fix any constraint violations introduced by crossover
        fixConstraintViolations(child1);
        fixConstraintViolations(child2);
        
        return {child1, child2};
    }
    
    // Mutate a solution by making small random changes
    void mutate(Solution& solution) {
        if (solution.empty()) return;
        
        double mutation_type = uniform_dist(rng);
        
        if (mutation_type < 0.4) {
            // Swap villages between helicopters (safer mutation)
            swapVillagesBetweenHelicopters(solution);
        } else if (mutation_type < 0.7) {
            // Optimize package allocation (always valid)
            optimizePackageAllocation(solution);
        } else if (mutation_type < 0.85) {
            // Add random trip (with proper validation)
            addRandomTripToSolution(solution);
        } else {
            // Remove random trip (always valid)
            removeRandomTrip(solution);
        }
        
        // Always validate and fix any constraint violations after mutation
        fixConstraintViolations(solution);
    }
    
    // Calculate population diversity (how different solutions are)
    double calculatePopulationDiversity(const vector<Solution>& population) {
        if (population.size() < 2) return 1.0;
        
        double total_diversity = 0.0;
        int comparisons = 0;
        
        // Compare pairs of solutions
        for (size_t i = 0; i < population.size(); i++) {
            for (size_t j = i + 1; j < population.size(); j++) {
                total_diversity += calculateSolutionDistance(population[i], population[j]);
                comparisons++;
            }
        }
        
        return comparisons > 0 ? total_diversity / comparisons : 0.0;
    }
    
    // Calculate "distance" between two solutions
    double calculateSolutionDistance(const Solution& sol1, const Solution& sol2) {
        double distance = 0.0;
        
        for (size_t h = 0; h < min(sol1.size(), sol2.size()); h++) {
            // Count different villages visited
            set<int> villages1, villages2;
            
            for (const auto& trip : sol1[h].trips) {
                for (const auto& drop : trip.drops) {
                    villages1.insert(drop.village_id);
                }
            }
            
            for (const auto& trip : sol2[h].trips) {
                for (const auto& drop : trip.drops) {
                    villages2.insert(drop.village_id);
                }
            }
            
            // Jaccard distance
            set<int> intersection, union_set;
            set_intersection(villages1.begin(), villages1.end(),
                           villages2.begin(), villages2.end(),
                           inserter(intersection, intersection.begin()));
            set_union(villages1.begin(), villages1.end(),
                     villages2.begin(), villages2.end(),
                     inserter(union_set, union_set.begin()));
            
            if (!union_set.empty()) {
                distance += 1.0 - (double)intersection.size() / union_set.size();
            }
        }
        
        return distance;
    }
    
    // Calculate average fitness of population
    double calculateAverageFitness(const vector<double>& fitness) {
        double sum = 0.0;
        for (double f : fitness) {
            if (f != -numeric_limits<double>::infinity()) {
                sum += f;
            }
        }
        return sum / fitness.size();
    }
    
    // Restart population with new diverse solutions
    vector<Solution> restartPopulation(const Solution& best_solution,
                                     const vector<Solution>& current_population,
                                     const vector<double>& fitness) {
        vector<Solution> new_population;
        
        // Keep a few elite solutions
        vector<int> elite_indices = getEliteIndices(fitness);
        for (int i = 0; i < min(5, (int)elite_indices.size()); i++) {
            new_population.push_back(current_population[elite_indices[i]]);
        }
        
        // Generate new diverse solutions
        while (new_population.size() < POPULATION_SIZE) {
            double rand_val = uniform_dist(rng);
            
            if (rand_val < 0.3) {
                // New greedy solution
                new_population.push_back(generateGreedySolution());
            } else if (rand_val < 0.6) {
                // New random solution
                new_population.push_back(generateRandomSolution());
            } else {
                // Mutated version of best solution
                Solution mutated = best_solution;
                for (int i = 0; i < 3; i++) {  // Multiple mutations
                    mutate(mutated);
                }
                new_population.push_back(mutated);
            }
        }
        
        return new_population;
    }
    
    // Helper methods (reuse from simulate.cpp with modifications)
    double evaluateSolution(const Solution& solution) {
        // Same logic as simulate.cpp
        double total_value = 0.0;
        double total_cost = 0.0;
        
        vector<double> food_delivered(problem.villages.size() + 1, 0.0);
        vector<double> other_delivered(problem.villages.size() + 1, 0.0);
        
        for (const auto& helicopter_plan : solution) {
            const Helicopter& helicopter = problem.helicopters[helicopter_plan.helicopter_id - 1];
            
            double helicopter_total_distance = 0.0;
            
            for (const auto& trip : helicopter_plan.trips) {
                double trip_distance = calculateTripDistance(helicopter, trip);
                helicopter_total_distance += trip_distance;
                
                if (trip_distance > helicopter.distance_capacity) {
                    return -numeric_limits<double>::infinity();
                }
                
                double trip_weight = trip.dry_food_pickup * problem.packages[0].weight +
                                   trip.perishable_food_pickup * problem.packages[1].weight +
                                   trip.other_supplies_pickup * problem.packages[2].weight;
                
                if (trip_weight > helicopter.weight_capacity) {
                    return -numeric_limits<double>::infinity();
                }
                
                if (!trip.drops.empty()) {
                    total_cost += helicopter.fixed_cost + helicopter.alpha * trip_distance;
                }
                
                // Calculate value (same logic as simulate.cpp)
                for (const auto& drop : trip.drops) {
                    const Village& village = problem.villages[drop.village_id - 1];
                    int village_id = drop.village_id;
                    
                    double max_food_needed = village.population * 9.0;
                    double food_room_left = max(0.0, max_food_needed - food_delivered[village_id]);
                    double food_in_this_drop = drop.dry_food + drop.perishable_food;
                    double effective_food_this_drop = min(food_in_this_drop, food_room_left);
                    
                    double effective_vp = min((double)drop.perishable_food, effective_food_this_drop);
                    double value_from_p = effective_vp * problem.packages[1].value;
                    double remaining_effective_food = effective_food_this_drop - effective_vp;
                    double effective_vd = min((double)drop.dry_food, remaining_effective_food);
                    double value_from_d = effective_vd * problem.packages[0].value;
                    
                    total_value += value_from_p + value_from_d;
                    
                    double max_other_needed = village.population * 1.0;
                    double other_room_left = max(0.0, max_other_needed - other_delivered[village_id]);
                    double effective_vo = min((double)drop.other_supplies, other_room_left);
                    total_value += effective_vo * problem.packages[2].value;
                    
                    food_delivered[village_id] += food_in_this_drop;
                    other_delivered[village_id] += drop.other_supplies;
                }
            }
            
            if (helicopter_total_distance > problem.d_max) {
                return -numeric_limits<double>::infinity();
            }
        }
        
        return total_value - total_cost;
    }
    
    // More helper methods (simplified versions from simulate.cpp)
    Trip createOptimalTripForVillage(const Helicopter& helicopter, const Village& village) {
        // Same logic as simulate.cpp
        Trip trip;
        
        int food_needed = 9 * village.population;
        int other_needed = village.population;
        
        double max_weight = helicopter.weight_capacity;
        int max_other = min(other_needed, (int)(max_weight / problem.packages[2].weight));
        double remaining_weight = max_weight - max_other * problem.packages[2].weight;
        
        int max_perishable = min(food_needed, (int)(remaining_weight / problem.packages[1].weight));
        remaining_weight -= max_perishable * problem.packages[1].weight;
        
        int max_dry = min(max(0, food_needed - max_perishable), (int)(remaining_weight / problem.packages[0].weight));
        
        if (max_perishable + max_dry + max_other == 0) {
            return trip;
        }
        
        trip.perishable_food_pickup = max_perishable;
        trip.dry_food_pickup = max_dry;
        trip.other_supplies_pickup = max_other;
        
        Drop drop;
        drop.village_id = village.id;
        drop.perishable_food = max_perishable;
        drop.dry_food = max_dry;
        drop.other_supplies = max_other;
        
        trip.drops.push_back(drop);
        return trip;
    }
    
    double calculateTripDistance(const Helicopter& helicopter, const Trip& trip) {
        Point home_city = problem.cities[helicopter.home_city_id - 1];
        Point current_pos = home_city;
        double distance = 0.0;
        
        for (const auto& drop : trip.drops) {
            const Village& village = problem.villages[drop.village_id - 1];
            distance += ::distance(current_pos, village.coords);
            current_pos = village.coords;
        }
        distance += ::distance(current_pos, home_city);
        
        return distance;
    }
    
    void swapVillagesBetweenHelicopters(Solution& solution) {
        if (solution.size() < 2) return;
        
        int h1 = rng() % solution.size();
        int h2 = rng() % solution.size();
        while (h1 == h2 && solution.size() > 1) {
            h2 = rng() % solution.size();
        }
        
        if (solution[h1].trips.empty() || solution[h2].trips.empty()) return;
        
        int t1 = rng() % solution[h1].trips.size();
        int t2 = rng() % solution[h2].trips.size();
        
        if (solution[h1].trips[t1].drops.empty() || solution[h2].trips[t2].drops.empty()) return;
        
        int d1 = rng() % solution[h1].trips[t1].drops.size();
        int d2 = rng() % solution[h2].trips[t2].drops.size();
        
        // Swap village drops
        swap(solution[h1].trips[t1].drops[d1], solution[h2].trips[t2].drops[d2]);
        
        // Recalculate package requirements for both affected trips
        const Helicopter& helicopter1 = problem.helicopters[h1];
        const Helicopter& helicopter2 = problem.helicopters[h2];
        
        recalculatePackagePickups(helicopter1, solution[h1].trips[t1]);
        recalculatePackagePickups(helicopter2, solution[h2].trips[t2]);
    }
    
    void optimizePackageAllocation(Solution& solution) {
        if (solution.empty()) return;
        
        int h = rng() % solution.size();
        if (solution[h].trips.empty()) return;
        
        int t = rng() % solution[h].trips.size();
        Trip& trip = solution[h].trips[t];
        
        if (trip.drops.empty()) return;
        
        const Helicopter& helicopter = problem.helicopters[h];
        // Always recalculate to ensure validity
        recalculatePackagePickups(helicopter, trip);
    }
    
    void addRandomTripToSolution(Solution& solution) {
        if (solution.empty()) return;
        
        int h = rng() % solution.size();
        const Helicopter& helicopter = problem.helicopters[h];
        
        // Check if helicopter has remaining distance capacity
        double used_distance = calculateHelicopterTotalDistance(solution[h]);
        if (used_distance >= problem.d_max * 0.9) return;
        
        Trip new_trip = createRandomTrip(helicopter);
        if (!new_trip.drops.empty()) {
            double trip_distance = calculateTripDistance(helicopter, new_trip);
            if (used_distance + trip_distance <= problem.d_max && 
                trip_distance <= helicopter.distance_capacity) {
                solution[h].trips.push_back(new_trip);
            }
        }
    }
    
    void removeRandomTrip(Solution& solution) {
        if (solution.empty()) return;
        
        int h = rng() % solution.size();
        if (solution[h].trips.empty()) return;
        
        int t = rng() % solution[h].trips.size();
        solution[h].trips.erase(solution[h].trips.begin() + t);
    }
    
    void addRandomVillageToTrip(const Helicopter& helicopter, Trip& trip) {
        if (problem.villages.empty()) return;
        
        int village_idx = rng() % problem.villages.size();
        const Village& village = problem.villages[village_idx];
        
        // Check if this village is already in the trip
        for (const auto& drop : trip.drops) {
            if (drop.village_id == village.id) return;
        }
        
        // Add village drop with zero initial allocation
        Drop new_drop;
        new_drop.village_id = village.id;
        new_drop.dry_food = 0;
        new_drop.perishable_food = 0;
        new_drop.other_supplies = 0;
        
        trip.drops.push_back(new_drop);
        
        // Check if trip is still feasible after adding village
        if (calculateTripDistance(helicopter, trip) <= helicopter.distance_capacity) {
            // Recalculate package allocation for the entire trip
            recalculatePackagePickups(helicopter, trip);
        } else {
            // Remove the village if trip becomes infeasible
            trip.drops.pop_back();
        }
    }
    
    void recalculatePackagePickups(const Helicopter& helicopter, Trip& trip) {
        if (trip.drops.empty()) return;
        
        // Calculate total needs for all villages in this trip
        int total_food_needed = 0, total_other_needed = 0;
        
        for (const auto& drop : trip.drops) {
            const Village& village = problem.villages[drop.village_id - 1];
            total_food_needed += 9 * village.population;
            total_other_needed += village.population;
        }
        
        // Calculate optimal package allocation within weight constraint
        double max_weight = helicopter.weight_capacity;
        int optimal_other = min(total_other_needed, (int)(max_weight / problem.packages[2].weight));
        double remaining_weight = max_weight - optimal_other * problem.packages[2].weight;
        
        int optimal_perishable = min(total_food_needed, (int)(remaining_weight / problem.packages[1].weight));
        remaining_weight -= optimal_perishable * problem.packages[1].weight;
        
        int optimal_dry = min(max(0, total_food_needed - optimal_perishable), 
                             (int)(remaining_weight / problem.packages[0].weight));
        
        // Set pickup amounts
        trip.perishable_food_pickup = optimal_perishable;
        trip.dry_food_pickup = optimal_dry;
        trip.other_supplies_pickup = optimal_other;
        
        // Redistribute packages to drops ensuring we never exceed pickup amounts
        redistributePackagesToDrops(trip);
    }
    
    void redistributePackagesToDrops(Trip& trip) {
        if (trip.drops.empty()) return;
        
        int remaining_perishable = trip.perishable_food_pickup;
        int remaining_dry = trip.dry_food_pickup;
        int remaining_other = trip.other_supplies_pickup;
        
        // Clear all existing drop amounts first
        for (auto& drop : trip.drops) {
            drop.perishable_food = 0;
            drop.dry_food = 0;
            drop.other_supplies = 0;
        }
        
        // Distribute packages proportionally based on village needs
        for (auto& drop : trip.drops) {
            const Village& village = problem.villages[drop.village_id - 1];
            
            int food_needed = 9 * village.population;
            int other_needed = village.population;
            
            // Distribute perishable food first (it's worth more)
            int perishable_for_this_village = min(remaining_perishable, food_needed);
            drop.perishable_food = perishable_for_this_village;
            remaining_perishable -= perishable_for_this_village;
            food_needed -= perishable_for_this_village;
            
            // Fill remaining food need with dry food
            int dry_for_this_village = min(remaining_dry, food_needed);
            drop.dry_food = dry_for_this_village;
            remaining_dry -= dry_for_this_village;
            
            // Give other supplies
            int other_for_this_village = min(remaining_other, other_needed);
            drop.other_supplies = other_for_this_village;
            remaining_other -= other_for_this_village;
        }
    }
    
    // Calculate total distance used by a helicopter
    double calculateHelicopterTotalDistance(const HelicopterPlan& plan) {
        if (plan.helicopter_id < 1 || (size_t)plan.helicopter_id > problem.helicopters.size()) {
            return 0.0;
        }
        
        const Helicopter& helicopter = problem.helicopters[plan.helicopter_id - 1];
        double total_distance = 0.0;
        
        for (const auto& trip : plan.trips) {
            total_distance += calculateTripDistance(helicopter, trip);
        }
        
        return total_distance;
    }
    
    // Fix any constraint violations in a solution
    void fixConstraintViolations(Solution& solution) {
        for (auto& helicopter_plan : solution) {
            const Helicopter& helicopter = problem.helicopters[helicopter_plan.helicopter_id - 1];
            
            // Fix each trip to ensure drops don't exceed pickups
            for (auto& trip : helicopter_plan.trips) {
                // Recalculate everything to ensure consistency
                recalculatePackagePickups(helicopter, trip);
                
                // Remove trips that violate distance constraints
                double trip_distance = calculateTripDistance(helicopter, trip);
                if (trip_distance > helicopter.distance_capacity) {
                    // Mark for removal by clearing drops
                    trip.drops.clear();
                    trip.dry_food_pickup = 0;
                    trip.perishable_food_pickup = 0;
                    trip.other_supplies_pickup = 0;
                }
            }
            
            // Remove empty trips
            helicopter_plan.trips.erase(
                remove_if(helicopter_plan.trips.begin(), helicopter_plan.trips.end(),
                         [](const Trip& trip) { return trip.drops.empty(); }),
                helicopter_plan.trips.end());
            
            // Check total distance constraint and remove trips if necessary
            double total_distance = calculateHelicopterTotalDistance(helicopter_plan);
            while (total_distance > problem.d_max && !helicopter_plan.trips.empty()) {
                // Remove the last trip (least important)
                helicopter_plan.trips.pop_back();
                total_distance = calculateHelicopterTotalDistance(helicopter_plan);
            }
        }
    }
};

// Main entry point for genetic algorithm
Solution solveWithGeneticAlgorithm(const ProblemData& problem) {
    GeneticAlgorithm ga(problem);
    return ga.solve();
}