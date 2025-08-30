#include "simulate.h"
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

class SimulatedAnnealing {
private:
    const ProblemData& problem;
    mt19937 rng;  // Random number generator
    uniform_real_distribution<double> uniform_dist;  // For generating random 0-1 values
    chrono::steady_clock::time_point start_time;
    double time_limit_seconds;

    // Algorithm settings - tuned for good performance
    static constexpr double INITIAL_TEMP = 1000.0;    // Starting temperature (high = accept more bad moves)
    static constexpr double COOLING_RATE = 0.999;     // How fast temperature drops each iteration
    static constexpr double MIN_TEMP = 0.01;          // Stop when temperature gets this low
    static constexpr int RESTART_INTERVAL = 1000;     // Restart every 1000 iterations
    static constexpr int SHOULDER_DETECTION_WINDOW = 500;  // Look at last 500 solutions
    static constexpr double SHOULDER_TOLERANCE = 1e-6;     // If solutions don't vary much, we're on a shoulder
    
public:
    SimulatedAnnealing(const ProblemData& prob) 
        : problem(prob), rng(chrono::steady_clock::now().time_since_epoch().count()),
          uniform_dist(0.0, 1.0) {
        time_limit_seconds = problem.time_limit_minutes * 60.0;  // Convert minutes to seconds
        start_time = chrono::steady_clock::now();
    }

    Solution solve() {
        // Start with a greedy solution
        Solution best_solution = generateInitialSolution();
        double best_value = evaluateSolution(best_solution);
        
        // Keep track of current solution we're working with
        Solution current_solution = best_solution;
        double current_value = best_value;
        
        // Simulated annealing variables
        double temperature = INITIAL_TEMP;
        int iteration = 0;
        int iterations_without_improvement = 0;
        int restarts = 0;
        
        // Keep track of recent solution values to detect when we're stuck
        vector<double> recent_values;
        recent_values.reserve(SHOULDER_DETECTION_WINDOW);
        
        cout << "Starting Simulated Annealing with time limit: " << time_limit_seconds << "s" << endl;
        cout << "Initial solution value: " << best_value << endl;
        
        // Main optimization loop - keep going until time runs out or temperature too low
        while (!isTimeUp() && temperature > MIN_TEMP) {
            // Try a small change to current solution
            Solution neighbor = generateConstraintAwareNeighbor(current_solution);
            double neighbor_value = evaluateSolution(neighbor);
            
            // Skip if the neighbor violates constraints
            if (neighbor_value == -numeric_limits<double>::infinity()) {
                iteration++;
                continue;
            }
            
            // Decide whether to accept this neighbor
            if (shouldAccept(current_value, neighbor_value, temperature)) {
                current_solution = neighbor;
                current_value = neighbor_value;
                
                // If it's the best we've seen, remember it
                if (neighbor_value > best_value) {
                    best_solution = neighbor;
                    best_value = neighbor_value;
                    iterations_without_improvement = 0;
                    cout << "New best solution found: " << best_value << " at iteration " << iteration << endl;
                } else {
                    iterations_without_improvement++;
                }
            } else {
                iterations_without_improvement++;
            }
            
            // Remember this solution value for shoulder detection
            recent_values.push_back(current_value);
            if (recent_values.size() > SHOULDER_DETECTION_WINDOW) {
                recent_values.erase(recent_values.begin());  // Remove oldest
            }
            
            // Cool down - makes us pickier about accepting bad moves
            temperature *= COOLING_RATE;
            iteration++;
            
            // Check if we should restart (escape local optimum or shoulder)
            bool should_restart = false;
            
            // Time for regular restart?
            if (iteration % RESTART_INTERVAL == 0) {
                should_restart = true;
            }
            
            // Been too long without improvement? (stuck in local optimum)
            if (iterations_without_improvement > RESTART_INTERVAL / 2) {
                should_restart = true;
            }
            
            // Are we on a shoulder? (solution values barely changing)
            if (recent_values.size() >= SHOULDER_DETECTION_WINDOW && 
                iteration > SHOULDER_DETECTION_WINDOW) {
                double min_val = *min_element(recent_values.begin(), recent_values.end());
                double max_val = *max_element(recent_values.begin(), recent_values.end());
                if (abs(max_val - min_val) < SHOULDER_TOLERANCE) {
                    should_restart = true;
                }
            }
            
            // Do the restart
            if (should_restart && !isTimeUp()) {
                restarts++;
                if (restarts % 5 == 0) {  // Don't spam output
                    cout << "Restart #" << restarts << " at iteration " << iteration 
                         << ", temp: " << temperature << ", best: " << best_value << endl;
                }
                
                // Generate a new starting point (mix of random and best solution)
                Solution restart_solution = generateDiversifiedSolution(best_solution);
                double restart_value = evaluateSolution(restart_solution);
                
                // Use this new starting point if it's good or with some probability
                if (restart_value > current_value || uniform_dist(rng) < 0.4) {
                    current_solution = restart_solution;
                    current_value = restart_value;
                }
                
                // Reset temperature but make it a bit lower each restart
                temperature = INITIAL_TEMP * max(0.1, 1.0 - (double)restarts * 0.1);
                iterations_without_improvement = 0;
                recent_values.clear();
            }
            
            // Show progress occasionally
            if (iteration % 10000 == 0) {
                double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start_time).count();
                cout << "Iteration " << iteration << ", Temp: " << temperature 
                     << ", Current: " << current_value << ", Best: " << best_value 
                     << ", Time: " << elapsed << "s" << endl;
            }
        }
        
        // Final report
        double final_elapsed = chrono::duration<double>(chrono::steady_clock::now() - start_time).count();
        cout << "Final best solution value: " << best_value << " after " << iteration 
             << " iterations and " << restarts << " restarts" << endl;
        cout << "Total time used: " << final_elapsed << "s out of " << time_limit_seconds << "s" << endl;
        
        return best_solution;
    }

private:
    // Check if we're running out of time (use 99% to be safe)
    bool isTimeUp() {
        auto current_time = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(current_time - start_time).count();
        return elapsed >= time_limit_seconds * 0.99;
    }
    
    // Simulated annealing acceptance rule
    bool shouldAccept(double current_value, double neighbor_value, double temperature) {
        if (neighbor_value > current_value) {
            return true; // Always take better solutions
        }
        
        // For worse solutions, accept with probability based on how bad and temperature
        double delta = neighbor_value - current_value;  // This will be negative
        double probability = exp(delta / temperature);  // Higher temp = more likely to accept bad moves
        return uniform_dist(rng) < probability;
    }
    
    // Calculate the objective value of a solution (same logic as format checker)
    double evaluateSolution(const Solution& solution) {
        double total_value = 0.0;
        double total_cost = 0.0;
        
        // Keep track of what we've already delivered to each village
        vector<double> food_delivered(problem.villages.size() + 1, 0.0);
        vector<double> other_delivered(problem.villages.size() + 1, 0.0);
        
        // Go through each helicopter's plan
        for (const auto& helicopter_plan : solution) {
            const Helicopter& helicopter = problem.helicopters[helicopter_plan.helicopter_id - 1];
            Point home_city = problem.cities[helicopter.home_city_id - 1];
            
            double helicopter_total_distance = 0.0;
            
            // Go through each trip
            for (const auto& trip : helicopter_plan.trips) {
                // Calculate how far this trip travels
                double trip_distance = 0.0;
                Point current_pos = home_city;
                
                for (const auto& drop : trip.drops) {
                    const Village& village = problem.villages[drop.village_id - 1];
                    trip_distance += distance(current_pos, village.coords);
                    current_pos = village.coords;
                }
                trip_distance += distance(current_pos, home_city); // Return home
                
                helicopter_total_distance += trip_distance;
                
                // Check if this trip violates constraints
                if (trip_distance > helicopter.distance_capacity) {
                    return -numeric_limits<double>::infinity(); // Invalid!
                }
                
                // Check weight constraint
                double trip_weight = trip.dry_food_pickup * problem.packages[0].weight +
                                   trip.perishable_food_pickup * problem.packages[1].weight +
                                   trip.other_supplies_pickup * problem.packages[2].weight;
                
                if (trip_weight > helicopter.weight_capacity) {
                    return -numeric_limits<double>::infinity(); // Invalid!
                }
                
                // Check if we're dropping more than we picked up
                int total_d_dropped = 0, total_p_dropped = 0, total_o_dropped = 0;
                for (const auto& drop : trip.drops) {
                    total_d_dropped += drop.dry_food;
                    total_p_dropped += drop.perishable_food;
                    total_o_dropped += drop.other_supplies;
                }
                
                if (total_d_dropped > trip.dry_food_pickup || 
                    total_p_dropped > trip.perishable_food_pickup || 
                    total_o_dropped > trip.other_supplies_pickup) {
                    return -numeric_limits<double>::infinity(); // Invalid!
                }
                
                // Add trip cost (fixed cost + distance cost)
                if (!trip.drops.empty()) {
                    total_cost += helicopter.fixed_cost + helicopter.alpha * trip_distance;
                }
                
                // Calculate value gained from each drop
                for (const auto& drop : trip.drops) {
                    const Village& village = problem.villages[drop.village_id - 1];
                    int village_id = drop.village_id;
                    
                    // Food value is capped at 9 per person - can't get value for excess food
                    double max_food_needed = village.population * 9.0;
                    double food_room_left = max(0.0, max_food_needed - food_delivered[village_id]);
                    double food_in_this_drop = drop.dry_food + drop.perishable_food;
                    double effective_food_this_drop = min(food_in_this_drop, food_room_left);
                    
                    // Perishable food is worth more, so count it first
                    double effective_vp = min((double)drop.perishable_food, effective_food_this_drop);
                    double value_from_p = effective_vp * problem.packages[1].value;
                    double remaining_effective_food = effective_food_this_drop - effective_vp;
                    double effective_vd = min((double)drop.dry_food, remaining_effective_food);
                    double value_from_d = effective_vd * problem.packages[0].value;
                    
                    total_value += value_from_p + value_from_d;
                    
                    // Other supplies are capped at 1 per person
                    double max_other_needed = village.population * 1.0;
                    double other_room_left = max(0.0, max_other_needed - other_delivered[village_id]);
                    double effective_vo = min((double)drop.other_supplies, other_room_left);
                    total_value += effective_vo * problem.packages[2].value;
                    
                    // Update what we've delivered so far
                    food_delivered[village_id] += food_in_this_drop;
                    other_delivered[village_id] += drop.other_supplies;
                }
            }
            
            // Check total distance constraint
            if (helicopter_total_distance > problem.d_max) {
                return -numeric_limits<double>::infinity(); // Invalid!
            }
        }
        
        return total_value - total_cost;  // Final objective: value minus cost
    }
    
    // Create a good starting solution using greedy approach
    Solution generateInitialSolution() {
        Solution solution;
        
        // For each helicopter, figure out which villages are profitable to visit
        vector<vector<pair<double, int>>> helicopter_village_scores(problem.helicopters.size());
        
        for (size_t h = 0; h < problem.helicopters.size(); h++) {
            const Helicopter& helicopter = problem.helicopters[h];
            Point home_city = problem.cities[helicopter.home_city_id - 1];
            
            // Score each village based on value per distance
            for (size_t v = 0; v < problem.villages.size(); v++) {
                const Village& village = problem.villages[v];
                double dist = distance(home_city, village.coords);
                double round_trip_dist = 2.0 * dist;
                
                // Can this helicopter even reach this village?
                if (round_trip_dist <= helicopter.distance_capacity) {
                    // Estimate how much value we could get
                    double potential_value = min(9 * village.population, 
                        (int)(helicopter.weight_capacity / min({problem.packages[0].weight, 
                                                               problem.packages[1].weight}))) * problem.packages[1].value;
                    potential_value += min(village.population, 
                        (int)(helicopter.weight_capacity / problem.packages[2].weight)) * problem.packages[2].value;
                    
                    double trip_cost = helicopter.fixed_cost + helicopter.alpha * round_trip_dist;
                    double score = (potential_value - trip_cost) / round_trip_dist;  // Value per distance
                    
                    helicopter_village_scores[h].push_back({score, village.id});
                }
            }
            
            // Sort villages by score (best first)
            sort(helicopter_village_scores[h].begin(), helicopter_village_scores[h].end(), 
                 greater<pair<double, int>>());
        }
        
        // Create a plan for each helicopter
        for (size_t h = 0; h < problem.helicopters.size(); h++) {
            HelicopterPlan plan;
            plan.helicopter_id = (int)h + 1;
            
            generateImprovedGreedyPlan(problem.helicopters[h], helicopter_village_scores[h], plan);
            solution.push_back(plan);
        }
        
        return solution;
    }
    
    // Create trips for a helicopter using greedy village selection
    void generateImprovedGreedyPlan(const Helicopter& helicopter, 
                                  const vector<pair<double, int>>& village_scores,
                                  HelicopterPlan& plan) {
        Point home_city = problem.cities[helicopter.home_city_id - 1];
        double total_distance_used = 0.0;
        vector<bool> village_visited(problem.villages.size() + 1, false);
        
        // Visit villages in order of profitability
        for (const auto& [score, village_id] : village_scores) {
            if (score <= 0 || village_visited[village_id]) continue;
            
            const Village& village = problem.villages[village_id - 1];
            double round_trip_dist = 2.0 * distance(home_city, village.coords);
            
            // Check if we can afford this trip
            if (total_distance_used + round_trip_dist > problem.d_max) continue;
            if (round_trip_dist > helicopter.distance_capacity) continue;
            
            // Create a trip to this village
            Trip trip = createOptimalTripForVillage(helicopter, village);
            
            if (trip.drops.empty()) continue;
            
            plan.trips.push_back(trip);
            village_visited[village_id] = true;
            total_distance_used += round_trip_dist;
        }
    }
    
    // Figure out the best package mix for a single village
    Trip createOptimalTripForVillage(const Helicopter& helicopter, const Village& village) {
        Trip trip;
        
        // This village needs 9 food per person and 1 other supply per person
        int food_needed = 9 * village.population;
        int other_needed = village.population;
        
        // Package weights
        double max_weight = helicopter.weight_capacity;
        double perishable_weight = problem.packages[1].weight;
        double dry_weight = problem.packages[0].weight;
        double other_weight = problem.packages[2].weight;
        
        // Allocate other supplies first (they're required)
        int max_other = min(other_needed, (int)(max_weight / other_weight));
        double remaining_weight = max_weight - max_other * other_weight;
        
        // Then perishable food (it's worth more)
        int max_perishable = min(food_needed, (int)(remaining_weight / perishable_weight));
        remaining_weight -= max_perishable * perishable_weight;
        
        // Fill remaining space with dry food
        int max_dry = min(max(0, food_needed - max_perishable), (int)(remaining_weight / dry_weight));
        
        // If we can't carry anything useful, skip this trip
        if (max_perishable + max_dry + max_other == 0) {
            return trip; // Empty trip
        }
        
        trip.perishable_food_pickup = max_perishable;
        trip.dry_food_pickup = max_dry;
        trip.other_supplies_pickup = max_other;
        
        // Drop everything at this village
        Drop drop;
        drop.village_id = village.id;
        drop.perishable_food = max_perishable;
        drop.dry_food = max_dry;
        drop.other_supplies = max_other;
        
        trip.drops.push_back(drop);
        return trip;
    }
    
    // Generate a neighbor solution by making a small change
    Solution generateConstraintAwareNeighbor(const Solution& current) {
        Solution neighbor = current;
        
        // Try different types of moves until we get a valid one
        int max_attempts = 5;
        for (int attempt = 0; attempt < max_attempts; attempt++) {
            neighbor = current;
            
            double rand_val = uniform_dist(rng);
            
            // Pick a random type of move
            if (rand_val < 0.25) {
                swapVillagesBetweenHelicopters(neighbor);  // Move a village from one helicopter to another
            } else if (rand_val < 0.5) {
                optimizePackageAllocation(neighbor);       // Adjust how many packages we take
            } else if (rand_val < 0.7) {
                reorderVillagesInTrip(neighbor);           // Change the order we visit villages
            } else if (rand_val < 0.85) {
                mergeOrSplitTrips(neighbor);               // Combine two trips or split one
            } else {
                addBestPossibleTrip(neighbor);             // Try to add a new profitable trip
            }
            
            // Check if this neighbor is valid
            if (isFeasibleSolution(neighbor)) {
                return neighbor;
            }
        }
        
        return current; // If we can't make a valid change, keep current solution
    }
    
    // Move a village drop from one helicopter to another
    void swapVillagesBetweenHelicopters(Solution& solution) {
        if (solution.size() < 2) return;
        
        // Pick two different helicopters
        int h1 = rng() % solution.size();
        int h2 = rng() % solution.size();
        while (h1 == h2 && solution.size() > 1) {
            h2 = rng() % solution.size();
        }
        
        if (solution[h1].trips.empty() || solution[h2].trips.empty()) return;
        
        // Pick random trips from each helicopter
        int t1 = rng() % solution[h1].trips.size();
        int t2 = rng() % solution[h2].trips.size();
        
        if (solution[h1].trips[t1].drops.empty() || solution[h2].trips[t2].drops.empty()) return;
        
        // Swap a village drop between the trips
        int d1 = rng() % solution[h1].trips[t1].drops.size();
        int d2 = rng() % solution[h2].trips[t2].drops.size();
        
        swap(solution[h1].trips[t1].drops[d1], solution[h2].trips[t2].drops[d2]);
    }
    
    // Recalculate optimal package amounts for a trip
    void optimizePackageAllocation(Solution& solution) {
        if (solution.empty()) return;
        
        int h = rng() % solution.size();
        if (solution[h].trips.empty()) return;
        
        int t = rng() % solution[h].trips.size();
        Trip& trip = solution[h].trips[t];
        
        if (trip.drops.empty()) return;
        
        const Helicopter& helicopter = problem.helicopters[h];
        
        // Calculate total needs for all villages in this trip
        int total_food_needed = 0, total_other_needed = 0;
        for (const auto& drop : trip.drops) {
            const Village& village = problem.villages[drop.village_id - 1];
            total_food_needed += 9 * village.population;
            total_other_needed += village.population;
        }
        
        // Allocate optimally within weight limit
        double max_weight = helicopter.weight_capacity;
        int optimal_other = min(total_other_needed, (int)(max_weight / problem.packages[2].weight));
        double remaining_weight = max_weight - optimal_other * problem.packages[2].weight;
        
        int optimal_perishable = min(total_food_needed, (int)(remaining_weight / problem.packages[1].weight));
        remaining_weight -= optimal_perishable * problem.packages[1].weight;
        
        int optimal_dry = min(max(0, total_food_needed - optimal_perishable), 
                             (int)(remaining_weight / problem.packages[0].weight));
        
        trip.perishable_food_pickup = optimal_perishable;
        trip.dry_food_pickup = optimal_dry;
        trip.other_supplies_pickup = optimal_other;
        
        // Spread these packages across all drops in the trip
        redistributePackagesToDrops(trip);
    }
    
    // Change the order we visit villages in a trip (might reduce distance)
    void reorderVillagesInTrip(Solution& solution) {
        if (solution.empty()) return;
        
        int h = rng() % solution.size();
        if (solution[h].trips.empty()) return;
        
        int t = rng() % solution[h].trips.size();
        Trip& trip = solution[h].trips[t];
        
        if (trip.drops.size() < 2) return;
        
        // Randomly shuffle the villages in this trip
        shuffle(trip.drops.begin(), trip.drops.end(), rng);
    }
    
    // Try to combine two trips into one (saves fixed cost)
    void mergeOrSplitTrips(Solution& solution) {
        if (solution.empty()) return;
        
        int h = rng() % solution.size();
        if (solution[h].trips.size() < 2) return;
        
        const Helicopter& helicopter = problem.helicopters[h];
        
        // Pick two trips to merge
        int t1 = rng() % solution[h].trips.size();
        int t2 = rng() % solution[h].trips.size();
        while (t1 == t2 && solution[h].trips.size() > 1) {
            t2 = rng() % solution[h].trips.size();
        }
        
        // Combine all drops from both trips
        Trip merged_trip;
        merged_trip.drops = solution[h].trips[t1].drops;
        merged_trip.drops.insert(merged_trip.drops.end(), 
                                solution[h].trips[t2].drops.begin(), 
                                solution[h].trips[t2].drops.end());
        
        // Check if the merged trip fits within distance constraint
        if (calculateTripDistance(helicopter, merged_trip) <= helicopter.distance_capacity) {
            // Figure out how many packages we need for the merged trip
            recalculatePackagePickups(helicopter, merged_trip);
            
            // Replace the two trips with the merged one
            solution[h].trips[t1] = merged_trip;
            solution[h].trips.erase(solution[h].trips.begin() + max(t1, t2));
            solution[h].trips.erase(solution[h].trips.begin() + min(t1, t2));
        }
    }
    
    // Try to add a new profitable trip to a helicopter
    void addBestPossibleTrip(Solution& solution) {
        if (solution.empty()) return;
        
        // Look for a helicopter that has spare distance capacity
        for (size_t h = 0; h < solution.size(); h++) {
            const Helicopter& helicopter = problem.helicopters[h];
            double used_distance = calculateHelicopterTotalDistance(solution[h]);
            
            if (used_distance >= problem.d_max * 0.9) continue;  // Too close to limit
            
            // Find the most profitable village this helicopter could still visit
            int best_village = findBestUnservedVillage(helicopter, used_distance);
            if (best_village != -1) {
                const Village& village = problem.villages[best_village - 1];
                Trip new_trip = createOptimalTripForVillage(helicopter, village);
                
                if (!new_trip.drops.empty()) {
                    solution[h].trips.push_back(new_trip);
                    break;  // Only add one trip per call
                }
            }
        }
    }
    
    // Generate a new solution that's different from current best (for restarts)
    Solution generateDiversifiedSolution(const Solution& best_solution) {
        double diversification_factor = uniform_dist(rng);
        
        if (diversification_factor < 0.3) {
            // 30% chance: Start completely fresh
            return generateInitialSolution();
        } else if (diversification_factor < 0.7) {
            // 40% chance: Mix random solution with best solution
            Solution diversified = generateInitialSolution();
            
            // Randomly copy some good trips from best solution
            size_t min_size = min(diversified.size(), best_solution.size());
            for (size_t h = 0; h < min_size; h++) {
                if (uniform_dist(rng) < 0.5 && !best_solution[h].trips.empty()) {
                    int random_trip = rng() % best_solution[h].trips.size();
                    diversified[h].trips.push_back(best_solution[h].trips[random_trip]);
                }
            }
            
            return diversified;
        } else {
            // 30% chance: Take best solution and mess it up a bit
            Solution perturbed = best_solution;
            
            // Make several random changes
            int num_perturbations = 3 + (rng() % 5); // 3-7 changes
            for (int i = 0; i < num_perturbations; i++) {
                double rand_op = uniform_dist(rng);
                if (rand_op < 0.4) {
                    swapVillagesBetweenHelicopters(perturbed);
                } else if (rand_op < 0.7) {
                    optimizePackageAllocation(perturbed);
                } else {
                    reorderVillagesInTrip(perturbed);
                }
            }
            
            return perturbed;
        }
    }
    
    // Quick check if a solution violates any hard constraints
    bool isFeasibleSolution(const Solution& solution) {
        for (const auto& helicopter_plan : solution) {
            const Helicopter& helicopter = problem.helicopters[helicopter_plan.helicopter_id - 1];
            double total_distance = 0.0;
            
            for (const auto& trip : helicopter_plan.trips) {
                double trip_distance = calculateTripDistance(helicopter, trip);
                if (trip_distance > helicopter.distance_capacity) return false;
                
                double trip_weight = trip.dry_food_pickup * problem.packages[0].weight +
                                   trip.perishable_food_pickup * problem.packages[1].weight +
                                   trip.other_supplies_pickup * problem.packages[2].weight;
                if (trip_weight > helicopter.weight_capacity) return false;
                
                total_distance += trip_distance;
            }
            
            if (total_distance > problem.d_max) return false;
        }
        return true;
    }
    
    // Calculate how far a trip travels
    double calculateTripDistance(const Helicopter& helicopter, const Trip& trip) {
        Point home_city = problem.cities[helicopter.home_city_id - 1];
        Point current_pos = home_city;
        double distance = 0.0;
        
        // Visit each village in the trip
        for (const auto& drop : trip.drops) {
            const Village& village = problem.villages[drop.village_id - 1];
            distance += ::distance(current_pos, village.coords);
            current_pos = village.coords;
        }
        distance += ::distance(current_pos, home_city);  // Return home
        
        return distance;
    }
    
    // Calculate total distance used by a helicopter across all its trips
    double calculateHelicopterTotalDistance(const HelicopterPlan& plan) {
        if (plan.helicopter_id < 1 || (size_t)plan.helicopter_id > problem.helicopters.size()) return 0.0;
        
        const Helicopter& helicopter = problem.helicopters[plan.helicopter_id - 1];
        double total_distance = 0.0;
        
        for (const auto& trip : plan.trips) {
            total_distance += calculateTripDistance(helicopter, trip);
        }
        
        return total_distance;
    }
    
    // Find the most profitable village this helicopter could still visit
    int findBestUnservedVillage(const Helicopter& helicopter, double used_distance) {
        Point home_city = problem.cities[helicopter.home_city_id - 1];
        double best_score = -1;
        int best_village = -1;
        
        for (const auto& village : problem.villages) {
            double round_trip_dist = 2.0 * distance(home_city, village.coords);
            
            // Can we afford this trip?
            if (used_distance + round_trip_dist <= problem.d_max &&
                round_trip_dist <= helicopter.distance_capacity) {
                
                // Estimate profit from this village
                double potential_value = min(9 * village.population, 
                    (int)(helicopter.weight_capacity / problem.packages[1].weight)) * problem.packages[1].value;
                double trip_cost = helicopter.fixed_cost + helicopter.alpha * round_trip_dist;
                double score = potential_value - trip_cost;
                
                if (score > best_score) {
                    best_score = score;
                    best_village = village.id;
                }
            }
        }
        
        return best_village;
    }
    
    // Distribute packages from pickup to all the drops in a trip
    void redistributePackagesToDrops(Trip& trip) {
        if (trip.drops.empty()) return;
        
        int remaining_perishable = trip.perishable_food_pickup;
        int remaining_dry = trip.dry_food_pickup;
        int remaining_other = trip.other_supplies_pickup;
        
        // Give each village what it needs (in order)
        for (auto& drop : trip.drops) {
            const Village& village = problem.villages[drop.village_id - 1];
            
            int food_needed = 9 * village.population;
            int other_needed = village.population;
            
            // Give perishable food first (it's worth more)
            drop.perishable_food = min(remaining_perishable, food_needed);
            remaining_perishable -= drop.perishable_food;
            food_needed -= drop.perishable_food;
            
            // Fill remaining food need with dry food
            drop.dry_food = min(remaining_dry, food_needed);
            remaining_dry -= drop.dry_food;
            
            // Give other supplies
            drop.other_supplies = min(remaining_other, other_needed);
            remaining_other -= drop.other_supplies;
        }
    }
    
    // Calculate how many packages to pick up for a trip with multiple villages
    void recalculatePackagePickups(const Helicopter& helicopter, Trip& trip) {
        int total_food_needed = 0, total_other_needed = 0;
        
        // Add up needs from all villages in this trip
        for (const auto& drop : trip.drops) {
            const Village& village = problem.villages[drop.village_id - 1];
            total_food_needed += 9 * village.population;
            total_other_needed += village.population;
        }
        
        // Allocate packages optimally within weight limit
        double max_weight = helicopter.weight_capacity;
        int optimal_other = min(total_other_needed, (int)(max_weight / problem.packages[2].weight));
        double remaining_weight = max_weight - optimal_other * problem.packages[2].weight;
        
        int optimal_perishable = min(total_food_needed, (int)(remaining_weight / problem.packages[1].weight));
        remaining_weight -= optimal_perishable * problem.packages[1].weight;
        
        int optimal_dry = min(max(0, total_food_needed - optimal_perishable), 
                             (int)(remaining_weight / problem.packages[0].weight));
        
        trip.perishable_food_pickup = optimal_perishable;
        trip.dry_food_pickup = optimal_dry;
        trip.other_supplies_pickup = optimal_other;
        
        // Distribute these packages to the drops
        redistributePackagesToDrops(trip);
    }
};

// Main entry point - create and run the simulated annealing algorithm
Solution solveWithSimulatedAnnealing(const ProblemData& problem) {
    SimulatedAnnealing sa(problem);
    return sa.solve();
}
