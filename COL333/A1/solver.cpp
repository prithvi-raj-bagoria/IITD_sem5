#include "solver.h"
#include <iostream>
#include <chrono>


//Algorithm
#include "algo/simulate.h"
#include "algo/genetic.h"
#include "algo/greedy.h"

using namespace std;
// You can add any helper functions or classes you need here.

/**
 * @brief The main function to implement your search/optimization algorithm.
 * Uses Simulated Annealing with restarts to solve the helicopter routing problem.
 */
// Solution solve(const ProblemData& problem) {
//     cout << "Starting solver with Greedy approach..." << endl;
    
//     // Use the greedy algorithm from the algo folder
//     Solution solution = solveWithGreedy(problem);
    
//     cout << "Solver finished." << endl;
//     return solution;
// }

/**
 * @file greedy.cpp
 * @brief Enhanced Greedy Heuristic for Disaster Relief Helicopter Routing
 * 
 * ALGORITHM INTUITION & IDEA:
 * ===========================
 * 
 * Core Philosophy: "Act like a smart disaster relief coordinator who dynamically 
 * balances efficiency, urgency, and fairness while maximizing aid delivery under 
 * tight resource constraints."
 * 
 * Key Principles:
 * 1. DYNAMIC PRIORITIZATION: Prioritize helicopters by capability (capacity × range / cost × fuel)
 * 2. VILLAGE URGENCY: Factor in unmet demand and remoteness from cities
 * 3. VALUE/WEIGHT OPTIMIZATION: Greedily pack highest value-per-weight packages
 * 4. MULTI-VILLAGE TRIPS: Bundle multiple villages per trip when possible
 * 5. CONSTRAINT AWARENESS: Strictly enforce weight, distance, and DMax limits
 * 6. ADAPTIVE DECISIONS: Recompute priorities as conditions change
 * 
 * Algorithm Flow:
 * - Initialize tracking systems for demands, deliveries, helicopter states
 * - Build priority queue of helicopters based on efficiency and village urgency
 * - For each helicopter: find best batch of villages to serve in one trip
 * - Use greedy knapsack to optimally pack packages for each village
 * - Apply TSP heuristic to optimize route through villages
 * - Update all tracking systems and recompute priorities
 * - Repeat until no profitable trips remain
 */

#include "solver.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <map>
#include <cmath>
#include <numeric>
#include <random>
#include <chrono>

using namespace std;

// ==================== CONSTANTS AND CONFIGURATION ====================

const int MAX_TRIES = 3;              // Multi-start iterations (reduced for time)
const int MAX_BATCH_SIZE = 3;          // Maximum villages per trip (reduced from 10 to a more practical size)
const double LAMBDA_HELICOPTER = 0.7; // Weight for helicopter efficiency in priority
const double LAMBDA_VILLAGE = 0.3;    // Weight for village urgency in priority

// ==================== HELPER STRUCTURES ====================

struct GlobalState {
    vector<double> unmet_food_demand;   // Remaining food needed per village
    vector<double> unmet_other_demand;  // Remaining other supplies needed per village
    vector<double> helicopter_distances; // Total distance used by each helicopter
    map<int, double> village_urgencies;  // Urgency score per village
    
    GlobalState(const ProblemData& problem) {
        unmet_food_demand.resize(problem.villages.size() + 1);
        unmet_other_demand.resize(problem.villages.size() + 1);
        helicopter_distances.resize(problem.helicopters.size() + 1, 0.0);
        
        // Initialize village demands (9 meals per person, 1 other per person)
        for (const auto& village : problem.villages) {
            unmet_food_demand[village.id] = 9.0 * village.population;
            unmet_other_demand[village.id] = 1.0 * village.population;
        }
        
        updateAllUrgencies(problem);
    }
    
    void updateAllUrgencies(const ProblemData& problem) {
        for (const auto& village : problem.villages) {
            double total_unmet = unmet_food_demand[village.id] + unmet_other_demand[village.id] * 10.0;
            if (total_unmet <= 0) {
                village_urgencies[village.id] = 0.0;
                continue;
            }
            
            // Calculate average distance from all cities (remoteness factor)
            double avg_distance = 0.0;
            for (const auto& city : problem.cities) {
                avg_distance += distance(village.coords, city);
            }
            avg_distance /= problem.cities.size();
            
            // Higher urgency for remote villages with high unmet demand
            village_urgencies[village.id] = total_unmet * 10.0 / (avg_distance + 1.0);
        }
    }
    
    bool hasUnmetDemands() const {
        for (size_t i = 1; i < unmet_food_demand.size(); i++) {
            if (unmet_food_demand[i] > 0 || unmet_other_demand[i] > 0) {
                return true;
            }
        }
        return false;
    }
};

struct HelicopterPriority {
    double priority;
    int helicopter_index;
    
    bool operator<(const HelicopterPriority& other) const {
        return priority < other.priority; // For max-heap
    }
};

struct PackageRatio {
    int package_type; // 0=dry, 1=perishable, 2=other
    double adjusted_ratio;
    
    bool operator<(const PackageRatio& other) const {
        return adjusted_ratio > other.adjusted_ratio; // Sort descending
    }
};

// For clustering nearby villages
struct VillageDistance {
    int village_id;
    double distance;
    
    bool operator<(const VillageDistance& other) const {
        return distance < other.distance; // Sort by increasing distance
    }
};

// ==================== UTILITY FUNCTIONS ====================

/**
 * @brief Calculate total distance for a trip (home -> villages -> home)
 */
double calculateTripDistance(const Trip& trip, const ProblemData& problem, int home_city_id) {
    if (trip.drops.empty()) return 0.0;
    
    double total_distance = 0.0;
    Point current = problem.cities[home_city_id - 1];
    
    // Visit each village in sequence
    for (const auto& drop : trip.drops) {
        Point village_coords = problem.villages[drop.village_id - 1].coords;
        total_distance += distance(current, village_coords);
        current = village_coords;
    }
    
    // Return to home city
    total_distance += distance(current, problem.cities[home_city_id - 1]);
    return total_distance;
}

/**
 * @brief Calculate total weight of packages in a trip
 */
double calculateTripWeight(const Trip& trip, const ProblemData& problem) {
    return trip.dry_food_pickup * problem.packages[0].weight + 
           trip.perishable_food_pickup * problem.packages[1].weight + 
           trip.other_supplies_pickup * problem.packages[2].weight;
}

/**
 * @brief Calculate helicopter priority score combining efficiency and village urgency
 */
double computeHelicopterPriority(const Helicopter& helicopter, const ProblemData& problem,
                                const GlobalState& global_state) {
    // Find nearest unserved village
    double min_distance = numeric_limits<double>::max();
    double nearest_urgency = 0.0;
    
    Point home_pos = problem.cities[helicopter.home_city_id - 1];
    
    for (const auto& village : problem.villages) {
        if (global_state.unmet_food_demand[village.id] <= 0 && 
            global_state.unmet_other_demand[village.id] <= 0) {
            continue;
        }
        
        double dist = distance(home_pos, village.coords);
        if (dist < min_distance) {
            min_distance = dist;
            nearest_urgency = global_state.village_urgencies.at(village.id);
        }
    }
    
    if (min_distance == numeric_limits<double>::max()) return 0.0;
    
    // Calculate helicopter efficiency: (capacity × range) / (cost × fuel × distance)
    double helicopter_efficiency = (helicopter.weight_capacity * helicopter.distance_capacity) / 
                                 (helicopter.fixed_cost * helicopter.alpha * (min_distance + 1));
    
    // Combine helicopter efficiency with village urgency
    return LAMBDA_HELICOPTER * helicopter_efficiency + LAMBDA_VILLAGE * nearest_urgency;
}

/**
 * @brief Get dynamic package ratios with critical need adjustments
 */
vector<PackageRatio> getDynamicPackageRatios(const Village& village, const ProblemData& problem,
                                           const GlobalState& global_state) {
    vector<PackageRatio> ratios;
    
    for (int i = 0; i < 3; i++) {
        PackageRatio ratio;
        ratio.package_type = i;
        ratio.adjusted_ratio = problem.packages[i].value / problem.packages[i].weight;
        
        // Boost ratio if village is critically underserved for this package
        double unmet = (i < 2) ? global_state.unmet_food_demand[village.id] : 
                                global_state.unmet_other_demand[village.id];
        double total_need = (i < 2) ? village.population * 9.0 : village.population * 1.0;
        
        if (unmet > 0 && unmet > total_need * 0.8) { // Critically underserved
            ratio.adjusted_ratio *= 1.5; // Boost priority
        }
        
        ratios.push_back(ratio);
    }
    
    sort(ratios.begin(), ratios.end()); // Sort by adjusted ratio (descending)
    return ratios;
}

/**
 * @brief Create optimal trip for a single village using value/weight greedy packing
 */
Trip createOptimalTripForVillage(const Village& village, const Helicopter& helicopter,
                               const ProblemData& problem, GlobalState& global_state) {
    Trip trip;
    trip.dry_food_pickup = 0;
    trip.perishable_food_pickup = 0;
    trip.other_supplies_pickup = 0;
    
    double remaining_capacity = helicopter.weight_capacity;
    
    // Get dynamic package ratios for this village
    vector<PackageRatio> package_ratios = getDynamicPackageRatios(village, problem, global_state);
    
    Drop drop;
    drop.village_id = village.id;
    drop.dry_food = 0;
    drop.perishable_food = 0;
    drop.other_supplies = 0;
    
    // Allocate packages greedily by value/weight ratio
    for (const auto& ratio : package_ratios) {
        double unmet_need = 0;
        double package_weight = problem.packages[ratio.package_type].weight;
        
        if (ratio.package_type < 2) { // Food packages
            unmet_need = global_state.unmet_food_demand[village.id];
        } else { // Other supplies
            unmet_need = global_state.unmet_other_demand[village.id];
        }
        
        if (unmet_need <= 0 || remaining_capacity < package_weight) continue;
        
        int max_packages = min(static_cast<int>(unmet_need), 
                              static_cast<int>(remaining_capacity / package_weight));
        
        if (max_packages > 0) {
            switch (ratio.package_type) {
                case 0: // Dry food
                    drop.dry_food = max_packages;
                    trip.dry_food_pickup += max_packages;
                    break;
                case 1: // Perishable food
                    drop.perishable_food = max_packages;
                    trip.perishable_food_pickup += max_packages;
                    break;
                case 2: // Other supplies
                    drop.other_supplies = max_packages;
                    trip.other_supplies_pickup += max_packages;
                    break;
            }
            remaining_capacity -= max_packages * package_weight;
        }
    }
    
    // Only add drop if we're delivering something
    if (drop.dry_food > 0 || drop.perishable_food > 0 || drop.other_supplies > 0) {
        trip.drops.push_back(drop);
    }
    
    return trip;
}

/**
 * @brief Simple TSP heuristic to optimize village visit order
 */
void optimizeTripRoute(Trip& trip, const ProblemData& problem, int home_city_id) {
    if (trip.drops.size() <= 1) return;
    
    vector<Drop> optimized_drops;
    vector<bool> visited(trip.drops.size(), false);
    Point current_pos = problem.cities[home_city_id - 1];
    
    // Nearest neighbor heuristic
    for (size_t i = 0; i < trip.drops.size(); i++) {
        int best_idx = -1;
        double min_distance = numeric_limits<double>::max();
        
        for (size_t j = 0; j < trip.drops.size(); j++) {
            if (visited[j]) continue;
            
            Point village_pos = problem.villages[trip.drops[j].village_id - 1].coords;
            double dist = distance(current_pos, village_pos);
            
            if (dist < min_distance) {
                min_distance = dist;
                best_idx = j;
            }
        }
        
        if (best_idx != -1) {
            visited[best_idx] = true;
            optimized_drops.push_back(trip.drops[best_idx]);
            current_pos = problem.villages[trip.drops[best_idx].village_id - 1].coords;
        }
    }
    
    trip.drops = optimized_drops;
}

/**
 * @brief Evaluate net value of a trip (delivered value - trip cost)
 */
double evaluateTripValue(const Trip& trip, const ProblemData& problem, 
                        const Helicopter& helicopter, const GlobalState& global_state) {
    if (trip.drops.empty()) return 0.0;
    
    double total_value = 0.0;
    
    // Calculate value from each drop with capping logic
    for (const auto& drop : trip.drops) {
        const Village& village = problem.villages[drop.village_id - 1];
        
        // Food value with capping (max 9 meals per person)
        double max_food_needed = village.population * 9.0;
        double food_room_left = max(0.0, max_food_needed - 
                                   (village.population * 9.0 - global_state.unmet_food_demand[drop.village_id]));
        double food_in_drop = drop.dry_food + drop.perishable_food;
        double effective_food = min(static_cast<double>(food_in_drop), food_room_left);
        
        // Prioritize perishable food (higher value)
        double effective_perishable = min(static_cast<double>(drop.perishable_food), effective_food);
        double effective_dry = min(static_cast<double>(drop.dry_food), effective_food - effective_perishable);
        
        total_value += effective_perishable * problem.packages[1].value;
        total_value += effective_dry * problem.packages[0].value;
        
        // Other supplies value with capping (max 1 per person)
        double max_other_needed = village.population * 1.0;
        double other_room_left = max(0.0, max_other_needed - 
                                    (village.population * 1.0 - global_state.unmet_other_demand[drop.village_id]));
        double effective_other = min(static_cast<double>(drop.other_supplies), other_room_left);
        
        total_value += effective_other * problem.packages[2].value;
    }
    
    // Calculate trip cost
    double trip_distance = calculateTripDistance(trip, problem, helicopter.home_city_id);
    double trip_cost = helicopter.fixed_cost + helicopter.alpha * trip_distance;
    
    return total_value - trip_cost;
}

/**
 * @brief Find nearby villages that can be served in a single trip
 * 
 * This function implements the multi-village trip planning strategy by:
 * 1. Finding villages near the helicopter's home city with unmet demands
 * 2. Clustering them based on proximity and helicopter capacity
 * 3. Selecting a batch of villages that can be efficiently served together
 * 
 * @param helicopter The helicopter to plan for
 * @param problem The problem data
 * @param global_state Current state of all villages and helicopters
 * @param village_order Priority order of villages
 * @return vector<int> IDs of villages to visit in this trip
 */
vector<int> findNearbyVillages(const Helicopter& helicopter, const ProblemData& problem,
                              const GlobalState& global_state, const vector<int>& village_order) {
    // Get helicopter's home city position
    Point home_pos = problem.cities[helicopter.home_city_id - 1];
    
    // Calculate distance from home city to each village with unmet demand
    vector<VillageDistance> village_distances;
    for (int v_idx : village_order) {
        const Village& village = problem.villages[v_idx];
        
        // Skip if no remaining demand
        if (global_state.unmet_food_demand[village.id] <= 0 && 
            global_state.unmet_other_demand[village.id] <= 0) {
            continue;
        }
        
        double dist = distance(home_pos, village.coords);
        
        // Skip villages too far for a round trip
        if (dist * 2 > helicopter.distance_capacity * 0.8) {
            continue;
        }
        
        VillageDistance vd;
        vd.village_id = village.id;
        vd.distance = dist;
        village_distances.push_back(vd);
    }
    
    // Sort villages by distance from home city
    sort(village_distances.begin(), village_distances.end());
    
    // Select up to MAX_BATCH_SIZE villages for a batch trip
    vector<int> selected_villages;
    double estimated_total_distance = 0;
    double estimated_total_weight = 0;
    
    // First, add the closest village (if any)
    if (!village_distances.empty()) {
        selected_villages.push_back(village_distances[0].village_id);
        estimated_total_distance = village_distances[0].distance * 2; // Round trip
        
        // Estimate weight needed for this village (rough approximation)
        double food_demand = global_state.unmet_food_demand[village_distances[0].village_id];
        double other_demand = global_state.unmet_other_demand[village_distances[0].village_id];
        
        // Estimate package weights needed (simplified calculation)
        estimated_total_weight += food_demand * 0.7 * problem.packages[0].weight; // 70% dry food
        estimated_total_weight += food_demand * 0.3 * problem.packages[1].weight; // 30% perishable
        estimated_total_weight += other_demand * problem.packages[2].weight;      // other supplies
        
        // Consider additional nearby villages
        for (size_t i = 1; i < village_distances.size() && selected_villages.size() < MAX_BATCH_SIZE; i++) {
            // Calculate extra distance needed to include this village
            double extra_distance = 0;
            if (selected_villages.size() == 1) {
                // When we have only one village, calculate round trip through both villages
                extra_distance = distance(home_pos, problem.villages[village_distances[i].village_id - 1].coords) +
                              distance(problem.villages[village_distances[i].village_id - 1].coords, 
                                     problem.villages[selected_villages[0] - 1].coords) +
                              distance(problem.villages[selected_villages[0] - 1].coords, home_pos) -
                              estimated_total_distance;
            } else {
                // For 3+ villages, use a simple estimation (this is an approximation)
                extra_distance = village_distances[i].distance * 1.5;
            }
            
            // Estimate additional weight needed
            double extra_food = global_state.unmet_food_demand[village_distances[i].village_id];
            double extra_other = global_state.unmet_other_demand[village_distances[i].village_id];
            double extra_weight = extra_food * 0.7 * problem.packages[0].weight +
                                 extra_food * 0.3 * problem.packages[1].weight +
                                 extra_other * problem.packages[2].weight;
            
            // Check if adding this village would exceed helicopter constraints
            if (estimated_total_distance + extra_distance <= helicopter.distance_capacity * 0.95 &&
                estimated_total_weight + extra_weight <= helicopter.weight_capacity * 0.95) {
                
                selected_villages.push_back(village_distances[i].village_id);
                estimated_total_distance += extra_distance;
                estimated_total_weight += extra_weight;
            }
        }
    }
    
    // Return the batch of villages to visit
    return selected_villages;
}

/**
 * @brief Create multi-village trip plan optimized for helicopter and village needs
 * 
 * This function implements the core multi-village trip planning logic by:
 * 1. Selecting a batch of villages that can be efficiently served together
 * 2. Creating an optimal package allocation for each village
 * 3. Combining them into a single coordinated trip
 * 4. Optimizing the route to minimize travel distance
 * 
 * @param helicopter The helicopter to plan for
 * @param problem The problem data
 * @param global_state Current state of all villages and helicopters
 * @param village_order Priority order of villages
 * @return Trip The optimized multi-village trip plan
 */
Trip createMultiVillageTripPlan(const Helicopter& helicopter, const ProblemData& problem,
                             GlobalState& global_state, const vector<int>& village_order) {
    // Find a batch of villages to serve in this trip
    vector<int> village_batch = findNearbyVillages(helicopter, problem, global_state, village_order);
    
    // If no suitable villages found, return empty trip
    if (village_batch.empty()) {
        Trip empty_trip;
        return empty_trip;
    }
    
    // Initialize the trip
    Trip trip;
    trip.dry_food_pickup = 0;
    trip.perishable_food_pickup = 0;
    trip.other_supplies_pickup = 0;
    
    double remaining_capacity = helicopter.weight_capacity;
    
    // Create a temporary copy of global state to simulate allocations
    GlobalState temp_state = global_state;
    
    // First pass: create initial allocation for each village (greedily)
    for (int village_id : village_batch) {
        const Village& village = problem.villages[village_id - 1];
        
        // Get package ratios for this village
        vector<PackageRatio> package_ratios = getDynamicPackageRatios(village, problem, global_state);
        
        Drop drop;
        drop.village_id = village_id;
        drop.dry_food = 0;
        drop.perishable_food = 0;
        drop.other_supplies = 0;
        
        // Allocate packages greedily by value/weight ratio (similar to createOptimalTripForVillage)
        for (const auto& ratio : package_ratios) {
            double unmet_need = 0;
            double package_weight = problem.packages[ratio.package_type].weight;
            
            if (ratio.package_type < 2) { // Food packages
                unmet_need = temp_state.unmet_food_demand[village_id];
            } else { // Other supplies
                unmet_need = temp_state.unmet_other_demand[village_id];
            }
            
            if (unmet_need <= 0 || remaining_capacity < package_weight) continue;
            
            // Limit allocation to 80% of what's needed to leave room for other villages
            int max_packages = min(static_cast<int>(unmet_need * 0.8), 
                                  static_cast<int>(remaining_capacity / package_weight));
            
            if (max_packages > 0) {
                switch (ratio.package_type) {
                    case 0: // Dry food
                        drop.dry_food = max_packages;
                        trip.dry_food_pickup += max_packages;
                        temp_state.unmet_food_demand[village_id] -= max_packages;
                        break;
                    case 1: // Perishable food
                        drop.perishable_food = max_packages;
                        trip.perishable_food_pickup += max_packages;
                        temp_state.unmet_food_demand[village_id] -= max_packages;
                        break;
                    case 2: // Other supplies
                        drop.other_supplies = max_packages;
                        trip.other_supplies_pickup += max_packages;
                        temp_state.unmet_other_demand[village_id] -= max_packages;
                        break;
                }
                remaining_capacity -= max_packages * package_weight;
            }
        }
        
        // Add drop if we're delivering something
        if (drop.dry_food > 0 || drop.perishable_food > 0 || drop.other_supplies > 0) {
            trip.drops.push_back(drop);
        }
    }
    
    // Second pass: use remaining capacity for additional deliveries
    // This ensures we maximize helicopter utilization
    if (remaining_capacity > 0) {
        for (int village_id : village_batch) {
            // Skip if all needs are met
            if (temp_state.unmet_food_demand[village_id] <= 0 && 
                temp_state.unmet_other_demand[village_id] <= 0) {
                continue;
            }
            
            // Find corresponding drop
            int drop_idx = -1;
            for (size_t i = 0; i < trip.drops.size(); i++) {
                if (trip.drops[i].village_id == village_id) {
                    drop_idx = i;
                    break;
                }
            }
            
            // Create new drop if needed
            Drop* drop;
            if (drop_idx == -1) {
                trip.drops.push_back(Drop());
                drop = &trip.drops.back();
                drop->village_id = village_id;
                drop->dry_food = 0;
                drop->perishable_food = 0;
                drop->other_supplies = 0;
            } else {
                drop = &trip.drops[drop_idx];
            }
            
            // Allocate remaining capacity optimally
            vector<PackageRatio> package_ratios = getDynamicPackageRatios(problem.villages[village_id - 1], 
                                                                      problem, global_state);
            
            for (const auto& ratio : package_ratios) {
                double unmet_need = 0;
                double package_weight = problem.packages[ratio.package_type].weight;
                
                if (ratio.package_type < 2) { // Food packages
                    unmet_need = temp_state.unmet_food_demand[village_id];
                } else { // Other supplies
                    unmet_need = temp_state.unmet_other_demand[village_id];
                }
                
                if (unmet_need <= 0 || remaining_capacity < package_weight) continue;
                
                int additional_packages = min(static_cast<int>(unmet_need), 
                                           static_cast<int>(remaining_capacity / package_weight));
                
                if (additional_packages > 0) {
                    switch (ratio.package_type) {
                        case 0: // Dry food
                            drop->dry_food += additional_packages;
                            trip.dry_food_pickup += additional_packages;
                            break;
                        case 1: // Perishable food
                            drop->perishable_food += additional_packages;
                            trip.perishable_food_pickup += additional_packages;
                            break;
                        case 2: // Other supplies
                            drop->other_supplies += additional_packages;
                            trip.other_supplies_pickup += additional_packages;
                            break;
                    }
                    remaining_capacity -= additional_packages * package_weight;
                }
            }
        }
    }
    
    // Optimize the route (use TSP heuristic)
    optimizeTripRoute(trip, problem, helicopter.home_city_id);
    
    return trip;
}

/**
 * @brief Check if trip violates constraints and by how much
 * 
 * This function evaluates whether a trip violates weight or distance constraints
 * and quantifies the magnitude of the violation.
 * 
 * @param trip The trip to check
 * @param helicopter The helicopter for the trip
 * @param problem The problem data
 * @param current_distance Total distance already traveled by helicopter
 * @return pair<bool, double> First: has violation, Second: violation percentage
 */
pair<bool, double> checkConstraintViolations(const Trip& trip, const Helicopter& helicopter,
                                           const ProblemData& problem, double current_distance) {
    // Calculate trip metrics
    double trip_weight = calculateTripWeight(trip, problem);
    double trip_distance = calculateTripDistance(trip, problem, helicopter.home_city_id);
    double total_distance = current_distance + trip_distance;
    
    // Strict constraint checking without tolerance
    bool weight_violation = trip_weight > helicopter.weight_capacity;
    bool trip_distance_violation = trip_distance > helicopter.distance_capacity;
    bool total_distance_violation = total_distance > problem.d_max;
    
    // Return true if any constraint is violated
    return make_pair(weight_violation || trip_distance_violation || total_distance_violation, 0.0);
}

/**
 * @brief Run one iteration of the enhanced greedy algorithm
 */
Solution runGreedyIteration(const ProblemData& problem, vector<int>& helicopter_order, vector<int>& village_order) {
    Solution solution(problem.helicopters.size());
    
    // Initialize helicopter plans
    for (size_t i = 0; i < problem.helicopters.size(); i++) {
        solution[i].helicopter_id = problem.helicopters[i].id;
    }
    
    // Initialize global state tracking
    GlobalState global_state(problem);
    
    // Priority queue for helicopter scheduling
    priority_queue<HelicopterPriority> helicopter_queue;
    
    // Initial helicopter priorities
    for (int h_idx : helicopter_order) {
        const Helicopter& helicopter = problem.helicopters[h_idx];
        double priority = computeHelicopterPriority(helicopter, problem, global_state);
        
        if (priority > 0) {
            HelicopterPriority hp;
            hp.helicopter_index = h_idx;
            hp.priority = priority;
            helicopter_queue.push(hp);
        }
    }
    
    // Main assignment loop
    while (!helicopter_queue.empty() && global_state.hasUnmetDemands()) {
        // Get highest priority helicopter
        HelicopterPriority hp = helicopter_queue.top();
        helicopter_queue.pop();
        
        int h_idx = hp.helicopter_index;
        const Helicopter& helicopter = problem.helicopters[h_idx];
        
        // Skip if helicopter has reached distance limit
        if (global_state.helicopter_distances[helicopter.id] >= problem.d_max) {
            continue;
        }
        
        // FEATURE 1: MULTI-VILLAGE TRIP PLANNING
        // Instead of considering villages one at a time, plan a multi-village trip
        Trip best_trip = createMultiVillageTripPlan(helicopter, problem, global_state, village_order);
        
        // If no trip was created, continue with the next helicopter
        if (best_trip.drops.empty()) {
            continue;
        }
        
        // Evaluate trip value before constraint check
        double trip_value = evaluateTripValue(best_trip, problem, helicopter, global_state);
        
        // Get current helicopter distance
        double current_distance = global_state.helicopter_distances[helicopter.id];
        
        // Strict constraint checking without relaxation
        auto [has_violation, _] = checkConstraintViolations(best_trip, helicopter, problem, current_distance);
        
        if (has_violation) {
            // Skip this trip if it violates any constraint
            continue;
        }

        // Add best trip if profitable
        if (trip_value > 0.0 && !best_trip.drops.empty()) {
            solution[h_idx].trips.push_back(best_trip);
            
            // Update global state
            double best_distance = calculateTripDistance(best_trip, problem, helicopter.home_city_id);
            global_state.helicopter_distances[helicopter.id] += best_distance;
            
            // Update unmet demands
            for (const auto& drop : best_trip.drops) {
                global_state.unmet_food_demand[drop.village_id] -= (drop.dry_food + drop.perishable_food);
                global_state.unmet_other_demand[drop.village_id] -= drop.other_supplies;
                
                // Ensure non-negative
                global_state.unmet_food_demand[drop.village_id] = max(0.0, global_state.unmet_food_demand[drop.village_id]);
                global_state.unmet_other_demand[drop.village_id] = max(0.0, global_state.unmet_other_demand[drop.village_id]);
            }
            
            // Recalculate village urgencies
            global_state.updateAllUrgencies(problem);
            
            // Recompute helicopter priority and reinsert if still capable
            if (global_state.helicopter_distances[helicopter.id] < problem.d_max) {
                double new_priority = computeHelicopterPriority(helicopter, problem, global_state);
                if (new_priority > 0) {
                    HelicopterPriority new_hp;
                    new_hp.helicopter_index = h_idx;
                    new_hp.priority = new_priority;
                    helicopter_queue.push(new_hp);
                }
            }
        }
    }
    
    return solution;
}

/**
 * @brief Main enhanced greedy algorithm with multi-start
 */
Solution solve(const ProblemData& problem) {
    cout << "Starting Enhanced Greedy Algorithm with Multi-Village Planning..." << endl;
    
    Solution best_solution;
    double best_objective = -numeric_limits<double>::max();
    
    // Random number generator for multi-start
    random_device rd;
    default_random_engine rng(rd());
    
    // Multi-start loop
    for (int attempt = 0; attempt < MAX_TRIES; attempt++) {
        cout << "Multi-start attempt " << (attempt + 1) << " of " << MAX_TRIES << endl;
        
        // Create randomized orders for helicopters and villages
        vector<int> helicopter_order(problem.helicopters.size());
        vector<int> village_order(problem.villages.size());
        
        iota(helicopter_order.begin(), helicopter_order.end(), 0);
        iota(village_order.begin(), village_order.end(), 0);
        
        shuffle(helicopter_order.begin(), helicopter_order.end(), rng);
        shuffle(village_order.begin(), village_order.end(), rng);
        
        // Run greedy iteration with our enhanced features
        Solution solution = runGreedyIteration(problem, helicopter_order, village_order);
        
        // Simple objective evaluation (count trips and deliveries)
        double objective = 0.0;
        for (const auto& helicopter_plan : solution) {
            // Track trip metrics for detailed evaluation
            int total_trips = helicopter_plan.trips.size();
            int total_villages_served = 0;
            int total_food_delivered = 0;
            int total_other_delivered = 0;
            
            for (const auto& trip : helicopter_plan.trips) {
                total_villages_served += trip.drops.size();
                
                for (const auto& drop : trip.drops) {
                    total_food_delivered += drop.dry_food + drop.perishable_food;
                    total_other_delivered += drop.other_supplies;
                    
                    // Calculate value contribution (with package value weights)
                    objective += drop.dry_food * problem.packages[0].value;
                    objective += drop.perishable_food * problem.packages[1].value;
                    objective += drop.other_supplies * problem.packages[2].value;
                }
                
                // Subtract trip costs
                const Helicopter& helicopter = problem.helicopters[helicopter_plan.helicopter_id - 1];
                double trip_distance = calculateTripDistance(trip, problem, helicopter.home_city_id);
                double trip_cost = helicopter.fixed_cost + helicopter.alpha * trip_distance;
                objective -= trip_cost;
            }
            
            cout << "  Helicopter #" << helicopter_plan.helicopter_id 
                 << ": " << total_trips << " trips, " 
                 << total_villages_served << " villages, "
                 << total_food_delivered << " food, "
                 << total_other_delivered << " other supplies" << endl;
        }
        
        if (objective > best_objective) {
            best_objective = objective;
            best_solution = solution;
            cout << "New best solution found with objective value: " << best_objective << endl;
        }
    }
    
    cout << "Enhanced Greedy Algorithm completed." << endl;
    return best_solution;
}