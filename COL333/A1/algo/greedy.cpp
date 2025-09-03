#include "greedy.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <cmath>

using namespace std;

/**
 * @brief Calculate the total distance of a trip
 */
double calculateTripDistance(const Trip& trip, const ProblemData& problem, int home_city_id) {
    if (trip.drops.empty()) {
        return 0.0;
    }

    const Point& home_city = problem.cities[home_city_id - 1];
    Point current = home_city;
    double total_distance = 0.0;

    // Visit each village in order
    for (const auto& drop : trip.drops) {
        const Point& village_coords = problem.villages[drop.village_id - 1].coords;
        total_distance += distance(current, village_coords);
        current = village_coords;
    }

    // Return to home city
    total_distance += distance(current, home_city);
    return total_distance;
}

/**
 * @brief Calculate the weight of packages in a trip
 */
double calculateTripWeight(const Trip& trip, const ProblemData& problem) {
    return trip.dry_food_pickup * problem.packages[0].weight + 
           trip.perishable_food_pickup * problem.packages[1].weight + 
           trip.other_supplies_pickup * problem.packages[2].weight;
}

/**
 * @brief Evaluate the value added by a trip (value of delivered goods - trip cost)
 */
double evaluateTripValue(const Trip& trip, const ProblemData& problem, const Helicopter& helicopter, 
                        const vector<double>& food_delivered, const vector<double>& other_delivered) {
    double value_gained = 0.0;
    
    // Make a local copy to track what we deliver in this trip
    vector<double> trip_food_delivered = food_delivered;
    vector<double> trip_other_delivered = other_delivered;

    // Calculate value gained from each drop
    for (const auto& drop : trip.drops) {
        const Village& village = problem.villages[drop.village_id - 1];
        
        // Value capping logic (similar to format_checker.cpp)
        double max_food_needed = village.population * 9.0;
        double food_room_left = max(0.0, max_food_needed - trip_food_delivered[drop.village_id]);
        double food_in_this_drop = drop.dry_food + drop.perishable_food;
        double effective_food = min(food_in_this_drop, food_room_left);

        // Prioritize perishable food (higher value)
        double effective_perishable = min((double)drop.perishable_food, effective_food);
        double value_from_perishable = effective_perishable * problem.packages[1].value;

        // Then account for dry food with remaining capacity
        double remaining_effective_food = effective_food - effective_perishable;
        double effective_dry = min((double)drop.dry_food, remaining_effective_food);
        double value_from_dry = effective_dry * problem.packages[0].value;

        // Update food delivered
        trip_food_delivered[drop.village_id] += food_in_this_drop;

        // Now handle other supplies
        double max_other_needed = village.population * 1.0;
        double other_room_left = max(0.0, max_other_needed - trip_other_delivered[drop.village_id]);
        double effective_other = min((double)drop.other_supplies, other_room_left);
        double value_from_other = effective_other * problem.packages[2].value;

        // Update other supplies delivered
        trip_other_delivered[drop.village_id] += drop.other_supplies;

        // Add to total value gained
        value_gained += value_from_perishable + value_from_dry + value_from_other;
    }

    // Calculate trip cost
    double trip_distance = calculateTripDistance(trip, problem, helicopter.home_city_id);
    double trip_cost = helicopter.fixed_cost + (helicopter.alpha * trip_distance);

    // Return net value (value gained - trip cost)
    return value_gained - trip_cost;
}

/**
 * @brief Create an optimal package mix for delivering to a village
 */
Trip createOptimalTripForVillage(const Village& village, const Helicopter& helicopter, 
                               const ProblemData& problem,
                               double& remaining_food_needed, double& remaining_other_needed) {
    Trip trip;
    
    // Calculate how much we could deliver to this village
    double food_to_deliver = min(remaining_food_needed, 9.0 * village.population);
    double other_to_deliver = min(remaining_other_needed, 1.0 * village.population);
    
    // Package weights
    double dry_weight = problem.packages[0].weight;
    double perishable_weight = problem.packages[1].weight;
    double other_weight = problem.packages[2].weight;
    
    // Start with other supplies (they have different needs)
    int other_packages = min((int)other_to_deliver, (int)(helicopter.weight_capacity / other_weight));
    double remaining_capacity = helicopter.weight_capacity - (other_packages * other_weight);
    
    // Use perishable food first (higher value)
    int perishable_packages = min((int)food_to_deliver, (int)(remaining_capacity / perishable_weight));
    remaining_capacity -= perishable_packages * perishable_weight;
    
    // Use dry food for remaining capacity
    int dry_packages = min((int)(food_to_deliver - perishable_packages), (int)(remaining_capacity / dry_weight));
    
    // Update trip pickups
    trip.dry_food_pickup = dry_packages;
    trip.perishable_food_pickup = perishable_packages;
    trip.other_supplies_pickup = other_packages;
    
    // Create drop for this village
    Drop drop;
    drop.village_id = village.id;
    drop.dry_food = dry_packages;
    drop.perishable_food = perishable_packages;
    drop.other_supplies = other_packages;
    trip.drops.push_back(drop);
    
    // Update remaining needs
    remaining_food_needed -= (dry_packages + perishable_packages);
    remaining_other_needed -= other_packages;
    
    return trip;
}

/**
 * @brief Try to add a village to an existing trip if constraints allow
 */
bool tryAddVillageToTrip(Trip& trip, const Village& village, const Helicopter& helicopter, 
                        const ProblemData& problem, double& remaining_food_needed,
                        double& remaining_other_needed) {
    // Check if adding this village would exceed distance capacity
    Trip temp_trip = trip;
    Drop new_drop;
    new_drop.village_id = village.id;
    new_drop.dry_food = 0;
    new_drop.perishable_food = 0;
    new_drop.other_supplies = 0;
    temp_trip.drops.push_back(new_drop);
    
    double new_trip_distance = calculateTripDistance(temp_trip, problem, helicopter.home_city_id);
    if (new_trip_distance > helicopter.distance_capacity) {
        return false;
    }
    
    // Calculate remaining capacity
    double current_weight = calculateTripWeight(trip, problem);
    double remaining_capacity = helicopter.weight_capacity - current_weight;
    
    // Calculate how much we can deliver to this village
    double food_to_deliver = min(remaining_food_needed, 9.0 * village.population);
    double other_to_deliver = min(remaining_other_needed, 1.0 * village.population);
    
    // Package weights
    double dry_weight = problem.packages[0].weight;
    double perishable_weight = problem.packages[1].weight;
    double other_weight = problem.packages[2].weight;
    
    // Start with other supplies
    int other_packages = min((int)other_to_deliver, (int)(remaining_capacity / other_weight));
    remaining_capacity -= other_packages * other_weight;
    
    // Use perishable food first (higher value)
    int perishable_packages = min((int)food_to_deliver, (int)(remaining_capacity / perishable_weight));
    remaining_capacity -= perishable_packages * perishable_weight;
    
    // Use dry food for remaining capacity
    int dry_packages = min((int)(food_to_deliver - perishable_packages), (int)(remaining_capacity / dry_weight));
    
    // If we can't deliver anything, don't add this village
    if (other_packages + perishable_packages + dry_packages == 0) {
        return false;
    }
    
    // Create and add the drop
    Drop drop;
    drop.village_id = village.id;
    drop.dry_food = dry_packages;
    drop.perishable_food = perishable_packages;
    drop.other_supplies = other_packages;
    trip.drops.push_back(drop);
    
    // Update trip pickups
    trip.dry_food_pickup += dry_packages;
    trip.perishable_food_pickup += perishable_packages;
    trip.other_supplies_pickup += other_packages;
    
    // Update remaining needs
    remaining_food_needed -= (dry_packages + perishable_packages);
    remaining_other_needed -= other_packages;
    
    return true;
}

/**
 * @brief Score a village for helicopter selection based on distance and needs
 */
double scoreVillage(const Village& village, const Point& helicopter_position, 
                   double food_needed, double other_needed) {
    // If village doesn't need anything, score is 0
    if (food_needed <= 0 && other_needed <= 0) {
        return 0.0;
    }
    
    // Score based on distance (closer is better) and needs (more is better)
    double dist = distance(helicopter_position, village.coords);
    if (dist < 1.0) dist = 1.0; // Avoid division by zero
    
    // Combine factors - give more weight to villages with higher needs
    double need_factor = food_needed + (other_needed * 10.0); // Other supplies are more critical
    return need_factor / dist;
}

/**
 * @brief Create a greedy solution to the helicopter routing problem
 */
Solution solveWithGreedy(const ProblemData& problem) {
    Solution solution(problem.helicopters.size());
    
    // Initialize helicopter plans with correct ids
    for (size_t h = 0; h < problem.helicopters.size(); h++) {
        solution[h].helicopter_id = problem.helicopters[h].id;
    }
    
    // Track remaining needs for each village
    vector<double> remaining_food_needed(problem.villages.size() + 1);
    vector<double> remaining_other_needed(problem.villages.size() + 1);
    
    for (const auto& village : problem.villages) {
        remaining_food_needed[village.id] = 9.0 * village.population; // 9 meals per person
        remaining_other_needed[village.id] = 1.0 * village.population; // 1 other supply per person
    }
    
    // Track global state
    vector<double> food_delivered(problem.villages.size() + 1, 0.0);
    vector<double> other_delivered(problem.villages.size() + 1, 0.0);
    vector<double> helicopter_distance_used(problem.helicopters.size() + 1, 0.0);
    
    // Keep going until we can't deliver more
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        
        // Try each helicopter
        for (size_t h = 0; h < problem.helicopters.size(); h++) {
            const Helicopter& helicopter = problem.helicopters[h];
            
            // Skip if helicopter has reached distance limit
            if (helicopter_distance_used[helicopter.id] >= problem.d_max) {
                continue;
            }
            
            // Find the best trip for this helicopter
            Trip best_trip;
            double best_trip_value = 0.0;
            double best_trip_distance = 0.0;
            
            // Try each village as starting point for new trip
            for (const auto& village : problem.villages) {
                // Skip if village has no remaining needs
                if (remaining_food_needed[village.id] <= 0 && remaining_other_needed[village.id] <= 0) {
                    continue;
                }
                
                // Try a trip to just this village
                double village_food_needed = remaining_food_needed[village.id];
                double village_other_needed = remaining_other_needed[village.id];
                Trip trip = createOptimalTripForVillage(village, helicopter, problem, 
                                                       village_food_needed, village_other_needed);
                
                // Calculate trip distance
                double trip_distance = calculateTripDistance(trip, problem, helicopter.home_city_id);
                
                // Skip if trip exceeds distance constraints
                double total_distance = helicopter_distance_used[helicopter.id] + trip_distance;
                if (trip_distance > helicopter.distance_capacity || total_distance > problem.d_max) {
                    continue;
                }
                
                // Calculate value of this trip
                double trip_value = evaluateTripValue(trip, problem, helicopter, food_delivered, other_delivered);
                
                // Update best trip if this is better
                if (trip_value > best_trip_value) {
                    best_trip = trip;
                    best_trip_value = trip_value;
                    best_trip_distance = trip_distance;
                }
                
                // Now try adding nearby villages to this trip
                Trip multi_village_trip = trip;
                vector<double> local_food_needed = remaining_food_needed; // Make a copy
                vector<double> local_other_needed = remaining_other_needed; // Make a copy

                // Update local needs from first village
                local_food_needed[village.id] = village_food_needed;
                local_other_needed[village.id] = village_other_needed;
                
                // Create a set of visited villages to avoid duplicates
                unordered_set<int> visited_villages = {village.id};
                
                // Try adding up to 2 more villages (for a total of 3)
                for (int i = 0; i < 2; i++) {
                    // Find best village to add next
                    int best_next_village = -1;
                    double best_score = -1.0;
                    
                    // Current end position of trip
                    Point current_position = village.coords;
                    if (!multi_village_trip.drops.empty()) {
                        int last_village_id = multi_village_trip.drops.back().village_id;
                        current_position = problem.villages[last_village_id - 1].coords;
                    }
                    
                    // Score each unvisited village
                    for (const auto& next_village : problem.villages) {
                        // Skip if already visited or no needs
                        if (visited_villages.count(next_village.id) || 
                            (local_food_needed[next_village.id] <= 0 && 
                             local_other_needed[next_village.id] <= 0)) {
                            continue;
                        }
                        
                        double score = scoreVillage(next_village, current_position, 
                                                local_food_needed[next_village.id],
                                                local_other_needed[next_village.id]);
                        
                        if (score > best_score) {
                            best_score = score;
                            best_next_village = next_village.id;
                        }
                    }
                    
                    // If found a village to add
                    if (best_next_village != -1) {
                        const Village& next_village = problem.villages[best_next_village - 1];
                        
                        // Try to add it to the trip
                        if (tryAddVillageToTrip(multi_village_trip, next_village, helicopter, problem,
                                              local_food_needed[next_village.id], 
                                              local_other_needed[next_village.id])) {
                            visited_villages.insert(next_village.id);
                            
                            // Calculate new trip distance
                            double new_trip_distance = calculateTripDistance(multi_village_trip, problem, 
                                                                          helicopter.home_city_id);
                            
                            // Skip if trip exceeds distance constraints
                            double new_total_distance = helicopter_distance_used[helicopter.id] + new_trip_distance;
                            if (new_trip_distance > helicopter.distance_capacity || new_total_distance > problem.d_max) {
                                // Undo the addition
                                multi_village_trip.drops.pop_back();
                                continue;
                            }
                            
                            // Calculate value of multi-village trip
                            double multi_trip_value = evaluateTripValue(multi_village_trip, problem, helicopter,
                                                                     food_delivered, other_delivered);
                            
                            // Update best trip if this is better
                            if (multi_trip_value > best_trip_value) {
                                best_trip = multi_village_trip;
                                best_trip_value = multi_trip_value;
                                best_trip_distance = new_trip_distance;
                            }
                        }
                    } else {
                        // No more villages to add
                        break;
                    }
                }
            }
            
            // Add the best trip if it has positive value
            if (best_trip_value > 0.0 && !best_trip.drops.empty()) {
                // Update solution
                solution[h].trips.push_back(best_trip);
                
                // Update global state
                for (const auto& drop : best_trip.drops) {
                    food_delivered[drop.village_id] += drop.dry_food + drop.perishable_food;
                    other_delivered[drop.village_id] += drop.other_supplies;
                    
                    remaining_food_needed[drop.village_id] -= (drop.dry_food + drop.perishable_food);
                    remaining_other_needed[drop.village_id] -= drop.other_supplies;
                }
                
                helicopter_distance_used[helicopter.id] += best_trip_distance;
                made_progress = true;
            }
        }
    }
    
    return solution;
}
