#include "solver.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <queue>

using namespace std;
// Global variables for tracking time
chrono::steady_clock::time_point start_time;
chrono::milliseconds time_limit;

/**
     // Parameters for simulated annealing
    double initialTemp = 1000.0;  // Higher starting temperature for more exploration
    double coolingRate = 0.97;    // Slower cooling rate
    double temp = initialTemp;
    int noImprovementCount = 0;
    int maxNoImprovement = 3000;  // Allow more iterations without improvement
    
    cout << "Initial solution score: " << bestScore << ", feasible: " << bestFeasible << endl; Calculate distance between two points in 2D space
 */
double calculateDistance(const Point& p1, const Point& p2) {
    return distance(p1, p2);
}

/**
 * @brief Calculate the total distance of a trip
 * @param cities Vector of cities
 * @param villages Vector of villages
 * @param homeCity Home city point
 * @param route Vector of village IDs representing the route
 * @return Total distance of the trip
 */
double calculateTripDistance(const vector<Point>& cities, const vector<Village>& villages, 
                           const Point& homeCity, const vector<int>& route) {
    double totalDist = 0.0;
    
    Point currentPos = homeCity;
    
    // Visit each village in the route
    for (int villageId : route) {
        const Point& villagePos = villages[villageId - 1].coords;
        totalDist += calculateDistance(currentPos, villagePos);
        currentPos = villagePos;
    }
    
    // Return to home city
    totalDist += calculateDistance(currentPos, homeCity);
    
    return totalDist;
}

/**
 * @brief Check if we have enough time to continue optimization
 * @param buffer Time buffer in milliseconds to ensure we exit before hard deadline
 * @return True if we have enough time, false otherwise
 */
bool hasEnoughTime(int buffer_ms = 1000) {
    auto current_time = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(current_time - start_time);
    return elapsed + chrono::milliseconds(buffer_ms) < time_limit;
}

/**
 * @brief Calculate the value of delivering supplies to a village
 * @param villageId The ID of the village
 * @param dryFood Amount of dry food delivered
 * @param perishableFood Amount of perishable food delivered
 * @param otherSupplies Amount of other supplies delivered
 * @param villages Vector of villages
 * @param packages Vector of package types
 * @param existingDeliveries Map tracking existing deliveries to each village
 * @return Value gained from this delivery
 */
double calculateDeliveryValue(int villageId, int dryFood, int perishableFood, int otherSupplies,
                            const vector<Village>& villages, const vector<PackageInfo>& packages,
                            unordered_map<int, pair<double, double>>& existingDeliveries) {
    const Village& village = villages[villageId - 1];
    double maxFoodNeeded = village.population * 9.0;
    double maxOtherNeeded = village.population * 1.0;
    
    // Get existing deliveries or initialize to zero
    double& existingFood = existingDeliveries[villageId].first;
    double& existingOther = existingDeliveries[villageId].second;
    
    double foodRoomLeft = max(0.0, maxFoodNeeded - existingFood);
    double otherRoomLeft = max(0.0, maxOtherNeeded - existingOther);
    
    double foodInThisDrop = dryFood + perishableFood;
    double effectiveFoodThisDrop = min(foodInThisDrop, foodRoomLeft);
    
    // Prioritize perishable food (higher value)
    double effectivePerishable = min((double)perishableFood, effectiveFoodThisDrop);
    double valueFromPerishable = effectivePerishable * packages[1].value;
    
    double remainingEffectiveFood = effectiveFoodThisDrop - effectivePerishable;
    double effectiveDry = min((double)dryFood, remainingEffectiveFood);
    double valueFromDry = effectiveDry * packages[0].value;
    
    double effectiveOther = min((double)otherSupplies, otherRoomLeft);
    double valueFromOther = effectiveOther * packages[2].value;
    
    // Update existing deliveries
    existingFood += foodInThisDrop;
    existingOther += otherSupplies;
    
    return valueFromPerishable + valueFromDry + valueFromOther;
}

/**
 * @brief Calculate the total value and cost of a solution
 * @param solution The solution to evaluate
 * @param problem The problem data
 * @return A pair of (total value, feasibility) where feasibility is true if solution is feasible
 */
pair<double, bool> evaluateSolution(const Solution& solution, const ProblemData& problem) {
    unordered_map<int, pair<double, double>> deliveryTracker; // village_id -> (food, other)
    double totalValue = 0.0;
    double totalCost = 0.0;
    bool feasible = true;
    
    // Track distances by helicopter
    vector<double> heliTotalDistances(problem.helicopters.size(), 0.0);
    
    for (const auto& plan : solution) {
        int heliIdx = plan.helicopter_id - 1;
        const Helicopter& heli = problem.helicopters[heliIdx];
        Point homeCity = problem.cities[heli.home_city_id - 1];
        
        for (const auto& trip : plan.trips) {
            // Skip empty trips
            if (trip.drops.empty()) continue;
            
            // Calculate weight of the trip
            double tripWeight = (trip.dry_food_pickup * problem.packages[0].weight) +
                               (trip.perishable_food_pickup * problem.packages[1].weight) +
                               (trip.other_supplies_pickup * problem.packages[2].weight);
            
            // Check weight capacity
            if (tripWeight > heli.weight_capacity) {
                feasible = false;
            }
            
            // Build route to calculate distance
            vector<int> route;
            for (const auto& drop : trip.drops) {
                route.push_back(drop.village_id);
            }
            
            // Calculate trip distance
            double tripDistance = calculateTripDistance(problem.cities, problem.villages, homeCity, route);
            heliTotalDistances[heliIdx] += tripDistance;
            
            // Check trip distance capacity
            if (tripDistance > heli.distance_capacity) {
                feasible = false;
            }
            
            // Calculate cost
            double tripCost = heli.fixed_cost + (heli.alpha * tripDistance);
            totalCost += tripCost;
            
            // Calculate value from drops
            int totalDryDropped = 0, totalPerishableDropped = 0, totalOtherDropped = 0;
            double tripValue = 0.0;
            
            for (const auto& drop : trip.drops) {
                totalDryDropped += drop.dry_food;
                totalPerishableDropped += drop.perishable_food;
                totalOtherDropped += drop.other_supplies;
                
                tripValue += calculateDeliveryValue(
                    drop.village_id, drop.dry_food, drop.perishable_food, drop.other_supplies,
                    problem.villages, problem.packages, deliveryTracker
                );
            }
            
            // Check that we don't drop more than we picked up
            if (totalDryDropped > trip.dry_food_pickup || 
                totalPerishableDropped > trip.perishable_food_pickup || 
                totalOtherDropped > trip.other_supplies_pickup) {
                feasible = false;
            }
            
            totalValue += tripValue;
        }
        
        // Check DMax
        if (heliTotalDistances[heliIdx] > problem.d_max) {
            feasible = false;
        }
    }
    
    double score = totalValue - totalCost;
    return {score, feasible};
}

/**
 * @brief Generate an initial greedy solution
 * @param problem The problem data
 * @return An initial solution
 */
Solution generateInitialSolution(const ProblemData& problem) {
    Solution solution;
    
    // Random number generator for generating diverse initial solutions
    random_device rd;
    mt19937 gen(rd());
    
    // Calculate the needs of each village
    vector<pair<int, double>> villageNeeds; // (village_id, need_score)
    for (const auto& village : problem.villages) {
        double foodNeeded = village.population * 9.0;
        double otherNeeded = village.population * 1.0;
        
        // Simple heuristic: prioritize villages with larger populations
        double needScore = foodNeeded * max(problem.packages[0].value, problem.packages[1].value) + 
                         otherNeeded * problem.packages[2].value;
        villageNeeds.push_back({village.id, needScore});
    }
    
    // Sort villages by need score (descending)
    sort(villageNeeds.begin(), villageNeeds.end(), 
         [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Create empty plans for all helicopters
    for (const auto& helicopter : problem.helicopters) {
        solution.push_back({helicopter.id, {}});
    }
    
    // Track what's been delivered to each village
    unordered_map<int, pair<double, double>> deliveryTracker; // village_id -> (food, other)
    
    // Sort helicopters by cost efficiency (ascending)
    vector<int> sortedHelis;
    for (int i = 0; i < problem.helicopters.size(); ++i) {
        sortedHelis.push_back(i);
    }
    sort(sortedHelis.begin(), sortedHelis.end(), [&problem](int a, int b) {
        return problem.helicopters[a].alpha < problem.helicopters[b].alpha;
    });
    
    // Keep track of helicopter's remaining distance capacity
    vector<double> heliRemainingDistance(problem.helicopters.size(), problem.d_max);
    
    // For each village, try to satisfy its needs with the most efficient helicopter
    for (const auto& [villageId, _] : villageNeeds) {
        // Get village data
        const Village& village = problem.villages[villageId - 1];
        double foodNeeded = village.population * 9.0;
        double otherNeeded = village.population * 1.0;
        
        // Get existing deliveries
        double& existingFood = deliveryTracker[villageId].first;
        double& existingOther = deliveryTracker[villageId].second;
        
        double foodRoomLeft = max(0.0, foodNeeded - existingFood);
        double otherRoomLeft = max(0.0, otherNeeded - existingOther);
        
        if (foodRoomLeft <= 0 && otherRoomLeft <= 0) continue; // Village fully satisfied
        
        // Try each helicopter, starting with the most cost-efficient
        for (int heliIdx : sortedHelis) {
            const Helicopter& heli = problem.helicopters[heliIdx];
            Point homeCity = problem.cities[heli.home_city_id - 1];
            const Point& villagePos = village.coords;
            
            // Calculate round-trip distance to village
            double tripDistance = calculateDistance(homeCity, villagePos) + calculateDistance(villagePos, homeCity);
            
            // Check if helicopter can make the trip
            if (tripDistance > heli.distance_capacity || tripDistance > heliRemainingDistance[heliIdx]) {
                continue; // This helicopter can't reach the village
            }
            
            // Calculate how much we can deliver in one trip
            double remainingWeightCapacity = heli.weight_capacity;
            
            // More aggressive strategy: try to deliver maximum amount of food and supplies
            
            // Calculate what percentage of remaining capacity to use (80-100%)
            uniform_real_distribution<> capUsageDist(0.8, 1.0);
            double capacityUsageRatio = capUsageDist(gen);
            double capacityToUse = remainingWeightCapacity * capacityUsageRatio;
            
            // Prioritize perishable food (higher value)
            int perishableToDeliver = min((int)(foodRoomLeft), 
                                        (int)(capacityToUse * 0.6 / problem.packages[1].weight));
            remainingWeightCapacity -= perishableToDeliver * problem.packages[1].weight;
            
            // Then dry food
            int dryToDeliver = min((int)(foodRoomLeft - perishableToDeliver), 
                                 (int)(remainingWeightCapacity * 0.7 / problem.packages[0].weight));
            remainingWeightCapacity -= dryToDeliver * problem.packages[0].weight;
            
            // Finally other supplies
            int otherToDeliver = min((int)(otherRoomLeft), 
                                   (int)(remainingWeightCapacity / problem.packages[2].weight));
            
            // Create a trip if we're delivering anything
            if (perishableToDeliver > 0 || dryToDeliver > 0 || otherToDeliver > 0) {
                Trip trip;
                trip.dry_food_pickup = dryToDeliver;
                trip.perishable_food_pickup = perishableToDeliver;
                trip.other_supplies_pickup = otherToDeliver;
                
                Drop drop;
                drop.village_id = villageId;
                drop.dry_food = dryToDeliver;
                drop.perishable_food = perishableToDeliver;
                drop.other_supplies = otherToDeliver;
                
                trip.drops.push_back(drop);
                
                // Add to solution
                solution[heliIdx].trips.push_back(trip);
                
                // Update tracking
                existingFood += dryToDeliver + perishableToDeliver;
                existingOther += otherToDeliver;
                heliRemainingDistance[heliIdx] -= tripDistance;
                
                // Village satisfied for now, go to next village
                break;
            }
        }
    }
    
    return solution;
}

/**
 * @brief Local search optimization of a solution using various neighborhood operators
 * @param initialSolution The initial solution to optimize
 * @param problem The problem data
 * @param maxIterations Maximum number of iterations (0 for unlimited)
 * @return An optimized solution
 */
Solution localSearchOptimization(const Solution& initialSolution, const ProblemData& problem, int maxIterations = 0) {
    Solution bestSolution = initialSolution;
    auto [bestScore, bestFeasible] = evaluateSolution(bestSolution, problem);
    
    Solution currentSolution = bestSolution;
    double currentScore = bestScore;
    bool currentFeasible = bestFeasible;
    
    // Random number generator
    random_device rd;
    mt19937 gen(rd());
    
    // Parameters for simulated annealing
    double initialTemp = 1000.0;  // Increased temperature for more exploration
    double coolingRate = 0.98;    // Slower cooling for more thorough exploration
    double temp = initialTemp;
    int noImprovementCount = 0;
    int maxNoImprovement = 500;  // Allow more iterations without improvement
    
    // Save the best feasible solution separately
    Solution bestFeasibleSolution;
    double bestFeasibleScore = -numeric_limits<double>::max();
    bool hasFeasibleSolution = false;
    
    // Problem size-based parameters
    int problemSize = problem.villages.size() + problem.helicopters.size();
    int reheatInterval = min(500, max(100, problemSize * 2));
    
    cout << "Initial solution score: " << bestScore << ", feasible: " << bestFeasible << endl;
    
    // Track number of iterations for logging
    int iteration = 0;
    
    // Perform local search until time runs out, max iterations reached, or no improvement for too many iterations
    while (hasEnoughTime(5000) && noImprovementCount < maxNoImprovement && 
           (maxIterations == 0 || iteration < maxIterations)) {
        iteration++;
        
        // Copy current solution to create neighbor
        Solution neighborSolution = currentSolution;
        
        // Choose a random neighborhood operator
        uniform_int_distribution<> opDist(0, 5);
        int operator_choice = opDist(gen);
        
        // Apply the chosen operator to generate a neighbor solution
        switch (operator_choice) {
            case 0: {
                // Add a new trip to a random helicopter
                if (!neighborSolution.empty()) {
                    uniform_int_distribution<> heliDist(0, neighborSolution.size() - 1);
                    int heliIdx = heliDist(gen);
                    
                    // Choose a random village to visit
                    uniform_int_distribution<> villageDist(0, problem.villages.size() - 1);
                    int villageId = problem.villages[villageDist(gen)].id;
                    
                    // Create a new trip with some packages
                    Trip newTrip;
                    // Simple heuristic: take a random amount of packages within capacity
                    const Helicopter& heli = problem.helicopters[neighborSolution[heliIdx].helicopter_id - 1];
                    double remainingWeight = heli.weight_capacity;
                    
                    uniform_int_distribution<> packageDist(0, (int)(remainingWeight / problem.packages[0].weight / 3));
                    newTrip.dry_food_pickup = packageDist(gen);
                    remainingWeight -= newTrip.dry_food_pickup * problem.packages[0].weight;
                    
                    packageDist = uniform_int_distribution<>(0, (int)(remainingWeight / problem.packages[1].weight / 2));
                    newTrip.perishable_food_pickup = packageDist(gen);
                    remainingWeight -= newTrip.perishable_food_pickup * problem.packages[1].weight;
                    
                    packageDist = uniform_int_distribution<>(0, (int)(remainingWeight / problem.packages[2].weight));
                    newTrip.other_supplies_pickup = packageDist(gen);
                    
                    // Create a drop
                    Drop newDrop;
                    newDrop.village_id = villageId;
                    newDrop.dry_food = newTrip.dry_food_pickup;
                    newDrop.perishable_food = newTrip.perishable_food_pickup;
                    newDrop.other_supplies = newTrip.other_supplies_pickup;
                    
                    newTrip.drops.push_back(newDrop);
                    
                    // Add the trip if it's not empty
                    if (newTrip.dry_food_pickup > 0 || newTrip.perishable_food_pickup > 0 || newTrip.other_supplies_pickup > 0) {
                        neighborSolution[heliIdx].trips.push_back(newTrip);
                    }
                }
                break;
            }
            case 1: {
                // Remove a random trip
                for (auto& plan : neighborSolution) {
                    if (!plan.trips.empty()) {
                        uniform_int_distribution<> tripDist(0, plan.trips.size() - 1);
                        int tripIdx = tripDist(gen);
                        plan.trips.erase(plan.trips.begin() + tripIdx);
                        break;
                    }
                }
                break;
            }
            case 2: {
                // Modify a trip by changing package quantities
                for (auto& plan : neighborSolution) {
                    if (!plan.trips.empty()) {
                        uniform_int_distribution<> tripDist(0, plan.trips.size() - 1);
                        int tripIdx = tripDist(gen);
                        
                        // Get the helicopter's weight capacity
                        const Helicopter& heli = problem.helicopters[plan.helicopter_id - 1];
                        
                        // Make more aggressive adjustments to package quantities
                        // Higher multiplier for perishable food (which has higher value)
                        uniform_int_distribution<> adjustDist(-100, 200);
                        plan.trips[tripIdx].dry_food_pickup = max(0, plan.trips[tripIdx].dry_food_pickup + adjustDist(gen));
                        
                        uniform_int_distribution<> adjustDistP(-50, 300); // More positive bias for perishable
                        plan.trips[tripIdx].perishable_food_pickup = max(0, plan.trips[tripIdx].perishable_food_pickup + adjustDistP(gen));
                        
                        uniform_int_distribution<> adjustDistO(-80, 150);
                        plan.trips[tripIdx].other_supplies_pickup = max(0, plan.trips[tripIdx].other_supplies_pickup + adjustDistO(gen));
                        
                        // Make sure drops don't exceed pickups
                        for (auto& drop : plan.trips[tripIdx].drops) {
                            drop.dry_food = min(drop.dry_food, plan.trips[tripIdx].dry_food_pickup);
                            drop.perishable_food = min(drop.perishable_food, plan.trips[tripIdx].perishable_food_pickup);
                            drop.other_supplies = min(drop.other_supplies, plan.trips[tripIdx].other_supplies_pickup);
                        }
                        break;
                    }
                }
                break;
            }
            case 3: {
                // Swap two villages between trips
                // Find two plans with trips
                vector<pair<int, int>> plansWithTrips; // (plan_idx, trip_idx)
                for (int i = 0; i < neighborSolution.size(); ++i) {
                    for (int j = 0; j < neighborSolution[i].trips.size(); ++j) {
                        if (!neighborSolution[i].trips[j].drops.empty()) {
                            plansWithTrips.push_back({i, j});
                        }
                    }
                }
                
                if (plansWithTrips.size() >= 2) {
                    uniform_int_distribution<> planDist(0, plansWithTrips.size() - 1);
                    int idx1 = planDist(gen);
                    int idx2;
                    do {
                        idx2 = planDist(gen);
                    } while (idx2 == idx1);
                    
                    auto [planIdx1, tripIdx1] = plansWithTrips[idx1];
                    auto [planIdx2, tripIdx2] = plansWithTrips[idx2];
                    
                    if (!neighborSolution[planIdx1].trips[tripIdx1].drops.empty() && 
                        !neighborSolution[planIdx2].trips[tripIdx2].drops.empty()) {
                        
                        uniform_int_distribution<> dropDist1(0, neighborSolution[planIdx1].trips[tripIdx1].drops.size() - 1);
                        uniform_int_distribution<> dropDist2(0, neighborSolution[planIdx2].trips[tripIdx2].drops.size() - 1);
                        
                        int dropIdx1 = dropDist1(gen);
                        int dropIdx2 = dropDist2(gen);
                        
                        swap(
                            neighborSolution[planIdx1].trips[tripIdx1].drops[dropIdx1],
                            neighborSolution[planIdx2].trips[tripIdx2].drops[dropIdx2]
                        );
                        
                        // Update package pickups
                        for (auto& plan : {
                            &neighborSolution[planIdx1].trips[tripIdx1],
                            &neighborSolution[planIdx2].trips[tripIdx2]
                        }) {
                            int totalDry = 0, totalPerishable = 0, totalOther = 0;
                            for (const auto& drop : plan->drops) {
                                totalDry += drop.dry_food;
                                totalPerishable += drop.perishable_food;
                                totalOther += drop.other_supplies;
                            }
                            plan->dry_food_pickup = totalDry;
                            plan->perishable_food_pickup = totalPerishable;
                            plan->other_supplies_pickup = totalOther;
                        }
                    }
                }
                break;
            }
            case 4: {
                // Transfer a village from one trip to another
                vector<pair<int, int>> plansWithTrips; // (plan_idx, trip_idx)
                for (int i = 0; i < neighborSolution.size(); ++i) {
                    for (int j = 0; j < neighborSolution[i].trips.size(); ++j) {
                        if (!neighborSolution[i].trips[j].drops.empty()) {
                            plansWithTrips.push_back({i, j});
                        }
                    }
                }
                
                if (plansWithTrips.size() >= 2) {
                    uniform_int_distribution<> planDist(0, plansWithTrips.size() - 1);
                    int srcIdx = planDist(gen);
                    int destIdx;
                    do {
                        destIdx = planDist(gen);
                    } while (destIdx == srcIdx);
                    
                    auto [srcPlanIdx, srcTripIdx] = plansWithTrips[srcIdx];
                    auto [destPlanIdx, destTripIdx] = plansWithTrips[destIdx];
                    
                    if (!neighborSolution[srcPlanIdx].trips[srcTripIdx].drops.empty()) {
                        uniform_int_distribution<> dropDist(0, neighborSolution[srcPlanIdx].trips[srcTripIdx].drops.size() - 1);
                        int dropIdx = dropDist(gen);
                        
                        // Move drop to destination trip
                        neighborSolution[destPlanIdx].trips[destTripIdx].drops.push_back(
                            neighborSolution[srcPlanIdx].trips[srcTripIdx].drops[dropIdx]
                        );
                        neighborSolution[srcPlanIdx].trips[srcTripIdx].drops.erase(
                            neighborSolution[srcPlanIdx].trips[srcTripIdx].drops.begin() + dropIdx
                        );
                        
                        // Update package pickups for both trips
                        for (auto& plan : {
                            &neighborSolution[srcPlanIdx].trips[srcTripIdx],
                            &neighborSolution[destPlanIdx].trips[destTripIdx]
                        }) {
                            int totalDry = 0, totalPerishable = 0, totalOther = 0;
                            for (const auto& drop : plan->drops) {
                                totalDry += drop.dry_food;
                                totalPerishable += drop.perishable_food;
                                totalOther += drop.other_supplies;
                            }
                            plan->dry_food_pickup = totalDry;
                            plan->perishable_food_pickup = totalPerishable;
                            plan->other_supplies_pickup = totalOther;
                        }
                    }
                }
                break;
            }
            case 5: {
                // Optimize a trip's route using 2-opt
                for (auto& plan : neighborSolution) {
                    if (!plan.trips.empty()) {
                        uniform_int_distribution<> tripDist(0, plan.trips.size() - 1);
                        int tripIdx = tripDist(gen);
                        
                        if (plan.trips[tripIdx].drops.size() >= 4) {
                            uniform_int_distribution<> indexDist(0, plan.trips[tripIdx].drops.size() - 1);
                            int i = indexDist(gen);
                            int j;
                            do {
                                j = indexDist(gen);
                            } while (abs(i-j) <= 1 || i == j);
                            
                            if (i > j) swap(i, j);
                            
                            // Reverse the portion of the route between i and j
                            reverse(
                                plan.trips[tripIdx].drops.begin() + i,
                                plan.trips[tripIdx].drops.begin() + j + 1
                            );
                        }
                    }
                }
                break;
            }
        }
        
        // Evaluate the neighbor solution
        auto [neighborScore, neighborFeasible] = evaluateSolution(neighborSolution, problem);
        
        // Decide whether to accept the neighbor
        bool accept = false;
        
        // Always accept if it's better
        if (neighborScore > currentScore || (neighborFeasible && !currentFeasible)) {
            accept = true;
        } else {
            // Accept with some probability based on temperature (simulated annealing)
            // Modified to be more accepting of worse solutions to explore more
            double scoreDiff = neighborScore - currentScore;
            double acceptProb = exp(scoreDiff / temp);
            
            // Increase acceptance probability for feasible solutions 
            if (neighborFeasible && !currentFeasible) {
                acceptProb = min(1.0, acceptProb * 2.0);
            }
            
            uniform_real_distribution<> probDist(0.0, 1.0);
            accept = probDist(gen) < acceptProb;
        }
        
        if (accept) {
            currentSolution = neighborSolution;
            currentScore = neighborScore;
            currentFeasible = neighborFeasible;
            
            // Update best solution if this one is better
            if ((currentFeasible && !bestFeasible) || 
                (currentFeasible == bestFeasible && currentScore > bestScore)) {
                bestSolution = currentSolution;
                bestScore = currentScore;
                bestFeasible = currentFeasible;
                noImprovementCount = 0;
                
                cout << "Iteration " << iteration << ": New best score " << bestScore 
                     << ", feasible: " << bestFeasible << endl;
            } else {
                noImprovementCount++;
            }
        } else {
            noImprovementCount++;
        }
        
        // Cool down temperature
        temp *= coolingRate;
        
        // Periodically increase temperature to escape local optima (restart mechanism)
        if (iteration % reheatInterval == 0) {
            // More aggressive reheating when stuck
            if (noImprovementCount > 300) {
                temp = initialTemp; // Full reheat
                cout << "Full temperature reheat to escape local optimum" << endl;
            } 
            else if (noImprovementCount > 100) {
                temp = initialTemp * 0.7;
                cout << "Temperature increased to escape local optimum" << endl;
            }
        }
        
        // Save best feasible solution separately
        if (currentFeasible) {
            if (!hasFeasibleSolution || currentScore > bestFeasibleScore) {
                bestFeasibleSolution = currentSolution;
                bestFeasibleScore = currentScore;
                hasFeasibleSolution = true;
            }
        }
        
        // Periodically log progress
        if (iteration % 500 == 0) {
            auto elapsed = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - start_time);
            cout << "Iteration " << iteration 
                 << ", time elapsed: " << elapsed.count() / 1000.0 << "s" 
                 << ", best score: " << bestScore 
                 << ", feasible: " << bestFeasible << endl;
        }
    }
    
    // Return the best feasible solution if available, otherwise return the best solution found
    if (hasFeasibleSolution) {
        cout << "Final solution score: " << bestFeasibleScore << ", feasible: 1"
             << ", after " << iteration << " iterations" << endl;
        return bestFeasibleSolution;
    } else {
        cout << "Final solution score: " << bestScore << ", feasible: " << bestFeasible 
             << ", after " << iteration << " iterations" << endl;
        return bestSolution;
    }
}

/**
 * @brief Generate a random solution for random restart
 * @param problem The problem data
 * @param gen Random number generator
 * @return A randomly generated solution
 */
Solution generateRandomSolution(const ProblemData& problem, mt19937& gen) {
    Solution solution;
    
    // Create empty plans for all helicopters
    for (const auto& helicopter : problem.helicopters) {
        solution.push_back({helicopter.id, {}});
    }
    
    // Random distributions
    uniform_int_distribution<> heliDist(0, problem.helicopters.size() - 1);
    uniform_int_distribution<> villageDist(0, problem.villages.size() - 1);
    uniform_int_distribution<> numTripsDist(1, min(10, (int)(problem.villages.size())));
    uniform_int_distribution<> numDropsDist(1, min(5, (int)(problem.villages.size())));
    uniform_real_distribution<> ratioThreshold(0.0, 1.0);
    
    // For each helicopter, create random trips
    for (int h = 0; h < problem.helicopters.size(); ++h) {
        const Helicopter& heli = problem.helicopters[h];
        
        // Decide how many trips this helicopter will make
        int numTrips = numTripsDist(gen);
        
        for (int t = 0; t < numTrips; ++t) {
            // 30% chance to skip this trip
            if (ratioThreshold(gen) < 0.3) continue;
            
            Trip trip;
            
            // Decide package quantities - randomize but biased toward perishable
            double weight_capacity = heli.weight_capacity * ratioThreshold(gen);
            
            // Ratio of each type of package - prioritize perishable
            double perishableRatio = ratioThreshold(gen) * 0.7 + 0.2; // 0.2 to 0.9
            double dryRatio = (1.0 - perishableRatio) * (ratioThreshold(gen) * 0.7 + 0.2); // 0.2 to 0.9 of remaining
            double otherRatio = 1.0 - perishableRatio - dryRatio;
            
            double perishableWeight = weight_capacity * perishableRatio;
            double dryWeight = weight_capacity * dryRatio;
            double otherWeight = weight_capacity * otherRatio;
            
            int perishablePkgs = perishableWeight / problem.packages[1].weight;
            int dryPkgs = dryWeight / problem.packages[0].weight;
            int otherPkgs = otherWeight / problem.packages[2].weight;
            
            trip.perishable_food_pickup = perishablePkgs;
            trip.dry_food_pickup = dryPkgs;
            trip.other_supplies_pickup = otherPkgs;
            
            // Generate drops
            int numDrops = numDropsDist(gen);
            numDrops = min(numDrops, (int)problem.villages.size());
            
            // Track which villages have been visited in this trip
            vector<bool> visited(problem.villages.size(), false);
            
            int totalDry = 0, totalPerishable = 0, totalOther = 0;
            
            for (int d = 0; d < numDrops; ++d) {
                // Select a random unvisited village
                int villageIdx;
                do {
                    villageIdx = villageDist(gen);
                } while (visited[villageIdx]);
                
                visited[villageIdx] = true;
                
                Drop drop;
                drop.village_id = problem.villages[villageIdx].id;
                
                // Distribute packages randomly
                uniform_int_distribution<> dryDist(0, max(0, trip.dry_food_pickup - totalDry));
                uniform_int_distribution<> perishableDist(0, max(0, trip.perishable_food_pickup - totalPerishable));
                uniform_int_distribution<> otherDist(0, max(0, trip.other_supplies_pickup - totalOther));
                
                // Ensure we leave some packages for other drops if needed
                int maxDropRatio = (d == numDrops - 1) ? 1 : 3; // Use all remaining on last drop
                
                drop.dry_food = dryDist(gen) / maxDropRatio;
                drop.perishable_food = perishableDist(gen) / maxDropRatio;
                drop.other_supplies = otherDist(gen) / maxDropRatio;
                
                // Ensure we have something to drop
                if (drop.dry_food == 0 && drop.perishable_food == 0 && drop.other_supplies == 0) {
                    if (trip.dry_food_pickup > totalDry) drop.dry_food = 1;
                    else if (trip.perishable_food_pickup > totalPerishable) drop.perishable_food = 1;
                    else if (trip.other_supplies_pickup > totalOther) drop.other_supplies = 1;
                }
                
                totalDry += drop.dry_food;
                totalPerishable += drop.perishable_food;
                totalOther += drop.other_supplies;
                
                trip.drops.push_back(drop);
                
                // Stop if we've used all packages
                if (totalDry >= trip.dry_food_pickup && 
                    totalPerishable >= trip.perishable_food_pickup && 
                    totalOther >= trip.other_supplies_pickup) {
                    break;
                }
            }
            
            // Adjust pickups to match drops to ensure feasibility
            trip.dry_food_pickup = totalDry;
            trip.perishable_food_pickup = totalPerishable;
            trip.other_supplies_pickup = totalOther;
            
            // Only add trip if it has drops
            if (!trip.drops.empty()) {
                solution[h].trips.push_back(trip);
            }
        }
    }
    
    return solution;
}

/**
 * @brief The main function to implement your search/optimization algorithm with random restarts.
 */
Solution solve(const ProblemData& problem) {
    cout << "Starting solver..." << endl;

    // Initialize time tracking
    start_time = chrono::steady_clock::now();
    // Set time limit to 95% of the available time to ensure we return before deadline
    time_limit = chrono::milliseconds(long(problem.time_limit_minutes * 60 * 1000 * 0.95));
    
    cout << "Time limit: " << problem.time_limit_minutes << " minutes" << endl;
    cout << "Will optimize until: " << (problem.time_limit_minutes * 60 * 0.95) << " seconds" << endl;
    
    // Random number generator for consistent results
    random_device rd;
    mt19937 gen(rd());
    
    Solution bestSolution;
    double bestScore = -numeric_limits<double>::max();
    bool bestFeasible = false;
    
    // We'll now optimize until we run out of time instead of a fixed number of restarts
    int restartsAttempted = 0;
    
    // Dynamic iteration allocation based on problem size
    int baseIterations = 100;
    int iterationScaleFactor = 10;  
    int problemSize = problem.villages.size() + problem.helicopters.size();
    
    // Keep track of time to ensure we don't exceed the limit
    while (hasEnoughTime(5000)) {
        restartsAttempted++;
        
        // Dynamically adjust iterations based on how much time we've used
        auto currentTime = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(currentTime - start_time);
        double elapsedSeconds = elapsed.count() / 1000.0;
        double totalSeconds = problem.time_limit_minutes * 60 * 0.95;
        double timeRemainingRatio = (totalSeconds - elapsedSeconds) / totalSeconds;
        
        // Every 10th restart, use the greedy solution as the starting point
        // Others use random solutions for diversity
        Solution initialSolution;
        if (restartsAttempted % 10 == 1) {
            initialSolution = generateInitialSolution(problem);
            cout << "Restart " << restartsAttempted << ": Using greedy initial solution" << endl;
        } else {
            initialSolution = generateRandomSolution(problem, gen);
            cout << "Restart " << restartsAttempted << ": Using random initial solution" << endl;
        }
        
        // Calculate dynamic iterations - more iterations when we have more time left
        // and fewer as we approach the time limit
        int iterationsForThisRestart;
        if (restartsAttempted < 100) {
            // In early stages, use shorter runs to explore diverse starting points
            iterationsForThisRestart = baseIterations + (int)(iterationScaleFactor * problemSize * timeRemainingRatio);
        } else if (restartsAttempted < 1000) {
            // In middle stages, balance exploration and exploitation
            iterationsForThisRestart = baseIterations * 2 + (int)(iterationScaleFactor * problemSize * timeRemainingRatio);
        } else {
            // In later stages, do more exploitation with longer runs
            iterationsForThisRestart = baseIterations * 4 + (int)(iterationScaleFactor * problemSize * timeRemainingRatio);
        }
        
        // Cap the iterations to avoid excessively long runs
        iterationsForThisRestart = min(iterationsForThisRestart, 2000);
        
        // Optimize the solution using local search with calculated iterations
        auto [currentScore, currentFeasible] = evaluateSolution(initialSolution, problem);
        cout << "Initial solution score: " << currentScore << ", feasible: " << currentFeasible << endl;
        
        Solution optimizedSolution = localSearchOptimization(initialSolution, problem, iterationsForThisRestart);
        auto [optimizedScore, optimizedFeasible] = evaluateSolution(optimizedSolution, problem);
        
        cout << "After local search: score " << optimizedScore << ", feasible: " << optimizedFeasible << endl;
        
        // Update best solution if this one is better
        if ((optimizedFeasible && !bestFeasible) || 
            (optimizedFeasible == bestFeasible && optimizedScore > bestScore)) {
            bestSolution = optimizedSolution;
            bestScore = optimizedScore;
            bestFeasible = optimizedFeasible;
            cout << "New best solution found with score: " << bestScore << ", feasible: " << bestFeasible << endl;
        }
        
        // Print progress every 10 restarts
        if (restartsAttempted % 10 == 0) {
            double elapsedMinutes = elapsed.count() / 60000.0;
            double remainingMinutes = problem.time_limit_minutes * 0.95 - elapsedMinutes;
            cout << "Progress: " << restartsAttempted << " restarts, " 
                 << elapsedMinutes << " minutes elapsed, " 
                 << remainingMinutes << " minutes remaining" << endl;
        }
        
        cout << "Completed restart " << restartsAttempted 
             << ", time elapsed: " << elapsedSeconds << "s" << endl;
    }
    
    // Final log
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now() - start_time);
    cout << "Solver finished after " << restartsAttempted << " restarts." << endl;
    cout << "Total time: " << elapsed.count() / 1000.0 << "s ("
         << (elapsed.count() / 60000.0) << " minutes)" << endl;
    cout << "Best score: " << bestScore << ", feasible: " << bestFeasible << endl;
    
    return bestSolution;
}