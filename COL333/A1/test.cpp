#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <algorithm>

namespace fs = std::filesystem;

// Random number generation utilities
std::random_device rd;
std::mt19937 gen(rd());

// Function to generate random float in a range
float randomFloat(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

// Function to generate random integer in a range
int randomInt(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

// Structure to define parameter ranges for different difficulty levels
struct DifficultyParams {
    int minCities, maxCities;
    int minVillages, maxVillages;
    int minHelicopters, maxHelicopters;
    int minStranded, maxStranded;
    float minDMax, maxDMax;
    float minWeightCap, maxWeightCap;
    float minDistCap, maxDistCap;
};

// Structure to store test case results
struct TestResult {
    std::string inputFile;
    std::string outputFile;
    bool success;
    double executionTime;
    double objectiveValue;
    std::string errorMessage;
};

DifficultyParams getDifficultyParams(const std::string& difficulty) {
    DifficultyParams params;
    
    if (difficulty == "easy") {
        params.minCities = 2;
        params.maxCities = 10;
        params.minVillages = 3;
        params.maxVillages = 7;
        params.minHelicopters = 2;
        params.maxHelicopters = 4;
        params.minStranded = 100;
        params.maxStranded = 1000;
        params.minDMax = 100;
        params.maxDMax = 200;
        params.minWeightCap = 80;
        params.maxWeightCap = 150;
        params.minDistCap = 25;
        params.maxDistCap = 60;
    } else if (difficulty == "medium") {
        params.minCities = 3;
        params.maxCities = 8;
        params.minVillages = 8;
        params.maxVillages = 15;
        params.minHelicopters = 3;
        params.maxHelicopters = 8;
        params.minStranded = 500;
        params.maxStranded = 3000;
        params.minDMax = 150;
        params.maxDMax = 400;
        params.minWeightCap = 100;
        params.maxWeightCap = 250;
        params.minDistCap = 40;
        params.maxDistCap = 100;
    } else { // hard
        params.minCities = 5;
        params.maxCities = 25;          // Updated based on spec: up to 25 cities
        params.minVillages = 15;
        params.maxVillages = 1200;      // Updated based on spec: up to 1200 villages
        params.minHelicopters = 5;
        params.maxHelicopters = 40;     // Updated based on spec: up to 40 helicopters
        params.minStranded = 1000;
        params.maxStranded = 10000;
        params.minDMax = 300;
        params.maxDMax = 800;
        params.minWeightCap = 150;
        params.maxWeightCap = 500;
        params.minDistCap = 60;
        params.maxDistCap = 200;
    }
    
    return params;
}

// Create directories if they don't exist
void ensureDirectoriesExist() {
    if (!fs::exists("input")) {
        fs::create_directory("input");
    }
    if (!fs::exists("output")) {
        fs::create_directory("output");
    }
}

// Function to generate a test case
std::string generateTestCase(const std::string& difficulty, int testCaseNum) {
    std::string filename = "input/" + difficulty + "_" + std::to_string(testCaseNum) + ".txt";
    std::ofstream outfile(filename);
    
    // Get parameters for the specified difficulty level
    DifficultyParams params = getDifficultyParams(difficulty);
    
    // Time limit (in minutes) - adjust based on difficulty
    // For hard difficulty, use 5 minutes as specified
    int timeLimit = (difficulty == "easy") ? 1 : ((difficulty == "medium") ? 2 : 5);
    outfile << timeLimit << std::endl;
    
    // DMax - maximum total distance any helicopter can travel
    float DMax = randomFloat(params.minDMax, params.maxDMax);
    outfile << DMax << std::endl;
    
    // Packet parameters: w(d) v(d) w(p) v(p) w(o) v(o)
    float w_d = randomFloat(0.01, 0.05);
    float v_d = randomFloat(0.8, 1.2);
    float w_p = randomFloat(0.05, 0.2);
    float v_p = randomFloat(1.5, 2.5); // Perishable is more valuable
    float w_o = randomFloat(0.005, 0.02);
    float v_o = randomFloat(0.1, 0.3);
    outfile << w_d << " " << v_d << " " << w_p << " " << v_p << " " << w_o << " " << v_o << std::endl;
    
    // Cities
    int numCities = randomInt(params.minCities, params.maxCities);
    outfile << numCities;
    
    // Generate city coordinates
    float mapSize = (difficulty == "easy") ? 20 : ((difficulty == "medium") ? 50 : 100);
    std::vector<std::pair<float, float>> cityCoords;
    
    for (int i = 0; i < numCities; i++) {
        float x = randomFloat(0, mapSize);
        float y = randomFloat(0, mapSize);
        cityCoords.push_back({x, y});
        outfile << " " << x << " " << y;
    }
    outfile << std::endl;
    
    // Villages
    int numVillages = randomInt(params.minVillages, params.maxVillages);
    outfile << numVillages;
    
    // Generate village coordinates and stranded people
    for (int i = 0; i < numVillages; i++) {
        float x = randomFloat(0, mapSize);
        float y = randomFloat(0, mapSize);
        int stranded = randomInt(params.minStranded, params.maxStranded);
        outfile << " " << x << " " << y << " " << stranded;
    }
    outfile << std::endl;
    
    // Helicopters
    int numHelicopters = randomInt(params.minHelicopters, params.maxHelicopters);
    outfile << numHelicopters;
    
    for (int i = 0; i < numHelicopters; i++) {
        int homeCity = randomInt(1, numCities); // 1-indexed as per problem
        float weightCap = randomFloat(params.minWeightCap, params.maxWeightCap);
        float distCap = randomFloat(params.minDistCap, params.maxDistCap);
        float fixedCost = randomFloat(5.0, 20.0);
        float alpha = randomFloat(0.5, 2.0);
        
        outfile << " " << homeCity << " " << weightCap << " " << distCap << " " 
                << fixedCost << " " << alpha;
    }
    outfile << std::endl;
    
    outfile.close();
    std::cout << "Generated test case: " << filename << std::endl;
    return filename;
}

// Execute the main program on a given input file
TestResult runMain(const std::string& inputFile) {
    TestResult result;
    result.inputFile = inputFile;
    
    // Generate output file name
    std::string basename = fs::path(inputFile).filename().string();
    result.outputFile = "output/" + basename;
    
    std::cout << "Running main on " << inputFile << "..." << std::endl;
    
    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Execute main program - removed "./" prefix for Windows compatibility
    std::string command = "./main " + inputFile + " " + result.outputFile;
    int exitCode = std::system(command.c_str());
    
    auto end = std::chrono::high_resolution_clock::now();
    result.executionTime = std::chrono::duration<double>(end - start).count();
    
    if (exitCode != 0) {
        result.success = false;
        result.errorMessage = "Main program execution failed with exit code: " + std::to_string(exitCode);
        std::cerr << result.errorMessage << std::endl;
        return result;
    }
    
    result.success = true;
    std::cout << "Main execution completed in " << result.executionTime << " seconds" << std::endl;
    
    return result;
}

// Run the format checker on the output file
void runFormatChecker(TestResult& result) {
    std::cout << "Running format checker on " << result.outputFile << "..." << std::endl;
    
    // Create a temporary file to capture the format checker output
    std::string tempFile = "temp_format_checker_output.txt";
    // Removed "./" prefix for Windows compatibility
    std::string command = "./format_checker " + result.inputFile + " " + result.outputFile + " > " + tempFile;
    
    int exitCode = std::system(command.c_str());
    
    if (exitCode != 0) {
        result.success = false;
        result.errorMessage = "Format checker failed with exit code: " + std::to_string(exitCode);
        std::cerr << result.errorMessage << std::endl;
        return;
    }
    
    // Extract the objective value from the format checker output
    std::ifstream tempIn(tempFile);
    std::string line;
    bool foundValue = false;
    
    while (std::getline(tempIn, line)) {
        // Look for "FINAL SCORE" instead of "OBJECTIVE VALUE"
        if (line.find("FINAL SCORE:") != std::string::npos) {
            std::istringstream iss(line);
            std::string token;
            
            // Skip "FINAL" and "SCORE:"
            iss >> token >> token;
            
            // The next token should be the score value
            if (iss >> result.objectiveValue) {
                foundValue = true;
                break;
            }
        }
    }
    
    tempIn.close();
    
    // Clean up temp file
    std::remove(tempFile.c_str());
    
    if (foundValue) {
        std::cout << "Format check passed. Objective value: " << result.objectiveValue << std::endl;
    } else {
        std::cout << "Format check passed, but couldn't extract objective value." << std::endl;
        result.objectiveValue = 0.0; // Set a default value
    }
}

// Write statistics to a file
void writeStatistics(const std::vector<TestResult>& results) {
    std::ofstream statsFile("statistics.txt");
    
    statsFile << "====================================================\n";
    statsFile << "         DISASTER RELIEF HELICOPTER ROUTING         \n";
    statsFile << "               TESTING STATISTICS                   \n";
    statsFile << "====================================================\n\n";
    
    statsFile << "Tests run: " << results.size() << "\n";
    
    int successCount = 0;
    for (const auto& result : results) {
        if (result.success) successCount++;
    }
    
    statsFile << "Successful tests: " << successCount << "/" << results.size() << "\n\n";
    
    // Write statistics for each difficulty level
    const std::vector<std::string> difficultyLevels = {"easy", "medium", "hard"};
    for (const std::string& difficulty : difficultyLevels) {
        std::vector<TestResult> difficultyResults;
        for (const auto& result : results) {
            if (result.inputFile.find(difficulty) != std::string::npos) {
                difficultyResults.push_back(result);
            }
        }
        
        if (difficultyResults.empty()) continue;
        
        statsFile << "====================== " << difficulty << " ======================\n";
        statsFile << "Tests: " << difficultyResults.size() << "\n";
        
        double avgTime = 0;
        double avgObjective = 0;
        double bestObjective = -std::numeric_limits<double>::infinity();
        double worstObjective = std::numeric_limits<double>::infinity();
        
        for (const auto& result : difficultyResults) {
            if (result.success) {
                avgTime += result.executionTime;
                avgObjective += result.objectiveValue;
                bestObjective = std::max(bestObjective, result.objectiveValue);
                worstObjective = std::min(worstObjective, result.objectiveValue);
            }
        }
        
        int successCount = 0;
        for (const auto& result : difficultyResults) {
            if (result.success) successCount++;
        }
        
        if (successCount > 0) {
            avgTime /= successCount;
            avgObjective /= successCount;
            
            statsFile << "Success rate: " << successCount << "/" << difficultyResults.size() << "\n";
            statsFile << "Average execution time: " << avgTime << " seconds\n";
            statsFile << "Average objective value: " << avgObjective << "\n";
            statsFile << "Best objective value: " << bestObjective << "\n";
            statsFile << "Worst objective value: " << worstObjective << "\n";
        } else {
            statsFile << "No successful tests for this difficulty level.\n";
        }
        
        statsFile << "\n";
    }
    
    // Detailed results table
    statsFile << "=============== Detailed Results ===============\n";
    statsFile << std::left << std::setw(25) << "Input File" 
              << std::setw(15) << "Status"
              << std::setw(15) << "Time (s)"
              << std::setw(15) << "Objective" << "\n";
    statsFile << std::string(70, '-') << "\n";
    
    for (const auto& result : results) {
        std::string status = result.success ? "SUCCESS" : "FAILED";
        std::string inputFileName = fs::path(result.inputFile).filename().string();
        
        statsFile << std::left << std::setw(25) << inputFileName
                  << std::setw(15) << status;
        
        if (result.success) {
            statsFile << std::setw(15) << std::fixed << std::setprecision(3) << result.executionTime
                      << std::setw(15) << std::fixed << std::setprecision(1) << result.objectiveValue;
        } else {
            statsFile << std::setw(15) << "N/A"
                      << std::setw(15) << "N/A";
        }
        
        statsFile << "\n";
    }
    
    statsFile.close();
    std::cout << "Statistics written to statistics.txt" << std::endl;
}

// New function to process existing files in a directory
std::vector<std::string> getAllFilesInDirectory(const std::string& directoryPath) {
    std::vector<std::string> files;
    
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().string());
        }
    }
    
    // Sort files alphabetically
    std::sort(files.begin(), files.end());
    return files;
}

// Generate plot data in CSV format for visualization
void generatePlotData(const std::vector<TestResult>& results) {
    std::ofstream plotDataFile("plot_data.csv");
    plotDataFile << "TestCase,ObjectiveValue,ExecutionTime\n";
    
    for (const auto& result : results) {
        if (result.success) {
            std::string testCaseName = fs::path(result.inputFile).filename().string();
            plotDataFile << testCaseName << "," 
                        << result.objectiveValue << "," 
                        << result.executionTime << "\n";
        }
    }
    
    plotDataFile.close();
    std::cout << "Plot data written to plot_data.csv" << std::endl;
}

// Process multiple files in batch mode
std::vector<TestResult> processBatchFiles(const std::string& inputDir, const std::string& outputDir) {
    std::vector<TestResult> results;
    std::vector<std::string> inputFiles = getAllFilesInDirectory(inputDir);
    
    std::cout << "Found " << inputFiles.size() << " input files to process." << std::endl;
    
    for (const auto& inputFile : inputFiles) {
        // Extract just the filename for the output
        std::string inputBasename = fs::path(inputFile).filename().string();
        
        // Create TestResult with full paths
        TestResult result;
        result.inputFile = inputFile;
        result.outputFile = outputDir + "/" + inputBasename;
        
        std::cout << "\nProcessing: " << inputBasename << std::endl;
        
        // Run the main solver
        auto start = std::chrono::high_resolution_clock::now();
        std::string command = "./main " + result.inputFile + " " + result.outputFile;
        int exitCode = std::system(command.c_str());
        auto end = std::chrono::high_resolution_clock::now();
        
        result.executionTime = std::chrono::duration<double>(end - start).count();
        
        if (exitCode != 0) {
            result.success = false;
            result.errorMessage = "Main program execution failed with exit code: " + std::to_string(exitCode);
            std::cerr << result.errorMessage << std::endl;
        } else {
            result.success = true;
            std::cout << "Main execution completed in " << result.executionTime << " seconds" << std::endl;
            
            // Run format checker
            runFormatChecker(result);
        }
        
        results.push_back(result);
    }
    
    return results;
}

// Enhanced writeStatistics function with more detailed output
void writeEnhancedStatistics(const std::vector<TestResult>& results) {
    // Call the original statistics function
    writeStatistics(results);
    
    // Generate plot data for visualization
    generatePlotData(results);
    
    // Additional summary output to console
    std::cout << "\n=== SUMMARY STATISTICS ===\n";
    std::cout << "Total test cases: " << results.size() << std::endl;
    
    int successCount = 0;
    double totalObjective = 0;
    double bestObjective = -std::numeric_limits<double>::infinity();
    double worstObjective = std::numeric_limits<double>::infinity();
    std::string bestCase, worstCase;
    
    for (const auto& result : results) {
        if (result.success) {
            successCount++;
            totalObjective += result.objectiveValue;
            
            if (result.objectiveValue > bestObjective) {
                bestObjective = result.objectiveValue;
                bestCase = fs::path(result.inputFile).filename().string();
            }
            
            if (result.objectiveValue < worstObjective) {
                worstObjective = result.objectiveValue;
                worstCase = fs::path(result.inputFile).filename().string();
            }
        }
    }
    
    if (successCount > 0) {
        std::cout << "Successful runs: " << successCount << "/" << results.size() << std::endl;
        std::cout << "Average objective: " << (totalObjective / successCount) << std::endl;
        std::cout << "Best case: " << bestCase << " (" << bestObjective << ")" << std::endl;
        std::cout << "Worst case: " << worstCase << " (" << worstObjective << ")" << std::endl;
    } else {
        std::cout << "No successful runs." << std::endl;
    }
    
    std::cout << "Detailed results available in statistics.txt and plot_data.csv\n";
}

int main(int argc, char* argv[]) {
    // Check if we have exactly 3 arguments
    if (argc == 3) {
        // Check if the first arg is a valid difficulty level
        std::string arg1 = argv[1];
        if (arg1 == "easy" || arg1 == "medium" || arg1 == "hard") {
            // Original mode: generate test cases
            std::string difficulty = arg1;
            int numTestCases;
            
            try {
                numTestCases = std::stoi(argv[2]);
            } catch (const std::invalid_argument& e) {
                std::cout << "Error: Second argument must be a number when using difficulty level mode." << std::endl;
                std::cout << "Usage: " << argv[0] << " <difficulty> <num_test_cases>" << std::endl;
                return 1;
            }
            
            // Ensure input and output directories exist
            ensureDirectoriesExist();
            
            std::cout << "Generating " << numTestCases << " " << difficulty << " test cases..." << std::endl;
            
            std::vector<TestResult> results;
            
            for (int i = 1; i <= numTestCases; i++) {
                // Generate test case
                std::string inputFile = generateTestCase(difficulty, i);
                
                // Run main on the test case
                TestResult result = runMain(inputFile);
                
                // Run format checker if main was successful
                if (result.success) {
                    runFormatChecker(result);
                }
                
                results.push_back(result);
            }
            
            // Write enhanced statistics (this will also generate plot_data.csv)
            writeEnhancedStatistics(results);
            
        } else {
            // New batch processing mode
            std::string inputDir = argv[1];
            std::string outputDir = argv[2];
            
            // Check if directories exist
            if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
                std::cout << "Error: Input directory '" << inputDir << "' does not exist or is not a directory." << std::endl;
                return 1;
            }
            
            // Create output directory if it doesn't exist
            if (!fs::exists(outputDir)) {
                std::cout << "Creating output directory: " << outputDir << std::endl;
                fs::create_directories(outputDir);
            }
            
            std::cout << "Batch processing mode: Processing files from '" << inputDir << "' to '" << outputDir << "'" << std::endl;
            
            // Process all files in the input directory
            std::vector<TestResult> results = processBatchFiles(inputDir, outputDir);
            
            // Generate enhanced statistics
            writeEnhancedStatistics(results);
        }
    } else {
        // Display usage
        std::cout << "Usage:" << std::endl;
        std::cout << "  " << argv[0] << " <difficulty> <num_test_cases>  - Generate and run test cases" << std::endl;
        std::cout << "  " << argv[0] << " <input_dir> <output_dir>       - Process existing test files" << std::endl;
        return 1;
    }
    
    return 0;
}