# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3

# Executable names
EXEC = main
CHECKER_EXEC = format_checker
TEST_EXEC = test

# Source files for the main solver
MAIN_SRCS = main.cpp io_handler.cpp solver.cpp

# Source file for the checker
CHECKER_SRCS = format_checker.cpp

# Source file for the test
TEST_SRCS = test.cpp

# Object files
MAIN_OBJS = $(MAIN_SRCS:.cpp=.o)
CHECKER_OBJS = $(CHECKER_SRCS:.cpp=.o)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)

# Default target - build everything
all: $(EXEC) $(CHECKER_EXEC) $(TEST_EXEC)

# Rule to link the main solver program
$(EXEC): $(MAIN_OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(MAIN_OBJS)

# Rule to build the checker executable
$(CHECKER_EXEC): $(CHECKER_OBJS) io_handler.o
	$(CXX) $(CXXFLAGS) -o $(CHECKER_EXEC) $(CHECKER_OBJS) io_handler.o

# Rule to build the test executable
$(TEST_EXEC): $(TEST_OBJS) io_handler.o solver.o
	$(CXX) $(CXXFLAGS) -o $(TEST_EXEC) $(TEST_OBJS) io_handler.o solver.o

# Generic rule to compile .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Specific dependencies for object files
main.o: main.cpp structures.h solver.h io_handler.h
io_handler.o: io_handler.cpp io_handler.h structures.h
solver.o: solver.cpp solver.h structures.h 
format_checker.o: format_checker.cpp structures.h io_handler.h
test.o: test.cpp structures.h io_handler.h

# Clean up build files
clean:
	rm -f $(EXEC) $(CHECKER_EXEC) $(TEST_EXEC) $(MAIN_OBJS) $(CHECKER_OBJS) $(TEST_OBJS)
	rm -f temp_format_checker_output.txt statistics.txt plot_data.csv objective_values.png
	rm -rf input output *.txt


# Create plot of results
plot:
	python3 plot_results.py

.PHONY: all clean distclean prepare run runtest check plot