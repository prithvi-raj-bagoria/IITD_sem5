# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

# Executable names
EXEC = main
CHECKER_EXEC = format_checker

# Source files for the main solver
MAIN_SRCS = main.cpp io_handler.cpp solver.cpp
ALGO_SRCS = algo/simulate.cpp algo/genetic.cpp algo/greedy.cpp

# Source file for the checker
CHECKER_SRCS = format_checker.cpp

# Object files
MAIN_OBJS = $(MAIN_SRCS:.cpp=.o)
ALGO_OBJS = $(ALGO_SRCS:.cpp=.o)
ALL_OBJS = $(MAIN_OBJS) $(ALGO_OBJS)
CHECKER_OBJS = $(CHECKER_SRCS:.cpp=.o)

# Default target
all: $(EXEC)

# Rule to link the main solver program
$(EXEC): $(ALL_OBJS)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(ALL_OBJS)

# Rule to build the checker executable
checker: $(CHECKER_EXEC)

# Checker executable depends on its own object file and io_handler.o
$(CHECKER_EXEC): $(CHECKER_OBJS) io_handler.o
	$(CXX) $(CXXFLAGS) -o $(CHECKER_EXEC) $(CHECKER_OBJS) io_handler.o

# Generic rule to compile .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Specific dependencies for object files
main.o: main.cpp structures.h solver.h io_handler.h
io_handler.o: io_handler.cpp io_handler.h structures.h
solver.o: solver.cpp solver.h structures.h algo/simulate.h algo/genetic.h algo/greedy.h
algo/simulate.o: algo/simulate.cpp algo/simulate.h structures.h
algo/genetic.o: algo/genetic.cpp algo/genetic.h structures.h
algo/greedy.o: algo/greedy.cpp algo/greedy.h structures.h
format_checker.o: format_checker.cpp structures.h io_handler.h

# Clean up build files
clean:
	rm -f $(EXEC) $(CHECKER_EXEC) $(ALL_OBJS) $(CHECKER_OBJS)

.PHONY: all clean checker