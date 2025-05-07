# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -fopenmp

# Directories
SRC_DIR = src
BIN_DIR = bin

# Source files
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)

# Executables
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cpp,$(BIN_DIR)/%,$(SOURCES))

# Default target
all: $(EXECUTABLES)

# Rule to build each executable
$(BIN_DIR)/%: $(SRC_DIR)/%.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

# Clean rule
clean:
	rm -rf $(BIN_DIR)/*
