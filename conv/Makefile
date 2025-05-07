# CUDA compiler and flags
NVCC = nvcc
NVCCFLAGS = -Icommon

# Directories
SRC_DIR = src
BIN_DIR = bin
INC_DIR = common

# Source files and corresponding executables
SOURCES := $(wildcard $(SRC_DIR)/*.cu)
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(SOURCES))

# Default target
all: $(EXECUTABLES)

# Rule to compile each .cu file into a binary
$(BIN_DIR)/%: $(SRC_DIR)/%.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Clean rule
clean:
	rm -rf $(BIN_DIR)/*
