# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -mavx2 -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas

# Target executable
TARGET = search_tool

# Source file
SRC = search.cpp

# Include and library paths
INCLUDES = -I/usr/include/openblas
LIBDIRS = -L/usr/lib

# Default rule
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SRC) $(LIBDIRS) $(LDFLAGS)

# Rule to clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean