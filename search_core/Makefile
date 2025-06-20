# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra

# Target executable
TARGET = search_tool

# Source file
SRC = search.cpp

# Default rule
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Rule to clean up build files
clean:
	rm -f $(TARGET)

.PHONY: all clean