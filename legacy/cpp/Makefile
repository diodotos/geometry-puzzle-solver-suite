CC=g++
CFLAGS=-g -Wall --std=c++17
TARGET=CubeSolver

SOURCES=Main.cpp ABCCube.cpp Cube2Pieces.cpp Heuristic.cpp Solvers.cpp utils.cpp
HEADERS=ABCCube.h Cube2Pieces.h Heuristic.h Solvers.h utils.h
OBJECTS=$(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJECTS)
