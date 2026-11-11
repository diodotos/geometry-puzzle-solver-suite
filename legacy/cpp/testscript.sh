#!/bin/bash

echo We are Fyze Tulyag and Richie Hsiung.
echo Our project is making an efficient and optimal 2x2x2 Rubik\'s Cube solver. 
echo We use an A\* search algorithm, and we are comparing its performance to a baseline BFS algorithm.
echo We also want to do analysis about how good our heuristic is.
echo This testing script will call the program with a few predetermined scrambles. 
echo It will solve these scrambles with different solvers to compare time taken.
echo Our program has a full CLI with many different modes and more features. This script just uses the default mode.
echo Documentation about how to use the CLI and a more robust analysis \(including nice graphs\) of the performance and quality of our heuristic can be found at:
echo https://github.com/diodotos/CubeSolver
echo You can expect to see our implementation, astardual, consistently and significantly beating the baseline, bfs, in terms of time taken.

echo 
echo FIRST SCRAMBLE
echo
./CubeSolver solve --scramble "F U F2 U R' U F' R2 U'" --solver astardual
./CubeSolver solve --scramble "F U F2 U R' U F' R2 U'" --solver bfs

echo
echo SECOND SCRAMBLE
echo
./CubeSolver solve --scramble "R' U F' R U' F2 U F2 U2" --solver astardual
./CubeSolver solve --scramble "R' U F' R U' F2 U F2 U2" --solver bfs

echo
echo THIRD SCRAMBLE
echo
./CubeSolver solve --scramble "U2 R F2 U' R F R' U R2" --solver astardual
./CubeSolver solve --scramble "U2 R F2 U' R F R' U R2" --solver bfs

echo
echo FOURTH SCRAMBLE
echo
./CubeSolver solve --scramble "R U2 F' R' F R2 F U2 F2" --solver astardual
./CubeSolver solve --scramble "R U2 F' R' F R2 F U2 F2" --solver bfs

echo
echo SIXTH SCRAMBLE -- ALL SOLVERS USED
echo
./CubeSolver solve --scramble "R' U F2 R2 F' U' R F' R'" --solver astarperf
./CubeSolver solve --scramble "R' U F2 R2 F' U' R F' R'" --solver astardual
./CubeSolver solve --scramble "R' U F2 R2 F' U' R F' R'" --solver astarori
./CubeSolver solve --scramble "R' U F2 R2 F' U' R F' R'" --solver astarperm
./CubeSolver solve --scramble "R' U F2 R2 F' U' R F' R'" --solver bfs
