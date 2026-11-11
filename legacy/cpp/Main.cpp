#include <unordered_map>
#include <iostream>
#include <string>
#include <sstream>
#include <queue>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <bitset>

#include "Cube2Pieces.h"
#include "Heuristic.h"
#include "Solvers.h"
#include "argparse.h"

void initArgparse(int argc, char** argv, argparse::ArgumentParser& program) {

	program.add_argument("mode")
		.required()
		.help("Operation mode: 'solve', 'benchmark', or 'heuristic'")
		.action([](const std::string& value) {
			static const std::vector<std::string> choices = { "solve", "benchmark", "heuristic"};
			if (std::find(choices.begin(), choices.end(), value) == choices.end()) {
				throw std::runtime_error("mode must be 'solve', benchmark', or 'heuristic'");
			}
			return value;
		});

	program.add_argument("--scramble")
		.help("Scramble to solve. Must be a string of moves separated by spaces, enclosed in double quotes. Example: \"U D R2 L2 F2 B2\"")
		.default_value(std::string(""));

	program.add_argument("--solver")
		.default_value(std::string("astardual"))
		.help("Type of solver to use in solve mode. Options are 'bfs', 'astarperf', 'astardual', 'astarori', 'astarperm'. Default is 'astardual'.")
		.action([](const std::string& value) {
			static const std::vector<std::string> choices = { "astarperf", "astardual", "astarori", "astarperm", "bfs" };
			if (std::find(choices.begin(), choices.end(), value) == choices.end()) {
				throw std::runtime_error("Invalid solver type.");
			}
			return value;
		});

	program.add_argument("--num-scrambles")
		.scan<'d', int>()
		.default_value(-1)
		.help("Number of scrambles to perform in benchmark mode.");

	program.parse_args(argc, argv);

	std::string mode = program.get<std::string>("mode");
	if (mode == "solve") {
		std::string scramble = program.get<std::string>("scramble");
		if (scramble.empty())
			throw std::runtime_error("Scramble must be provided in solve mode.");
	}
	else if (mode == "benchmark") {
		int num_scrambles = program.get<int>("--num-scrambles");
		if (num_scrambles < 1)
			throw std::runtime_error("Number of scrambles must be greater than 0.");
	}
}

void generateScrambles(int scramble_length, int num_scrambles)
{
	std::srand(std::time(nullptr));
	std::vector<std::string> moves = {"U", "U'", "U2", "D", "D'", "D2", "L", "L'", "L2", "R", "R'", "R2", "F", "F'", "F2", "B", "B'", "B2" };
	std::ofstream file("scrambles_tested.txt", std::ofstream::out);
	if (!file.is_open())
	{
		std::cerr << "Error opening file." << std::endl;
		return;
	}
	for (int i = 0; i < num_scrambles; i++)
	{
		std::string scramble = "";
		std::string last_move = "";
		char last_face = ' ';

		for (int j = 0; j < scramble_length; j++)
		{
			std::vector<std::string> valid_moves;

			// Filter out invalid moves
			for (const std::string& move : moves)
			{
				if (move[0] != last_face)
					valid_moves.push_back(move);
			}

			// Randomly select a move from the valid moves
			int move_index = std::rand() % valid_moves.size();
			std::string selected_move = valid_moves[move_index];

			last_move = selected_move;
			last_face = selected_move[0];

			scramble += selected_move + " ";
		}

		file << scramble << std::endl;
	}
}

std::vector<std::string> readScramblesFromFile(std::string filename)
{
	std::vector<std::string> scrambles;
	std::ifstream file(filename);
	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			scrambles.push_back(line);
		}
		file.close();
	}
	return scrambles;
}

std::pair<double, int> analyzeSolve(Cube2Pieces& cube, Solver& solver)
{
	auto start = std::chrono::high_resolution_clock::now();
	solver.solve();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	// Count the number of words in the solution string -- this is the number of moves
	std::istringstream iss(solver.getSolution());
	std::vector<std::string> tokens{ std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{} };
	return { elapsed_seconds.count(), tokens.size() };
}

void printData(const std::string& scramble, const Solver& s, const Cube2Pieces& cube, const std::pair<double, int> result)
{
	std::cout << "Scramble: " << scramble << std::endl;
	std::cout << "Solution: " << s.getSolution() << std::endl;
	std::cout << "Solution Length: " << result.second << " moves" << std::endl;
	std::cout << "Time: " << std::fixed << std::setprecision(6) << result.first << " seconds" << std::endl;
	std::cout << std::endl;
}

int main(int argc, char** argv)
{
	try {
		// Initialize argparse
		argparse::ArgumentParser program("CubeSolver");
		initArgparse(argc, argv, program);
		std::string mode = program.get<std::string>("mode");

		// Initialize heuristic lookup tables
		Heuristic::initOrientationLookup();
		Heuristic::initPermutationLookup();

		PermutationHeuristic permutationHeuristic;
		OrientationHeuristic orientationHeuristic;
		PerfectHeuristic perfectHeuristic;
		DualHeuristic dualHeuristic;


		if (mode == "solve")
		{
			std::string type = program.get<std::string>("--solver");
			std::string scramble = program.get<std::string>("--scramble");

			Cube2Pieces cube(scramble);
			std::pair<double, int> result;

			if (type == "astarperf")
			{
				// Don't load in the perfect heuristic lookup table until it's needed -- it's large
				Heuristic::initPerfectLookup();
				std::cout << "Solving with A* using perfect heuristic..." << std::endl;
				AStarSolver solver(cube, perfectHeuristic);
				result = analyzeSolve(cube, solver);
				printData(scramble, solver, cube, result);
			}
			else if (type == "astardual")
			{
				std::cout << "Solving with A* using dual heuristic..." << std::endl;
				AStarSolver solver(cube, dualHeuristic);
				result = analyzeSolve(cube, solver);
				printData(scramble, solver, cube, result);
			}
			else if (type == "astarori")
			{
				std::cout << "Solving with A* using orientation heuristic..." << std::endl;
				AStarSolver solver(cube, orientationHeuristic);
				result = analyzeSolve(cube, solver);
				printData(scramble, solver, cube, result);
			}
			else if (type == "astarperm")
			{
				std::cout << "Solving with A* using permutation heuristic..." << std::endl;
				AStarSolver solver(cube, permutationHeuristic);
				result = analyzeSolve(cube, solver);
				printData(scramble, solver, cube, result);
			}
			else if (type == "bfs")
			{
				std::cout << "Solving with BFS..." << std::endl;
				BFSSolver solver(cube);
				result = analyzeSolve(cube, solver);
				printData(scramble, solver, cube, result);
			}
			else {
				throw std::runtime_error("Invalid solver type.");
			}
		}

		else if (mode == "benchmark")
		{
			Heuristic::initPerfectLookup();
			int num_scrambles = program.get<int>("--num-scrambles");
			generateScrambles(15, num_scrambles);
			std::vector<std::string> scrambles = readScramblesFromFile("scrambles_tested.txt");

			std::ofstream file("benchmark_results.txt", std::ofstream::out);
			if (!file.is_open())
			{
				std::cerr << "Error opening file." << std::endl;
				return 1;
			}

			int it = 0;
			file << "Scramble,Length,Perf,Dual,Ori,Perm,BFS" << std::endl;
			for (const std::string& scramble : scrambles)
			{
				Cube2Pieces cube1(scramble);
				Cube2Pieces cube2(scramble);
				Cube2Pieces cube3(scramble);
				Cube2Pieces cube4(scramble);
				Cube2Pieces cubebfs(scramble);
				AStarSolver solver1(cube1, perfectHeuristic);
				AStarSolver solver2(cube2, dualHeuristic);
				AStarSolver solver3(cube3, orientationHeuristic);
				AStarSolver solver4(cube4, permutationHeuristic);
				BFSSolver solver5(cubebfs);

				std::vector<std::pair<double, int>> results;
				results.push_back(analyzeSolve(cube1, solver1));
				results.push_back(analyzeSolve(cube2, solver2));
				results.push_back(analyzeSolve(cube3, solver3));
				results.push_back(analyzeSolve(cube4, solver4));
				results.push_back(analyzeSolve(cubebfs, solver5));

				file << scramble << "," << results[0].second << "," << results[0].first << "," << results[1].first << "," << results[2].first << "," << results[3].first << "," << results[4].first << std::endl;
				it += 1;

				std::cout << "Scrambles analyzed: " << it << std::endl;
			}
		}

		else if (mode == "heuristic")
		{
			Heuristic::initPerfectLookup();
			std::ofstream file = std::ofstream("heuristic_evaluation.txt", std::ofstream::out);
			if (!file.is_open())
			{
				std::cerr << "Error opening file." << std::endl;
				return 1;
			}

			file << "Hash,Perfect,Orientation,Permutation,Dual" << std::endl;
			for (const auto& pair : Heuristic::perfectLookup)
			{
				const uint64_t mask_orientation = 0b1111111111111111000000000000000000000000;
				uint64_t orientationHash = (pair.first & mask_orientation) >> 24;
				uint64_t permutationHash = pair.first & ~mask_orientation;

				// Put in order of hash, perfect, orientation, permutation, dual

				file << pair.first << "," << pair.second << "," << Heuristic::orientationLookup[orientationHash] << "," << Heuristic::permutationLookup[permutationHash] << "," << std::max(Heuristic::orientationLookup[orientationHash], Heuristic::permutationLookup[permutationHash]) << std::endl;
			}
		}
	}
	catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
	}
}
