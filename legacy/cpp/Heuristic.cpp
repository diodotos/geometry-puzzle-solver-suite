#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <stdexcept>
#include <filesystem>

#include "Heuristic.h"
#include "Cube2Pieces.h"

// Static member initialization
std::unordered_map<uint64_t, uint16_t> Heuristic::orientationLookup;
std::unordered_map<uint64_t, uint16_t> Heuristic::permutationLookup;
std::unordered_map<uint64_t, uint16_t> Heuristic::perfectLookup;

// Base class functions
void Heuristic::initOrientationLookup()
{
	if (std::filesystem::exists("orientationLookup.txt"))
		readLookupFromFile(orientationLookup, "orientationLookup.txt");
	else {
		generateLookupTable(orientationLookup, &Cube2Pieces::orientationHash);
		writeLookupToFile(orientationLookup, "orientationLookup.txt");
		readLookupFromFile(orientationLookup, "orientationLookup.txt");
	}
}

void Heuristic::initPermutationLookup()
{
	if (std::filesystem::exists("permutationLookup.txt"))
		readLookupFromFile(permutationLookup, "permutationLookup.txt");
	else {
		generateLookupTable(permutationLookup, &Cube2Pieces::permutationHash);
		writeLookupToFile(permutationLookup, "permutationLookup.txt");
		readLookupFromFile(permutationLookup, "permutationLookup.txt");
	}
}

void Heuristic::initPerfectLookup()
{
	if (std::filesystem::exists("perfectLookup.txt"))
		readLookupFromFile(perfectLookup, "perfectLookup.txt");
	else {
		generateLookupTable(perfectLookup, &Cube2Pieces::cubeHash);
		writeLookupToFile(perfectLookup, "perfectLookup.txt");
		readLookupFromFile(perfectLookup, "perfectLookup.txt");
	}
}

void Heuristic::generateLookupTable(std::unordered_map<uint64_t, uint16_t>& lookup, uint32_t(Cube2Pieces::* hashFunction)() const)
{
	std::queue<std::pair<std::unique_ptr<AbstractCube>, uint16_t>> q; // BFS queue with depth.
	q.emplace(std::make_pair(std::make_unique<Cube2Pieces>(), 0)); // Use emplace 
	uint16_t searchDepth = 0;

	std::cout << "Generating lookup table..." << std::endl;
	while (!q.empty())
	{
		const auto& [current, depth] = std::move(q.front()); // use move semantics to avoid copying.
		if (!current)
			throw std::runtime_error("Error: generateLookupTable encountered a null pointer.");

		auto* currentRaw = dynamic_cast<Cube2Pieces*>(current.get());
		if (!currentRaw)
			throw std::runtime_error("Error: generateLookupTable encountered a null pointer.");

		if (depth > searchDepth)
		{
			std::cout << "Depth: " << searchDepth << std::endl;
			searchDepth = depth;
		}

		// TODO: switch order -- be consist with 64 bit version
		uint64_t hash = (currentRaw->*hashFunction)();
		if (lookup.find(hash) == lookup.end())
		{
			lookup[hash] = depth;
			for (auto& child : current->getNextMoves())
				q.emplace(std::move(child), depth + 1); // Emplace again
		}
		q.pop();
	}
}

void Heuristic::generateLookupTable(std::unordered_map<uint64_t, uint16_t>& lookup, uint64_t(Cube2Pieces::* hashFunction)() const)
{
	std::queue<std::pair<std::unique_ptr<AbstractCube>, uint16_t>> q; // BFS queue with depth.
	q.emplace(std::make_pair(std::make_unique<Cube2Pieces>(), 0)); // Use emplace 
	uint16_t searchDepth = 0;

	std::cout << "Generating lookup table..." << std::endl;
	while (!q.empty())
	{
		const auto& [current, depth] = std::move(q.front()); // use move semantics to avoid copying.

		if (!current)
			throw std::runtime_error("Error: generateLookupTable encountered a null pointer.");

		auto* currentRaw = dynamic_cast<Cube2Pieces*>(current.get());
		if (!currentRaw)
			throw std::runtime_error("Error: generateLookupTable encountered a null pointer.");

		if (depth > searchDepth)
		{
			std::cout << "Depth: " << searchDepth << std::endl;
			searchDepth = depth;
		}

		for (auto& child : current->getNextMoves())
		{
			auto* childRaw = dynamic_cast<Cube2Pieces*>(child.get());
			if (!childRaw)
				throw std::runtime_error("Error: generateLookupTable encountered a null pointer.");
			uint64_t hash = (childRaw->*hashFunction)();
			if (lookup.find(hash) == lookup.end())
			{
				lookup[hash] = depth + 1;
				q.emplace(std::move(child), depth + 1); // Emplace again
			}
		}
		q.pop();
	}
}

void Heuristic::writeLookupToFile(const std::unordered_map<uint64_t, uint16_t>&lookup, const std::string & filename)
{
	if (lookup.empty())
		throw std::invalid_argument("Error: writeLookupToFile called with an empty lookup table");
	if (filename.empty())
		throw std::invalid_argument("Error: writeLookupToFile called with an empty filename.");
	if (std::filesystem::exists(filename))
		throw std::runtime_error("Error: writeLookupToFile called with a filename that already exists: " + filename);

	std::ofstream file(filename);
	if (!file)
		throw std::runtime_error("Error: writeLookupToFile could not open the file for writing: " + filename);

	std::cout << "Writing " << lookup.size() << " entries to " << filename << std::endl;
	for (const auto& [hash, depth] : lookup)
		file << hash << " " << depth << std::endl;
	std::cout << "Finished writing to " << filename << std::endl;
	file.close();
}

void Heuristic::readLookupFromFile(std::unordered_map<uint64_t, uint16_t>& lookup, const std::string& filename)
{
	if (filename.empty())
		throw std::invalid_argument("Error: readLookupFromFile called with an empty filename.");
	if (!std::filesystem::exists(filename))
		throw std::runtime_error("Error: readLookupFromFile called with a filename that does not exist: " + filename);

	std::ifstream file(filename);
	if (!file)
		throw std::runtime_error("Error: readLookupFromFile could not open the file for reading: " + filename);

	std::cout << "Reading from " << filename << std::endl;
	uint64_t hash;
	uint16_t depth;
	while (file >> hash >> depth)
		lookup[hash] = depth;
	std::cout << "Finished reading from " << filename << std::endl;
	file.close();
}

// Specific heuristics
uint16_t OrientationHeuristic::heuristic(const Cube2Pieces& cube) const
{
	return orientationLookup[cube.orientationHash()];
}

uint16_t PermutationHeuristic::heuristic(const Cube2Pieces& cube) const
{
	return permutationLookup[cube.permutationHash()];
}

uint16_t DualHeuristic::heuristic(const Cube2Pieces& cube) const
{
	return std::max(orientationLookup[cube.orientationHash()], permutationLookup[cube.permutationHash()]);
}

uint16_t PerfectHeuristic::heuristic(const Cube2Pieces& cube) const
{
	return perfectLookup[cube.cubeHash()];
}
