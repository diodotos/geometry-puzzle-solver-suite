#include <vector>
#include <string>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "utils.h"
#include "ABCCube.h"
#include "Cube2Pieces.h"
#include "Heuristic.h"
#include "Solvers.h"

void BFSSolver::solve()
{
	solutionPath = "";
	std::queue<std::shared_ptr<BFSNode>> frontier;
	std::unordered_set<uint64_t> visited;

	frontier.emplace(std::make_shared<BFSNode>(std::make_shared<Cube2Pieces>(startCube), nullptr, AbstractCube::Move::None));
	visited.insert(startCube.cubeHash());

	while (!frontier.empty())
	{
		auto current = frontier.front();
		if (current->cube->isSolved())
		{
			reconstructPath(current);
			startCube.applyMoves(solutionPath);
			return;
		}
		frontier.pop();

		for (auto& nextCube : current->cube->getNextMoves())
		{
			uint64_t nextHash = static_cast<Cube2Pieces*>(nextCube.get())->cubeHash();
			if (visited.find(nextHash) == visited.end())
			{
				auto nextNode = std::make_shared<BFSNode>(std::make_shared<Cube2Pieces>(*static_cast<Cube2Pieces*>(nextCube.get())), current, nextCube->getPrevMove());
				visited.insert(nextHash);
				frontier.emplace(std::move(nextNode));
			}
		}
	}
}

void BFSSolver::reconstructPath(const std::shared_ptr<BFSNode>& endNode)
{
	std::vector<AbstractCube::Move> moves;
	auto current = endNode;
	while (current && current->move != AbstractCube::Move::None)
	{
		moves.push_back(current->move);
		current = current->parent;
	}
	std::reverse(moves.begin(), moves.end());  // Make sure the moves are in the correct order
	solutionPath = AbstractCube::moveToString(moves);
}

void AStarSolver::solve()
{
	/*
	 * openSet and nodeMap store essentially the same data, but we want to be able to do both
	 * lookup and a retrieval of the node with the lowest fScore in approximately O(1) time.
	 * So we use a priority queue for the latter and a hash map for the former.
	 * openSet is a priority queue of AStarNode objects, sorted by fScore, and is used for 
	 * selecting the next node to explore. nodeMap is a hash map of node hashes to fScores.
	 * We use it to check if a node has already been visited, and if it has, whether the 
	 * current path to it is better than the previous one. If it is, we add another
	 * AStarNode object to openSet with the new fScore. This results in redundancy
	 * in openSet since sometimes we store multiple AStarNode objects for the same node in
	 * tree, but with different fScores. But this tradeoff of memory is necessary for the
	 * O(1) lookup time. This does not affect the correctness of the algorithm, since the 
	 * priority queue will always select the node with the lowest fScore.
	*/
	solutionPath = "";
	std::priority_queue<std::shared_ptr<AStarNode>, std::vector<std::shared_ptr<AStarNode>>, AStarNodeCompare> openSet;
	std::unordered_map<uint64_t, std::pair<uint16_t, std::shared_ptr<AStarNode>>> nodeMap;

	// Initialize both openSet and nodeMap with the startCube
	auto startNode = std::make_shared<AStarNode>(std::make_shared<Cube2Pieces>(startCube), nullptr, 0, heuristic.heuristic(startCube), AbstractCube::Move::None);
	openSet.emplace(startNode);
	nodeMap[startCube.cubeHash()] = { 0, startNode };

	while (!openSet.empty())
	{
		auto current = openSet.top();
		if (current->cube->isSolved()) // Cube is solved
		{
			reconstructPath(current, nodeMap);
			startCube.applyMoves(solutionPath);
			return;
		}
		openSet.pop();

		if (current->gScore > nodeMap[current->cube->cubeHash()].first)
			continue;

		for (auto& nextCube : current->cube->getNextMoves())
		{
			uint16_t candidate_gScore = current->gScore + 1;
			uint64_t nextHash = static_cast<Cube2Pieces*>(nextCube.get())->cubeHash();

			// Either we discovered a new node or we found a better path to an existing node
			if (nodeMap.find(nextHash) == nodeMap.end() || candidate_gScore < nodeMap[nextHash].first)
			{
				auto nextNode = std::make_shared<AStarNode>(std::make_shared<Cube2Pieces>(*static_cast<Cube2Pieces*>(nextCube.get())), current, candidate_gScore, heuristic.heuristic(*static_cast<Cube2Pieces*>(nextCube.get())), nextCube->getPrevMove());
				nodeMap[nextHash] = { candidate_gScore, nextNode };
				openSet.emplace(std::move(nextNode));
			}
		}
	}
}

void AStarSolver::reconstructPath(const std::shared_ptr<AStarNode>& endNode, const std::unordered_map<uint64_t, std::pair<uint16_t, std::shared_ptr<AStarNode>>>& nodeMap)
{
	std::vector<AbstractCube::Move> moves;
	auto current = endNode;
	while (current && current->move != AbstractCube::Move::None)
	{
		moves.push_back(current->move);
		auto it = nodeMap.find(static_cast<Cube2Pieces*>(current->cube.get())->cubeHash());
		current = it != nodeMap.end() ? it->second.second->parent : nullptr;
	}
	std::reverse(moves.begin(), moves.end());  // Make sure the moves are in the correct order
	solutionPath = AbstractCube::moveToString(moves);
}
