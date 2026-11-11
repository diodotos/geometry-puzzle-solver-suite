#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

#include "utils.h"
#include "Cube2Pieces.h"
#include "Heuristic.h"

class Solver
{
public:
	enum class SolverType { BFS, ASTAR };

	// Ensure that CubeType is a derivative of AbstractCube or a related interface
	explicit Solver(Cube2Pieces& startCube) : startCube(startCube) { startCube.clearPrevMove(); }
	virtual void solve() = 0;
	std::string getSolution() const
	{
		if (!startCube.isSolved())
			throw std::runtime_error("Error: getSolution called on an unsolved cube.");
		return solutionPath;
	}

	// This class should be non-copyable because it contains a reference to an AbstractCube object.
	Solver(const Solver&) = delete;
	Solver& operator=(const Solver&) = delete;

protected:
	Cube2Pieces& startCube;
	std::string solutionPath = "";
};

class BFSSolver : public Solver
{
public:
	using Solver::Solver;
	void solve() override;
protected:
	struct BFSNode {
		std::shared_ptr<Cube2Pieces> cube;
		std::shared_ptr<BFSNode> parent;
		AbstractCube::Move move;

		BFSNode(std::shared_ptr<Cube2Pieces> cube, std::shared_ptr<BFSNode> parent, AbstractCube::Move move)
			: cube(cube), parent(parent), move(move) {}
	};

	void reconstructPath(const std::shared_ptr<BFSNode>& endNode);
};

class AStarSolver : public Solver
{
public:
	explicit AStarSolver(Cube2Pieces& startCube, Heuristic& heuristic)
		: Solver(startCube), heuristic(heuristic) {}
	void solve() override;
private:
	Heuristic& heuristic;

	struct AStarNode : public std::enable_shared_from_this<AStarNode> {
		std::shared_ptr<Cube2Pieces> cube;
		std::shared_ptr<AStarNode> parent;
		uint16_t gScore;
		uint16_t fScore;
		AbstractCube::Move move;

		AStarNode(std::shared_ptr<Cube2Pieces> cube, std::shared_ptr<AStarNode> parent, uint16_t gScore, uint16_t hScore, AbstractCube::Move move)
			: cube(cube), parent(parent), gScore(gScore), fScore(gScore + hScore), move(move) {}

		bool operator>(const AStarNode& other) const { return fScore > other.fScore; }

		std::shared_ptr<AStarNode> get_shared_this() { return shared_from_this(); }
	};

	// Required for the priority queue
	struct AStarNodeCompare {
		bool operator()(const std::shared_ptr<AStarNode>& lhs, const std::shared_ptr<AStarNode>& rhs) const {
			return *lhs > *rhs;
		}
	};

	void reconstructPath(const std::shared_ptr<AStarNode>& endNode, const std::unordered_map<uint64_t, std::pair<uint16_t, std::shared_ptr<AStarNode>>>& nodeMap);
};
