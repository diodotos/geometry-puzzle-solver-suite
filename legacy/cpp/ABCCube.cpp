#include <string>
#include <vector>
#include <stdexcept>
#include <array>

#include "ABCCube.h"
#include "utils.h"

/*
 * Static and const member variables. These are maps that make the below functions more readable and maintainable.
*/
const std::array<std::string, 19> AbstractCube::moveToStringMap = {
	"None", "U", "D", "F", "B", "R", "L", "U'", "D'", "F'", "B'", "R'", "L'", "U2", "D2", "F2", "B2", "R2", "L2"
};

const std::array<AbstractCube::Move, 19> AbstractCube::moveToInverseMap = {
	Move::None, Move::Ui, Move::Di, Move::Fi, Move::Bi, Move::Ri, Move::Li, Move::U, Move::D, Move::F, Move::B, Move::R, Move::L, Move::U2, Move::D2, Move::F2, Move::B2, Move::R2, Move::L2
};

const std::unordered_map<std::string, AbstractCube::Move> AbstractCube::stringToMoveMap = {
	{"None", Move::None}, {"U", Move::U}, {"D", Move::D}, {"F", Move::F}, {"B", Move::B}, {"R", Move::R}, {"L", Move::L},
	{"U'", Move::Ui}, {"D'", Move::Di}, {"F'", Move::Fi}, {"B'", Move::Bi}, {"R'", Move::Ri}, {"L'", Move::Li},
	{"U2", Move::U2}, {"D2", Move::D2}, {"F2", Move::F2}, {"B2", Move::B2}, {"R2", Move::R2}, {"L2", Move::L2}
};

const std::unordered_map<AbstractCube::Move, AbstractCube::RotationFunction> AbstractCube::moveToFunctionMap = 
{
	{Move::U, &AbstractCube::rotateU}, {Move::D, &AbstractCube::rotateD}, {Move::F, &AbstractCube::rotateF},
	{Move::B, &AbstractCube::rotateB}, {Move::R, &AbstractCube::rotateR}, {Move::L, &AbstractCube::rotateL},
	{Move::Ui, &AbstractCube::rotateUi}, {Move::Di, &AbstractCube::rotateDi}, {Move::Fi, &AbstractCube::rotateFi},
	{Move::Bi, &AbstractCube::rotateBi}, {Move::Ri, &AbstractCube::rotateRi}, {Move::Li, &AbstractCube::rotateLi},
	{Move::U2, &AbstractCube::rotateU2}, {Move::D2, &AbstractCube::rotateD2}, {Move::F2, &AbstractCube::rotateF2},
	{Move::B2, &AbstractCube::rotateB2}, {Move::R2, &AbstractCube::rotateR2}, {Move::L2, &AbstractCube::rotateL2}
};

/*
 * Public member functions
*/
AbstractCube& AbstractCube::rotateInverse(Move move)
{
	return (this->*moveToFunctionMap.at(moveToInverseMap[static_cast<int>(move)]) )();
}

AbstractCube& AbstractCube::rotateInverse(const std::vector<Move>& moves)
{
	// Iterate through the moves in reverse order and apply the inverse of each move
	for (auto it = moves.rbegin(); it != moves.rend(); ++it)
			rotateInverse(*it);
	return *this;
}

AbstractCube& AbstractCube::rotateInverse(const std::string& moves)
{
	std::vector<std::string> movesVec = split(moves, ' ');
	// As with before, iterate through the moves in reverse order and apply the inverse of each move
	for (auto it = movesVec.rbegin(); it != movesVec.rend(); ++it)
	{
		if (stringToMoveMap.find(*it) == stringToMoveMap.end())
			throw std::invalid_argument("Invalid move in sequence: " + *it);
		rotateInverse(stringToMoveMap.at(*it));
	}
	return *this;
}


AbstractCube& AbstractCube::applyMoves(Move move)
{
	return (this->*moveToFunctionMap.at(move))();
}

AbstractCube& AbstractCube::applyMoves(const std::vector<Move>& moves)
{
	for (Move move : moves)
		(this->*moveToFunctionMap.at(move))();
	return *this;
}

AbstractCube& AbstractCube::applyMoves(const std::string& moves)
{
	std::vector<std::string> movesVec = split(moves, ' ');
	for (const std::string& move : movesVec)
	{
		if (stringToMoveMap.find(move) == stringToMoveMap.end())
			throw std::invalid_argument("Invalid move in sequence: " + move);
		(this->*moveToFunctionMap.at(stringToMoveMap.at(move)))();
	}
	return *this;
}

AbstractCube::Move AbstractCube::getPrevMove() const
{
	return prevMove;
}

AbstractCube& AbstractCube::clearPrevMove()
{
	prevMove = Move::None;
	return *this;
}

std::string AbstractCube::moveToString(Move move)
{
	return moveToStringMap[static_cast<int>(move)];
};

std::string AbstractCube::moveToString(const std::vector<Move>& moves)
{
	std::string res;
	for (Move move : moves)
		res += moveToString(move) + " ";
	return trimWhitespace(res);
}

std::vector<std::unique_ptr<AbstractCube>> AbstractCube::getNextMoves()
{
	/*
	 * This function justifies the choice of endowing the cube with 18 possible moves (as opposed to 6 or 12),
	 * since we could have alternatively implemented U2, Ui as iterated U, etc.
	 * 
	 * With all 18 moves, it makes it easier to implement some optimizations that significantly reduce the size 
	 * of the search tree by reducing the branching factor. Naively, with 18 possible moves at a step, the 
	 * branching factor is 18. But:
	 * 
	 * Optimization 1: We avoid any consecutive rotations of the same face. 
	 *				   This also prevents us from inverting the previous move.
	 * Optimization 2: We allow commutative move pairs to only go in one direction.
	 *				   That is, U/D, F/B, R/L are commutative. We can fix it so that
	 *                 if in a solution, U and D or D and U are consecutive, we only 
	 *                 allow U to follow D, and not the other way around.
	 *
	 * With these optimizations, the average branching factor is slightly over 13. 
	 * More details about the branching factor in the original Korf paper.
	 * This is about the sparsest tree we can get without significantly more complex optimizations.
	 * 
	 * Also note that in the below implementation, we use std::move and emplace_back. 
	 * std::move is used to transfer ownership of the cloned cube to the vector, and emplace_back is used to
	 * construct the unique_ptr in place, avoiding (additional) copying.
	*/
	std::vector<std::unique_ptr<AbstractCube>> res;

	if (prevMove != Move::U && prevMove != Move::U2 && prevMove != Move::Ui && prevMove != Move::D && prevMove != Move::D2 && prevMove != Move::Di)
	{
		auto cubeU = clone();
		cubeU->rotateU();
		res.emplace_back(std::move(cubeU));

		auto cubeUi = clone();
		cubeUi->rotateUi();
		res.emplace_back(std::move(cubeUi));

		auto cubeU2 = clone();
		cubeU2->rotateU2();
		res.emplace_back(std::move(cubeU2));
	}
	if (prevMove != Move::D && prevMove != Move::D2 && prevMove != Move::Di)
	{
		auto cubeD = clone();
		cubeD->rotateD();
		res.emplace_back(std::move(cubeD));

		auto cubeDi = clone();
		cubeDi->rotateDi();
		res.emplace_back(std::move(cubeDi));

		auto cubeD2 = clone();
		cubeD2->rotateD2();
		res.emplace_back(std::move(cubeD2));
	}
	if (prevMove != Move::F && prevMove != Move::F2 && prevMove != Move::Fi && prevMove != Move::B && prevMove != Move::B2 && prevMove != Move::Bi)
	{
		auto cubeF = clone();
		cubeF->rotateF();
		res.emplace_back(std::move(cubeF));

		auto cubeFi = clone();
		cubeFi->rotateFi();
		res.emplace_back(std::move(cubeFi));

		auto cubeF2 = clone();
		cubeF2->rotateF2();
		res.emplace_back(std::move(cubeF2));
	}
	if (prevMove != Move::B && prevMove != Move::B2 && prevMove != Move::Bi)
	{
		auto cubeB = clone();
		cubeB->rotateB();
		res.emplace_back(std::move(cubeB));

		auto cubeBi = clone();
		cubeBi->rotateBi();
		res.emplace_back(std::move(cubeBi));

		auto cubeB2 = clone();
		cubeB2->rotateB2();
		res.emplace_back(std::move(cubeB2));
	}
	if (prevMove != Move::R && prevMove != Move::R2 && prevMove != Move::Ri && prevMove != Move::L && prevMove != Move::L2 && prevMove != Move::Li)
	{
		auto cubeR = clone();
		cubeR->rotateR();
		res.emplace_back(std::move(cubeR));

		auto cubeRi = clone();
		cubeRi->rotateRi();
		res.emplace_back(std::move(cubeRi));

		auto cubeR2 = clone();
		cubeR2->rotateR2();
		res.emplace_back(std::move(cubeR2));
	}
	if (prevMove != Move::L && prevMove != Move::L2 && prevMove != Move::Li)
	{
		auto cubeL = clone();
		cubeL->rotateL();
		res.emplace_back(std::move(cubeL));

		auto cubeLi = clone();
		cubeLi->rotateLi();
		res.emplace_back(std::move(cubeLi));

		auto cubeL2 = clone();
		cubeL2->rotateL2();
		res.emplace_back(std::move(cubeL2));
	}
	return res;
}

std::string AbstractCube::getInverse(const std::string& moves) 
{
	std::vector<std::string> movesVec = split(moves, ' ');
	std::string res;
	for (auto it = movesVec.rbegin(); it != movesVec.rend(); ++it)
		res += moveToString(moveToInverseMap[static_cast<int>(stringToMoveMap.at(*it))]) + " ";
	return trimWhitespace(res);
}

/*
 * Protected member functions
*/
void AbstractCube::setPrevMove(Move move)
{
	prevMove = move;
}

/*
 * Outside functions
*/
std::ostream& operator<<(std::ostream& os, const AbstractCube& cube)
{
	os << cube.toString();
	return os;
}
