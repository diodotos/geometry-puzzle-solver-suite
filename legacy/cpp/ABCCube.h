#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <memory>

#include "utils.h"

/* --------------------------------------------------------------------------------------------
 * We define an abstract base class (ABC) for an nxnxn Rubik's Cube.
 * In this project, we will use varying implementations of a 2x2x2 Rubik's Cube class,
 * as some implementations are better for some solving methods.
 * --------------------------------------------------------------------------------------------
*/

class AbstractCube
{
public:
	// Enum class for the moves that can be applied to the cube
	enum class Move : uint8_t {
		None = 0,
		U, D, F, B, R, L,
		Ui, Di, Fi, Bi, Ri, Li,
		U2, D2, F2, B2, R2, L2
	};

	// Typedef for a function pointer to a rotation function
	typedef AbstractCube& (AbstractCube::* RotationFunction)();

	// Constructor
	AbstractCube() : prevMove(Move::None) {}

	// Clone function - used for getting neighbors in the search tree
	// Should return a unique_pointer to a new instance of the derived class
	// Note this is not a copy constructor, it is used to create a new instance
	// of the derived class without knowing the derived class type a priori,
	// to preserve polymorphism
	virtual std::unique_ptr<AbstractCube> clone() const = 0; // TODO: figure out how to use templates to avoid having to override this

	// Standard rotations
	virtual AbstractCube& rotateU() = 0;
	virtual AbstractCube& rotateD() = 0;
	virtual AbstractCube& rotateR() = 0;
	virtual AbstractCube& rotateL() = 0;
	virtual AbstractCube& rotateF() = 0;
	virtual AbstractCube& rotateB() = 0;

	// Inverse rotations
	virtual AbstractCube& rotateUi() = 0;
	virtual AbstractCube& rotateDi() = 0;
	virtual AbstractCube& rotateRi() = 0;
	virtual AbstractCube& rotateLi() = 0;
	virtual AbstractCube& rotateFi() = 0;
	virtual AbstractCube& rotateBi() = 0;

	// Double rotations
	virtual AbstractCube& rotateU2() = 0;
	virtual AbstractCube& rotateD2() = 0;
	virtual AbstractCube& rotateR2() = 0;
	virtual AbstractCube& rotateL2() = 0;
	virtual AbstractCube& rotateF2() = 0;
	virtual AbstractCube& rotateB2() = 0;

	// Invert a rotation or sequence of rotations. 
	// Takes input as a single Move, a vector of Moves, or a string of moves
	AbstractCube& rotateInverse(Move move);
	AbstractCube& rotateInverse(const std::vector<Move>& moves);
	AbstractCube& rotateInverse(const std::string& moves);


	// Apply a move or sequence of moves to the current state of the cube
	// Takes input as a single Move, a vector of Moves, or a string of moves
	AbstractCube& applyMoves(Move move);
	AbstractCube& applyMoves(const std::vector<Move>& moves);
	AbstractCube& applyMoves(const std::string& moves);


	// Return the previous move as a Move
	Move getPrevMove() const;
	
	// Clear the previous move -- for circumstances where
	// the previous move should not be remembered
	AbstractCube& clearPrevMove();

	// Check if the cube is in a solved state
	virtual bool isSolved() const = 0;

	// Get the current configuration of the cube as a string
	virtual std::string toString() const = 0;

	// Convert a Move or sequence of Moves to standard rotation format as a string
	static std::string moveToString(Move move);
	static std::string moveToString(const std::vector<Move>& moves);

	// Get a list of configurations that can be reached from the current configuration
	// Intelligently filters out inverses/commutative move pairs
	std::vector<std::unique_ptr<AbstractCube>> getNextMoves();

	// TODO: make cubeHash mandatory for all cubes

	static std::string getInverse(const std::string& moves);


	// Virtual destructor
	virtual ~AbstractCube() {}

protected:
	// Updates the previous move
	// Should be called after a move is applied to the cube
	// Should only be accessable by the derived classes
	void setPrevMove(Move move);

private:
	Move prevMove;
	static const std::array<std::string, 19> moveToStringMap;
	static const std::array<Move, 19> moveToInverseMap;
	static const std::unordered_map<std::string, Move> stringToMoveMap;
	static const std::unordered_map<Move, RotationFunction> moveToFunctionMap;
};

std::ostream& operator<<(std::ostream& os, const AbstractCube& cube);
