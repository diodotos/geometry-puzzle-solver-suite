# pragma once

#include <cstdint>
#include <array>
#include <string>
#include <sstream>
#include <memory>

#include "ABCCube.h"

/* --------------------------------------------------------------------------------------------
 * This file contains the class definition for a 2x2x2 "piecewise" representation of 
 * the Rubik's Cube (in contrast to the "stickerwise" representation). The idea is that
 * the state of a 2x2x2 Rubik's Cube is completely specified by the permutation and orientation
 * states of all 8 corner pieces.
 * 
 * "Permutation" refers to the position that a piece occupies in the cube. There are 8 possible
 * positions for a corner, so we say the permutation ranges from 0-7.
 *
 * Though a piece may be correctly permuted, it may be "flipped" in place. For example,
 * start with a solved cube and twist a corner in place, every piece is in the correct
 * spot but the cube is not solved. The "orientation" of a piece refers to the number of
 * in-place twists we need to perform on a corner such that the U or D color on that corner
 * exists in the U or D layer. Since orientations cycle every 3 twists, we say the orientation
 * ranges from 0-2.
 * 
 * Thus we represent the cube with an array of 8 Corner objects. A Corner object identifies
 * the color of the piece and its orientation. The index in the array corresponds to the
 * position of the piece in the cube.
 * 
 *  0   1   2   3   4   5   6   7
 * URF ULF ULB URB DRF DLF DLB DRB
 * WRB WRG WOG WOB YRB YRG YOG YOB
 *
 * The first line in the abovg schema is the index of the array. The second line 
 * is the position that each index is mapped to. The third line defines the initial conditions
 * we use for a solved cube.
 * 
 * Why would we care to represent the cube this way? Because it makes more sense for pattern
 * database solving methods. The amount of possible orientations of the corners of the cube 
 * is not that large; same for permutations. It is only when we consider jointly permutations
 * and orientations of pieces does the number of possible configurations explode.
 * But we can come up with a good estimate (better: we can come up with a good A*-admissible 
 * estimate) of how far we are from a solved state given some scrambled cube by looking at 
 * information about the orientations and permutations of pieces individually.
 * 
 * This method of solving the cube will be explained in greater detail in the file
 * implementing the A* algorithm.
 * --------------------------------------------------------------------------------------------
*/

class Cube2Pieces : public AbstractCube
{
public:
	// Standard constructors
	Cube2Pieces(); 
	explicit Cube2Pieces(const std::string& moves) : Cube2Pieces() { applyMoves(moves); }
	Cube2Pieces(const Cube2Pieces& other); // Copy constructor

	std::unique_ptr<AbstractCube> clone() const override;

	// Rotations
	AbstractCube& rotateU() override;
	AbstractCube& rotateD() override;
	AbstractCube& rotateR() override;
	AbstractCube& rotateL() override;
	AbstractCube& rotateF() override;
	AbstractCube& rotateB() override;

	AbstractCube& rotateUi() override;
	AbstractCube& rotateDi() override;
	AbstractCube& rotateRi() override;
	AbstractCube& rotateLi() override;
	AbstractCube& rotateFi() override;
	AbstractCube& rotateBi() override;

	AbstractCube& rotateU2() override;
	AbstractCube& rotateD2() override;
	AbstractCube& rotateR2() override;
	AbstractCube& rotateL2() override;
	AbstractCube& rotateF2() override;
	AbstractCube& rotateB2() override;

	/*
	 * These are attempts to normalize the cube.
	*/

	Cube2Pieces& cubeRotateX();
	Cube2Pieces& cubeRotateY();
	Cube2Pieces& cubeRotateZ();

	Cube2Pieces& cubeRotateXi();
	Cube2Pieces& cubeRotateYi();
	Cube2Pieces& cubeRotateZi();

	Cube2Pieces& cubeRotateX2();
	Cube2Pieces& cubeRotateY2();
	Cube2Pieces& cubeRotateZ2();

	Cube2Pieces& normalize();

	bool isSolved() const override;

	std::string toString() const override;

	uint32_t permutationHash() const;
	uint32_t orientationHash() const;
	uint64_t cubeHash() const;

	friend bool operator==(const Cube2Pieces& lhs, const Cube2Pieces& rhs);
	friend bool operator!=(const Cube2Pieces& lhs, const Cube2Pieces& rhs);

	~Cube2Pieces() override = default; // Explicitly default if no custom behavior is needed

private:
	enum class Position : uint8_t { URF, ULF, ULB, URB, DRF, DLF, DLB, DRB };
	enum class Piece : uint8_t { WRB, WRG, WOG, WOB, YRB, YRG, YOG, YOB };

	static const std::array<std::string, 8> positionToStringMap;
	static const std::array<std::string, 8> pieceToStringMap;

	struct Corner
	{
		Piece piece;
		uint8_t orientation;
	};

	// Primary data array
	std::array<Corner, 8> corners;

	void updateCornerOrientation(Position pos, uint8_t numRotate);
};
