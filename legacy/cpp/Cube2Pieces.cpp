#include <string>
#include <array>
#include <sstream>
#include <memory>
#include <cstdint>

#include "Cube2Pieces.h"

/*
 * Static and const member variables
*/
const std::array<std::string, 8> Cube2Pieces::positionToStringMap = {
	"URF", "ULF", "ULB", "URB", "DRF", "DLF", "DLB", "DRB"
};
const std::array<std::string, 8> Cube2Pieces::pieceToStringMap = {
	"WRB", "WRG", "WOG", "WOB", "YRB", "YRG", "YOG", "YOB"
};

/*
 * Public member functions
*/

Cube2Pieces::Cube2Pieces() : AbstractCube()
{
	// Initialize the cube to a solved state
	for (uint8_t i = 0; i < 8; i++)
	{
		corners[i].piece = static_cast<Piece>(i);
		corners[i].orientation = 0;
	}
}

Cube2Pieces::Cube2Pieces(const Cube2Pieces& other) : AbstractCube()
{
	for (uint8_t i = 0; i < 8; i++)
	{
		corners[i].piece = other.corners[i].piece;
		corners[i].orientation = other.corners[i].orientation;
	}
	setPrevMove(other.getPrevMove());
}

std::unique_ptr<AbstractCube> Cube2Pieces::clone() const
{
	return std::make_unique<Cube2Pieces>(*this);
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateU()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::URF)];
	corners[static_cast<uint8_t>(Position::URF)] = corners[static_cast<uint8_t>(Position::URB)];
	corners[static_cast<uint8_t>(Position::URB)] = corners[static_cast<uint8_t>(Position::ULB)];
	corners[static_cast<uint8_t>(Position::ULB)] = corners[static_cast<uint8_t>(Position::ULF)];
	corners[static_cast<uint8_t>(Position::ULF)] = tmp;
	// Note that the orientation of the pieces does not change in U/D moves.
	setPrevMove(Move::U);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateD()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::DRF)];
	corners[static_cast<uint8_t>(Position::DRF)] = corners[static_cast<uint8_t>(Position::DLF)];
	corners[static_cast<uint8_t>(Position::DLF)] = corners[static_cast<uint8_t>(Position::DLB)];
	corners[static_cast<uint8_t>(Position::DLB)] = corners[static_cast<uint8_t>(Position::DRB)];
	corners[static_cast<uint8_t>(Position::DRB)] = tmp;
	// Note that the orientation of the pieces does not change in U/D moves.
	setPrevMove(Move::D);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateR()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::URF)];
	corners[static_cast<uint8_t>(Position::URF)] = corners[static_cast<uint8_t>(Position::DRF)];
	corners[static_cast<uint8_t>(Position::DRF)] = corners[static_cast<uint8_t>(Position::DRB)];
	corners[static_cast<uint8_t>(Position::DRB)] = corners[static_cast<uint8_t>(Position::URB)];
	corners[static_cast<uint8_t>(Position::URB)] = tmp;

	updateCornerOrientation(Position::URF, 1);
	updateCornerOrientation(Position::DRF, 2);
	updateCornerOrientation(Position::DRB, 1);
	updateCornerOrientation(Position::URB, 2);

	setPrevMove(Move::R);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateL()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::ULF)];
	corners[static_cast<uint8_t>(Position::ULF)] = corners[static_cast<uint8_t>(Position::ULB)];
	corners[static_cast<uint8_t>(Position::ULB)] = corners[static_cast<uint8_t>(Position::DLB)];
	corners[static_cast<uint8_t>(Position::DLB)] = corners[static_cast<uint8_t>(Position::DLF)];
	corners[static_cast<uint8_t>(Position::DLF)] = tmp;

	updateCornerOrientation(Position::ULF, 2);
	updateCornerOrientation(Position::ULB, 1);
	updateCornerOrientation(Position::DLB, 2);
	updateCornerOrientation(Position::DLF, 1);

	setPrevMove(Move::L);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateF()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::URF)];
	corners[static_cast<uint8_t>(Position::URF)] = corners[static_cast<uint8_t>(Position::ULF)];
	corners[static_cast<uint8_t>(Position::ULF)] = corners[static_cast<uint8_t>(Position::DLF)];
	corners[static_cast<uint8_t>(Position::DLF)] = corners[static_cast<uint8_t>(Position::DRF)];
	corners[static_cast<uint8_t>(Position::DRF)] = tmp;

	updateCornerOrientation(Position::URF, 2);
	updateCornerOrientation(Position::ULF, 1);
	updateCornerOrientation(Position::DLF, 2);
	updateCornerOrientation(Position::DRF, 1);

	setPrevMove(Move::F);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateB()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::URB)];
	corners[static_cast<uint8_t>(Position::URB)] = corners[static_cast<uint8_t>(Position::DRB)];
	corners[static_cast<uint8_t>(Position::DRB)] = corners[static_cast<uint8_t>(Position::DLB)];
	corners[static_cast<uint8_t>(Position::DLB)] = corners[static_cast<uint8_t>(Position::ULB)];
	corners[static_cast<uint8_t>(Position::ULB)] = tmp;

	updateCornerOrientation(Position::URB, 1);
	updateCornerOrientation(Position::DRB, 2);
	updateCornerOrientation(Position::DLB, 1);
	updateCornerOrientation(Position::ULB, 2);

	setPrevMove(Move::B);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateUi()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::URF)];
	corners[static_cast<uint8_t>(Position::URF)] = corners[static_cast<uint8_t>(Position::ULF)];
	corners[static_cast<uint8_t>(Position::ULF)] = corners[static_cast<uint8_t>(Position::ULB)];
	corners[static_cast<uint8_t>(Position::ULB)] = corners[static_cast<uint8_t>(Position::URB)];
	corners[static_cast<uint8_t>(Position::URB)] = tmp;
	// Note that the orientation of the pieces does not change in U/D moves.
	setPrevMove(Move::Ui);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateDi()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::DRF)];
	corners[static_cast<uint8_t>(Position::DRF)] = corners[static_cast<uint8_t>(Position::DRB)];
	corners[static_cast<uint8_t>(Position::DRB)] = corners[static_cast<uint8_t>(Position::DLB)];
	corners[static_cast<uint8_t>(Position::DLB)] = corners[static_cast<uint8_t>(Position::DLF)];
	corners[static_cast<uint8_t>(Position::DLF)] = tmp;
	// Note that the orientation of the pieces does not change in U/D moves.
	setPrevMove(Move::Di);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateRi()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::URF)];
	corners[static_cast<uint8_t>(Position::URF)] = corners[static_cast<uint8_t>(Position::URB)];
	corners[static_cast<uint8_t>(Position::URB)] = corners[static_cast<uint8_t>(Position::DRB)];
	corners[static_cast<uint8_t>(Position::DRB)] = corners[static_cast<uint8_t>(Position::DRF)];
	corners[static_cast<uint8_t>(Position::DRF)] = tmp;

	updateCornerOrientation(Position::URF, 1);
	updateCornerOrientation(Position::URB, 2);
	updateCornerOrientation(Position::DRB, 1);
	updateCornerOrientation(Position::DRF, 2);

	setPrevMove(Move::Ri);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateLi()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::ULF)];
	corners[static_cast<uint8_t>(Position::ULF)] = corners[static_cast<uint8_t>(Position::DLF)];
	corners[static_cast<uint8_t>(Position::DLF)] = corners[static_cast<uint8_t>(Position::DLB)];
	corners[static_cast<uint8_t>(Position::DLB)] = corners[static_cast<uint8_t>(Position::ULB)];
	corners[static_cast<uint8_t>(Position::ULB)] = tmp;

	updateCornerOrientation(Position::ULF, 2);
	updateCornerOrientation(Position::DLF, 1);
	updateCornerOrientation(Position::DLB, 2);
	updateCornerOrientation(Position::ULB, 1);

	setPrevMove(Move::Li);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateFi()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::URF)];
	corners[static_cast<uint8_t>(Position::URF)] = corners[static_cast<uint8_t>(Position::DRF)];
	corners[static_cast<uint8_t>(Position::DRF)] = corners[static_cast<uint8_t>(Position::DLF)];
	corners[static_cast<uint8_t>(Position::DLF)] = corners[static_cast<uint8_t>(Position::ULF)];
	corners[static_cast<uint8_t>(Position::ULF)] = tmp;

	updateCornerOrientation(Position::URF, 2);
	updateCornerOrientation(Position::DRF, 1);
	updateCornerOrientation(Position::DLF, 2);
	updateCornerOrientation(Position::ULF, 1);

	setPrevMove(Move::Fi);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateBi()
{
	Corner tmp = corners[static_cast<uint8_t>(Position::URB)];
	corners[static_cast<uint8_t>(Position::URB)] = corners[static_cast<uint8_t>(Position::ULB)];
	corners[static_cast<uint8_t>(Position::ULB)] = corners[static_cast<uint8_t>(Position::DLB)];
	corners[static_cast<uint8_t>(Position::DLB)] = corners[static_cast<uint8_t>(Position::DRB)];
	corners[static_cast<uint8_t>(Position::DRB)] = tmp;

	updateCornerOrientation(Position::URB, 1);
	updateCornerOrientation(Position::ULB, 2);
	updateCornerOrientation(Position::DLB, 1);
	updateCornerOrientation(Position::DRB, 2);

	setPrevMove(Move::Bi);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateU2()
{
	swap_items(corners[static_cast<uint8_t>(Position::URF)], corners[static_cast<uint8_t>(Position::ULB)]);
	swap_items(corners[static_cast<uint8_t>(Position::ULF)], corners[static_cast<uint8_t>(Position::URB)]);
	// Note that the orientation of the pieces does not change in double moves.
	setPrevMove(Move::U2);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateD2()
{
	swap_items(corners[static_cast<uint8_t>(Position::DRF)], corners[static_cast<uint8_t>(Position::DLB)]);
	swap_items(corners[static_cast<uint8_t>(Position::DLF)], corners[static_cast<uint8_t>(Position::DRB)]);
	// Note that the orientation of the pieces does not change in double moves.
	setPrevMove(Move::D2);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateR2()
{
	swap_items(corners[static_cast<uint8_t>(Position::URF)], corners[static_cast<uint8_t>(Position::DRB)]);
	swap_items(corners[static_cast<uint8_t>(Position::DRF)], corners[static_cast<uint8_t>(Position::URB)]);
	// Note that the orientation of the pieces does not change in double moves.
	setPrevMove(Move::R2);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateL2()
{
	swap_items(corners[static_cast<uint8_t>(Position::ULF)], corners[static_cast<uint8_t>(Position::DLB)]);
	swap_items(corners[static_cast<uint8_t>(Position::DLF)], corners[static_cast<uint8_t>(Position::ULB)]);
	// Note that the orientation of the pieces does not change in double moves.
	setPrevMove(Move::L2);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateF2()
{
	swap_items(corners[static_cast<uint8_t>(Position::URF)], corners[static_cast<uint8_t>(Position::DLF)]);
	swap_items(corners[static_cast<uint8_t>(Position::DRF)], corners[static_cast<uint8_t>(Position::ULF)]);
	// Note that the orientation of the pieces does not change in double moves.
	setPrevMove(Move::F2);
	return *this;
}

Cube2Pieces::AbstractCube& Cube2Pieces::rotateB2()
{
	swap_items(corners[static_cast<uint8_t>(Position::URB)], corners[static_cast<uint8_t>(Position::DLB)]);
	swap_items(corners[static_cast<uint8_t>(Position::ULB)], corners[static_cast<uint8_t>(Position::DRB)]);
	// Note that the orientation of the pieces does not change in double moves.
	setPrevMove(Move::B2);
	return *this;
}

bool Cube2Pieces::isSolved() const
{
	// Compare to the solved state of the cube
	// TODO: can just manually put in the value of the hash of the solved state to avoid instantiating a solved cube
	Cube2Pieces solved;
	return this->cubeHash() == solved.cubeHash();
}

std::string Cube2Pieces::toString() const
{
	std::stringstream ss;
	for (uint8_t i = 0; i < 8; i++)
	{
		ss << positionToStringMap[i] << ": ";
		ss << pieceToStringMap[static_cast<uint8_t>(corners[i].piece)] << " ";
		ss << static_cast<int>(corners[i].orientation) << std::endl;
	}
	return ss.str();
}

uint32_t Cube2Pieces::permutationHash() const
{
	Cube2Pieces normalized = *this;
	normalized.normalize();
	uint32_t hash = 0;
	// Since pieces are in the range 0-7, each of the pieces understood as integers can be represented in 3 bits.
	// And we have 8 pieces, so we need 24 bits to represent the permutation of the pieces.
	for (uint8_t i = 0; i < 8; i++)
		hash |= static_cast<uint32_t>(normalized.corners[i].piece) << (i * 3);
	return hash;
}

uint32_t Cube2Pieces::orientationHash() const
{
	Cube2Pieces normalized = *this;
	normalized.normalize();
	uint32_t hash = 0;
	// Similar idea to the permutation hash, but we only need 2 bits to represent the orientation of a piece.
	// Since we are in the range 0-2, we can represent the orientation of 8 pieces in 16 bits.
	for (uint8_t i = 0; i < 8; i++)
		hash |= static_cast<uint32_t>(normalized.corners[i].orientation) << (i * 2);
	return hash;
}

uint64_t Cube2Pieces::cubeHash() const
{
	Cube2Pieces normalized = *this;
	normalized.normalize();
	uint64_t hash = 0;
	for (uint8_t i = 0; i < 8; i++)
		hash |= static_cast<uint64_t>(normalized.corners[i].piece) << (i * 3);
	for (uint8_t i = 0; i < 8; i++)
		hash |= static_cast<uint64_t>(normalized.corners[i].orientation) << (24 + i * 2);
	return hash;
}

/*
 * Friend functions
*/
// TODO: revert this back to the original manual check of the corner pieces ?
bool operator==(const Cube2Pieces& lhs, const Cube2Pieces& rhs)
{
	return lhs.cubeHash() == rhs.cubeHash();
}

bool operator!=(const Cube2Pieces& lhs, const Cube2Pieces& rhs)
{
	return !(lhs == rhs);
}

/*
 * Private member functions
*/
void Cube2Pieces::updateCornerOrientation(Position pos, uint8_t numRotate)
{
	// Read the preface at the top of the header file. The value we assign here is the number of clockwise
	// twists we need to perform on a corner until the corner is oriented correctly.j
	corners[static_cast<uint8_t>(pos)].orientation = (corners[static_cast<uint8_t>(pos)].orientation + numRotate) % 3; 
}

/*
 * Normalizing experimentation
*/

Cube2Pieces& Cube2Pieces::cubeRotateX()
{
	rotateR();
	rotateLi();
	return *this;
}

Cube2Pieces& Cube2Pieces::cubeRotateY()
{
	rotateU();
	rotateDi();
	return *this;
}

Cube2Pieces& Cube2Pieces::cubeRotateZ()
{
	rotateF();
	rotateBi();
	return *this;
}

Cube2Pieces& Cube2Pieces::cubeRotateXi()
{
	rotateRi();
	rotateL();
	return *this;
}

Cube2Pieces& Cube2Pieces::cubeRotateYi()
{
	rotateUi();
	rotateD();
	return *this;
}

Cube2Pieces& Cube2Pieces::cubeRotateZi()
{
	rotateFi();
	rotateB();
	return *this;
}

Cube2Pieces& Cube2Pieces::cubeRotateX2()
{
	rotateR2();
	rotateL2();
	return *this;
}

Cube2Pieces& Cube2Pieces::cubeRotateY2()
{
	rotateU2();
	rotateD2();
	return *this;
}

Cube2Pieces& Cube2Pieces::cubeRotateZ2()
{
	rotateF2();
	rotateB2();
	return *this;
}

Cube2Pieces& Cube2Pieces::normalize()
{
	/*
	 * TODO: probably want to track the position index of WRB as a member variable
	 * 
	 * The idea is that the WRB corner piece has some position index. Given a state of the cube
	 * we can rotate the entire cube such that the WRB corner piece is in the URF position with orientation 0.
	 * This allows us to compare two states of the cube that are equivalent up to a rotation of the entire cube.
	 * 
	 * To my understanding, this is particular to cubes of even dimension. For cubes of odd dimension,
	 * there is no way to rotate the faces of the cube to affect a rotation of the entire cube,
	 * so we don't have to worry about normalizing the cube up to rotation (so long as you disallow slice moves).
	 *
	 * Also, done correctly, this means that we can reduce the size of the individual hashes by recognizing that 
	 * since the first entry in the hash is the WRB corner piece, we do not need to store this information;
	 * we only need to hash information about the other 7 pieces.
	*/

	// TODO: this is the part we need to track of as member variable
	// Extract the index of the WRB corner piece
	uint8_t WRBidx = 0;
	while (corners[WRBidx].piece != Piece::WRB)
		WRBidx++;

	/*
	 * Given a position and orientatino of the WRB corner pieces, we can rotate the cube such that WRB is in the URG position
	 * with W in U and R in F (that is, with orientation 0).
	 * The correct rotation for a given positions/orientation was manually computed.
	*/
	Position WRBpos = static_cast<Position>(WRBidx);
	if (WRBpos == Position::URF && corners[WRBidx].orientation == 0)
		; // Already good
	else if (WRBpos == Position::URF && corners[WRBidx].orientation == 1)
	{
		cubeRotateX();
		cubeRotateY();
	}
	else if (WRBpos == Position::URF && corners[WRBidx].orientation == 2)
	{
		cubeRotateXi();
		cubeRotateZi();
	}
	else if (WRBpos == Position::ULF && corners[WRBidx].orientation == 0)
	{
		cubeRotateYi();
	}
	else if (WRBpos == Position::ULF && corners[WRBidx].orientation == 1)
	{
		cubeRotateZ();
	}
	else if (WRBpos == Position::ULF && corners[WRBidx].orientation == 2)
	{
		cubeRotateY2();
		cubeRotateXi();
	}
	else if (WRBpos == Position::ULB && corners[WRBidx].orientation == 0)
	{
		cubeRotateY2();
	}
	else if (WRBpos == Position::ULB && corners[WRBidx].orientation == 1)
	{
		cubeRotateYi();
		cubeRotateZ();
	}
	else if (WRBpos == Position::ULB && corners[WRBidx].orientation == 2)
	{
		cubeRotateXi();
		cubeRotateZ();
	}
	else if (WRBpos == Position::URB && corners[WRBidx].orientation == 0)
	{
		cubeRotateY();
	}
	else if (WRBpos == Position::URB && corners[WRBidx].orientation == 1)
	{
		cubeRotateY2();
		cubeRotateZ();
	}
	else if (WRBpos == Position::URB && corners[WRBidx].orientation == 2)
	{
		cubeRotateXi();
	}
	else if (WRBpos == Position::DRF && corners[WRBidx].orientation == 0)
	{
		cubeRotateX2();
		cubeRotateY();
	}
	else if (WRBpos == Position::DRF && corners[WRBidx].orientation == 1)
	{
		cubeRotateZi();
	}
	else if (WRBpos == Position::DRF && corners[WRBidx].orientation == 2)
	{
		cubeRotateX();
	}
	else if (WRBpos == Position::DLF && corners[WRBidx].orientation == 0)
	{
		cubeRotateZ2();
	}
	else if (WRBpos == Position::DLF && corners[WRBidx].orientation == 1)
	{
		cubeRotateX();
		cubeRotateYi();
	}
	else if (WRBpos == Position::DLF && corners[WRBidx].orientation == 2)
	{
		cubeRotateYi();
		cubeRotateX();
	}
	else if (WRBpos == Position::DLB && corners[WRBidx].orientation == 0)
	{
		cubeRotateX2();
		cubeRotateYi();
	}
	else if (WRBpos == Position::DLB && corners[WRBidx].orientation == 1)
	{
		cubeRotateY2();
		cubeRotateZi();
	}
	else if (WRBpos == Position::DLB && corners[WRBidx].orientation == 2)
	{
		cubeRotateY2();
		cubeRotateX();
	}
	else if (WRBpos == Position::DRB && corners[WRBidx].orientation == 0)
	{
		cubeRotateX2();
	}
	else if (WRBpos == Position::DRB && corners[WRBidx].orientation == 1)
	{
		cubeRotateXi();
		cubeRotateY();
	}
	else if (WRBpos == Position::DRB && corners[WRBidx].orientation == 2)
	{
		cubeRotateY();
		cubeRotateX();
	}
	return *this;
}
