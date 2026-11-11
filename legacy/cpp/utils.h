#pragma once

#include <string>
#include <cstdint>
#include <vector>

/* --------------------------------------------------------------------------------------------
 * This file contains helpful utility functions, for things like string parsing.
 * --------------------------------------------------------------------------------------------
*/

// Swap two items that support copy construction and assignment.
template <typename T> 
void swap_items(T& a, T& b) {
	T temp = a;
	a = b;
	b = temp;
}

// Removes leading and trailing whitespace from a string.
std::string trimWhitespace(const std::string& str);

// Splits a string into a vector of strings, using a delimiter.
std::vector<std::string> split(const std::string& s, char delimiter);

