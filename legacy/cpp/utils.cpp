#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <sstream>

#include "utils.h"

std::string trimWhitespace(const std::string& str) {
	size_t first = str.find_first_not_of(' ');
	size_t last = str.find_last_not_of(' ');

	if (first == std::string::npos || last == std::string::npos) {
		// No non-whitespace characters found
		return "";
	}
	return str.substr(first, last - first + 1);
}

std::vector<std::string> split(const std::string& s, char delimiter) {
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delimiter)) {
		tokens.push_back(token);
	}
	return tokens;
}
