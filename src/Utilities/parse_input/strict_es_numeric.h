// Small, opt-in numerical checks used by screened protein electrostatics.
// Does not alter the common oxDNA input readers or their existing semantics.
#ifndef OXDNA_STRICT_ES_NUMERIC_H
#define OXDNA_STRICT_ES_NUMERIC_H

#include "../../defs.h"
#include "../oxDNAException.h"
#include "parse_input.h"

#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace oxdna_es_numeric {

inline std::string es_trim(const std::string &input) {
	const auto first = input.find_first_not_of(" \t\r\n");
	if(first == std::string::npos) return "";
	const auto last = input.find_last_not_of(" \t\r\n");
	return input.substr(first, last - first + 1);
}

inline double es_finite_double(const std::string &raw, const std::string &where) {
	const std::string token = es_trim(raw);
	if(token.empty()) throw oxDNAException("Missing number in %s", where.c_str());
	// Also reject nonfinite spelling lexically, independent of -ffast-math.
	for(unsigned char c : token) {
		if(!(std::isdigit(c) || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E')) {
			throw oxDNAException("Invalid numeric token '%s' in %s", token.c_str(), where.c_str());
		}
	}
	errno = 0;
	char *end = nullptr;
	const double value = std::strtod(token.c_str(), &end);
	if(end == token.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(value)) {
		throw oxDNAException("Invalid or nonfinite number '%s' in %s", token.c_str(), where.c_str());
	}
	return value;
}

inline int es_int(const std::string &raw, const std::string &where) {
	const std::string token = es_trim(raw);
	if(token.empty()) throw oxDNAException("Missing integer in %s", where.c_str());
	try {
		size_t used = 0;
		long long value = std::stoll(token, &used, 10);
		if(used != token.size() || value < std::numeric_limits<int>::min() ||
				value > std::numeric_limits<int>::max()) throw std::out_of_range("int");
		return static_cast<int>(value);
	}
	catch(const std::exception &) {
		throw oxDNAException("Invalid integer '%s' in %s", token.c_str(), where.c_str());
	}
}

inline number es_checked_number(long double value, const std::string &where) {
	if(!std::isfinite(value) ||
			std::fabs(value) > static_cast<long double>(std::numeric_limits<number>::max())) {
		throw oxDNAException("Protein electrostatics numeric overflow in %s", where.c_str());
	}
	return static_cast<number>(value);
}

inline void es_optional_number(input_file &inp, const char *key, number &target) {
	std::string source;
	if(getInputString(&inp, key, source, 0) != KEY_FOUND) return;
	target = es_checked_number(es_finite_double(source, key), key);
}

} // namespace oxdna_es_numeric
#endif // OXDNA_STRICT_ES_NUMERIC_H
