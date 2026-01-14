#include "util.h"

#include <algorithm>
#include <cctype>

std::string nam::util::lowercase(const std::string& s) {
  std::string out(s);
  std::transform(s.begin(), s.end(), out.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return out;
}
