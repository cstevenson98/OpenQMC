module;

#include <iostream>
#include <string>

export module Math;

export namespace math {
void print(const std::string& message) {
  std::cout << "Math module says: " << message << std::endl;
}

}  // namespace math