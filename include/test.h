#ifndef TEST_H
#define TEST_H

#include <string>
#include <vector>

// Function declarations that will be implemented in test.cu
std::string getTestString();
std::vector<int> getTestVector();
int addNumbers(int a, int b);
void printTestMessage();

#endif  // TEST_H