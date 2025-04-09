#include "../include/test.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

// Implementation of the functions declared in test.h

std::string getTestString() { return "Hello from test.cu!"; }

std::vector<int> getTestVector() {
  // Create a vector with some test values
  std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // Use std::random_device and std::shuffle to randomize the vector
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(vec.begin(), vec.end(), std::default_random_engine(seed));

  return vec;
}

int addNumbers(int a, int b) { return a + b; }

void printTestMessage() {
  std::cout << "This is a test message from test.cu" << std::endl;

  // Demonstrate using std::algorithm
  std::vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
  std::cout << "Original vector: ";
  for (int num : numbers) {
    std::cout << num << " ";
  }
  std::cout << std::endl;

  // Use std::sort to sort the vector
  std::sort(numbers.begin(), numbers.end());

  std::cout << "Sorted vector: ";
  for (int num : numbers) {
    std::cout << num << " ";
  }
  std::cout << std::endl;

  // Use std::find to find an element
  auto it = std::find(numbers.begin(), numbers.end(), 5);
  if (it != numbers.end()) {
    std::cout << "Found 5 at position: " << (it - numbers.begin()) << std::endl;
  }
}
