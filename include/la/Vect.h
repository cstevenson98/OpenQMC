//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_VECT_CUH
#define MAIN_VECT_CUH

#include <complex>
#include <memory>
#include <vector>

#include "core/types.h"

class VectImpl;

/**
 * @brief A class representing vectors. Supports vector algebra and uses
 * complex numbers.
 */
class Vect {
 public:
  /**
   * @brief Default constructor to initialize an empty Vect.
   */
  explicit Vect();

  /**
   * @brief Constructor to initialize Vect with given size.
   *
   * @param N Size of the vector.
   */
  explicit Vect(unsigned int N);

  /**
   * @brief Constructor to initialize Vect with given data.
   *
   * @param in Input vector data.
   */
  explicit Vect(t_hostVect &in);

  /**
   * @brief Destructor for Vect.
   */
  ~Vect();

  /**
   * @brief Copy constructor for Vect.
   *
   * @param other Another Vect object to copy from.
   */
  Vect(const Vect &other);

  /**
   * @brief Move constructor for Vect.
   *
   * @param other Another Vect object to move from.
   */
  Vect(Vect &&other) noexcept;

  /**
   * @brief Copy assignment operator for Vect.
   *
   * @param other Another Vect object to copy from.
   * @return Vect& Reference to the current object.
   */
  Vect &operator=(const Vect &other);

  /**
   * @brief Move assignment operator for Vect.
   *
   * @param other Another Vect object to move from.
   * @return Vect& Reference to the current object.
   */
  Vect &operator=(Vect &&other) noexcept;

  /**
   * @brief Computes the conjugate of the Vect.
   *
   * @return Vect Conjugated vector.
   */
  Vect Conj() const;

  /**
   * @brief Overloaded addition operator for Vect objects.
   *
   * @param A Another Vect object to add.
   * @return Vect Result of the addition.
   */
  Vect operator+(const Vect &A) const;

  /**
   * @brief Overloaded subtraction operator for Vect objects.
   *
   * @param A Another Vect object to subtract.
   * @return Vect Result of the subtraction.
   */
  Vect operator-(const Vect &A) const;

  /**
   * @brief Overloaded multiplication operator for scalar multiplication.
   *
   * @param alpha Scalar value to multiply.
   * @return Vect Result of the scalar multiplication.
   */
  Vect operator*(const t_cplx &alpha) const;

  /**
   * @brief Overloaded subscript operator to access vector elements.
   *
   * @param i Index of the element.
   * @return std::complex<double> Element at the specified position.
   */
  std::complex<double> operator[](unsigned int i) const;

  /**
   * @brief Computes the dot product of two Vect objects.
   *
   * @param A Another Vect object.
   * @return double Dot product result.
   */
  std::complex<double> Dot(const Vect &A) const;

  /**
   * @brief Computes the norm of the Vect.
   *
   * @return double Norm of the vector.
   */
  double Norm() const;

  /**
   * @brief Prints the Vect.
   *
   * @param kind Type of data to print (real, imaginary, etc.).
   */
  void Print(unsigned int kind) const;

  /**
   * @brief Prints the real part of the Vect.
   */
  void PrintRe() const;

  /**
   * @brief Prints the imaginary part of the Vect.
   */
  void PrintIm() const;

  /**
   * @brief Prints the absolute value of the Vect.
   */
  void PrintAbs() const;

  /**
   * @brief Gets the size of the Vect.
   *
   * @return int Size of the vector.
   */
  int size() const;

  /**
   * @brief Gets the host data of the Vect.
   *
   * @return const t_hostVect& Reference to the host data.
   */
  const t_hostVect &GetHostData() const;

  /**
   * @brief Gets the data of the Vect.
   *
   * @return std::vector<std::complex<double>> Data of the vector.
   */
  std::vector<std::complex<double>> GetData() const;

 private:
  std::unique_ptr<VectImpl> pImpl;
  Vect(std::unique_ptr<VectImpl> pImpl);
  friend class Sparse;  // Allow Sparse to access private members
};

/**
 * @brief Overloaded multiplication operator for scalar multiplication.
 *
 * @param alpha Scalar value to multiply.
 * @param rhs Vect object to multiply.
 * @return Vect Result of the scalar multiplication.
 */
Vect operator*(const t_cplx &alpha, const Vect &rhs);

#endif  // MAIN_VECT_CUH
