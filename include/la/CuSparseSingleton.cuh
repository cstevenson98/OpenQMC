//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#pragma once

#include <cusparse.h>
#include <stdexcept>

/**
 * @brief Singleton class to manage the cuSPARSE handle.
 * This ensures we only create one handle for the entire application.
 */
class CuSparseSingleton {
public:
  /**
   * @brief Get the singleton instance.
   *
   * @return CuSparseSingleton& Reference to the singleton instance.
   */
  static CuSparseSingleton &getInstance() {
    static CuSparseSingleton instance;
    return instance;
  }

  /**
   * @brief Get the cuSPARSE handle.
   *
   * @return cusparseHandle_t The cuSPARSE handle.
   */
  cusparseHandle_t getHandle() const { return handle; }

  /**
   * @brief Get the default matrix descriptor.
   *
   * @return cusparseMatDescr_t The default matrix descriptor.
   */
  cusparseMatDescr_t getDefaultMatDescr() const { return defaultMatDescr; }

  /**
   * @brief Delete copy constructor and assignment operator.
   */
  CuSparseSingleton(const CuSparseSingleton &) = delete;
  CuSparseSingleton &operator=(const CuSparseSingleton &) = delete;

private:
  /**
   * @brief Private constructor to prevent direct instantiation.
   */
  CuSparseSingleton() {
    // Create cuSPARSE handle
    cusparseStatus_t status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create cuSPARSE handle");
    }

    // Create default matrix descriptor
    status = cusparseCreateMatDescr(&defaultMatDescr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      cusparseDestroy(handle);
      throw std::runtime_error("Failed to create cuSPARSE matrix descriptor");
    }

    // Set matrix type and index base
    cusparseSetMatType(defaultMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(defaultMatDescr, CUSPARSE_INDEX_BASE_ZERO);
  }

  /**
   * @brief Destructor to clean up cuSPARSE resources.
   */
  ~CuSparseSingleton() {
    cusparseDestroyMatDescr(defaultMatDescr);
    cusparseDestroy(handle);
  }

  cusparseHandle_t handle;            ///< cuSPARSE handle
  cusparseMatDescr_t defaultMatDescr; ///< Default matrix descriptor
};