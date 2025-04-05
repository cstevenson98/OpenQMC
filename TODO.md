# OpenQMC TODO List

## PIMPL Implementation Tasks

### High Priority
- [X] Convert `Sparse` class to use PIMPL idiom
  - Create `SparseImpl` class in CUDA/Thrust implementation package
  - Move COO format implementation details to `SparseImpl`
  - Update `Sparse` interface to use `std::unique_ptr<SparseImpl>`
  - Ensure all existing functionality is preserved
  - Update tests to work with new PIMPL structure

- [ ] Implement host data access methods across all matrix classes
  - Add `GetHostData()` method to each class returning const reference to host data:
    - `Vect`: Return `const t_hostVect&`
    - `Dense`: Return `const t_hostMat&`
    - `Sparse`: Return `const std::vector<COOTuple>&`
    - `SparseELL`: Return `const t_hostMat&`
  - Add `CoeffRef()` method to each class for mutable access:
    - `Vect`: Return `std::complex<double>&`
    - `Dense`: Return `std::complex<double>&`
    - `Sparse`: Return `std::complex<double>&` (with COO tuple handling)
    - `SparseELL`: Return `std::complex<double>&`
  - Add appropriate const correctness
  - Update tests to verify host data access
  - Document memory ownership semantics

### Code Organization
- [ ] Review and standardize PIMPL implementation across all matrix classes
  - Ensure consistent naming conventions
  - Standardize method signatures between interface and implementation
  - Document PIMPL pattern usage in codebase

### Documentation
- [ ] Add detailed documentation for PIMPL pattern usage
  - Explain benefits of PIMPL in this codebase
  - Document CUDA/Thrust implementation details
  - Add examples of how to extend the pattern for new classes

### Testing
- [ ] Ensure comprehensive test coverage for all PIMPL implementations
  - Add tests for move/copy semantics
  - Verify ABI stability
  - Test CUDA/Thrust functionality

### Future Considerations
- [ ] Consider adding more sparse matrix formats (CSR, CSC)
- [ ] Evaluate performance impact of PIMPL pattern
- [ ] Consider adding GPU memory management optimizations 