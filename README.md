# OpenQMC (v0.1)
Library in C++ for open quantum systems simulations on CUDA GPUs

The main goals of this project are 1) to create a functional and minimal C++ library
for the simulation (time evolution, steady-states, eigenspectrum, multi-time correlators) of
open quantum systems and 2) have these simulations benefit from GPU acceleration wherever possible.

Many body open systems of course suffer from exponential scaling in their degrees of freedom, so 
large systems are out of the question to solve exactly. While brute forcing such calculations
is never going to be feasible on classical hardware, we can make attempts to dimish the size of the 
exponent which reduces the number of particles we can solve at once. One straightforward technique is to use sparse
array representations for operators and density matrices.
This saves computationally in terms of both space and time. Using a GPU can in some cases give us 
significant speed ups if we use it for parallel calculations. Sparse matrix multiplcation is such a
calculation ammenable to parallelisation on the GPU.

I wanted to write this library to provide a strong base to doing the really common computations
needed for open quantum systems reaseach. I take a lot of inspiration from QuTiP, and use it's
naming conventions for functions and constants. The difference is that this uses CUDA to accelerate
the calculations, so if your group have access to GPUs, this may be of use.

To-do
====

 - Implement a test suite for robustness
 - Include example problems

To build (Cmake)
---

With Cmake installed:

~~~
make
~~~

and then run relevant executables as

~~~
./build/main
~~~
