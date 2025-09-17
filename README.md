# sun_diffusion
Diffusion models for compact variables and SU(N) lattice field theories.

## Internal Notes
Notes on project goals, meetings, and progress can be found [here](https://docs.google.com/document/d/1-lY-2Q8via4YxJPQvmPctFySrqVRTgI--a8oGPFtGko/edit?usp=sharing).

## Conventions
Some notes on mathematical conventions (which can definitely be changed):
- The exponential map $\exp: \mathfrak{su}(N) \to {\rm SU}(N)$ is defined conventionally as $$U := \exp(iA)$$, where $A = A^\dagger$ is a hermitian matrix. In the math literature, one absorbs the factor of $i$ into the algebra-valued matrix $A$ and calls this an anti-hermitian matrix, but the physics convention makes explicit the imaginary unit in $\exp$. We adopt this convention automatically, so our functions that map between group and algebra, namely `sun.matrix_exp()` and `sun.matrix_log()`, expect a Hermitian matrix to be input. 
