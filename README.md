# sun_diffusion
Diffusion models for ${\rm SU}(N)$ degrees of freedom.

## Conventions
Some notes on mathematical conventions (which can definitely be changed):
- The exponential map $\exp: \mathfrak{su}(N) \to {\rm SU}(N)$ is defined conventionally as $$U := \exp(iA)$$, where $A = A^\dagger$ is a hermitian matrix. In the math literature, one absorbs the factor of $i$ into the algebra-valued matrix $A$ and calls this an anti-hermitian matrix, but the physics convention makes explicit the imaginary unit in $\exp$. We adopt this convention automatically, so our functions that map between group and algebra, namely `sun.matrix_exp()` and `sun.matrix_log()`, expect a Hermitian matrix to be input.

## Installation
### CPU only (default)
```bash
pip install sun_diffusion
```

### GPU (CUDA) Users
Before installing, make sure to install a CUDA-enabled PyTorch compatible with your GPU.
For example, for CUDA 12.4:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install sun_diffusion[gpu]
```
If PyTorch cannot detect a GPU, your code will fall back to CPU, or `set_device('cuda')` will raise an error. You can verify CUDA availability with a small example:
```python
from sun_diffusion.devices import set_device, summary, HAS_CUDA

# Check CUDA availability
print('CUDA available:', HAS_CUDA)
if HAS_CUDA:
    set_device('cuda', 0)
else:
    set_device('cpu')

print(summary())
```

## Quickstart / Examples
This package allows one to define actions and evaluate them on batches of ${\rm SU}(N)$ configurations:
```python
from sun_diffusion.action import SUNToyAction
from sun_diffusion.sun import random_sun_element

# Create a toy action
action = SUNToyAction(beta=1.0)

# Random batch of SU(3) matrices
batch_size = 4
U = random_sun_element(batch_size, Nc=3)

# Evaluate the action
S = action(U)
print(S)
```
```python-repl
>>> tensor([-0.0338, -0.0705, -0.5711, -0.7625])
```
