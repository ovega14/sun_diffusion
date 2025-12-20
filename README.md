# sun_diffusion
Diffusion models for ${\rm SU}(N)$ degrees of freedom.

What this package does:
- Implements diffusion processes on the ${\rm SU}(N)$ Lie group manifold with
  - different noise schedules
  - customizable parameters
- Includes utilities for manipulating matrices (and their spectra) on ${\rm SU}(N)$ and $\mathfrak{su}(N)$, including
  - diagonalization
  - canonicalization
  - eigendecomposition
  - algebra-to-group projections (and vice versa)
  - irreducible representations
- Provides heat kernel evaluations, sampling routines, and score functions


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
```python-repl
>>> CUDA available: True
>>> Using device: cuda:0 (NVIDIA GH200 120GB) with dtype: torch.float32
```
Further utilities for handling devices and dtypes can be found in the [`sun_diffusion.devices`](https://github.com/ovega14/sun_diffusion/blob/main/sun_diffusion/devices.py) module.

## Quickstart / Examples
**Note:** More in-depth examples can be found in the [`notebooks`](https://github.com/ovega14/sun_diffusion/blob/main/notebooks/) of this repository.

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
See the [`sun_diffusion.action`](https://github.com/ovega14/sun_diffusion/blob/main/sun_diffusion/action.py) module for more details.

This package also enables users to define diffusion processes on Euclidean space:
```python
import torch
from sun_diffusion.diffusion import VarianceExpandingDiffusion

batch_size = 512
x_0 = 0.1 * torch.randn((batch_size, 3))

# Diffuse x_0 -> x_1 on R^3
diffuser = VarianceExpandingDiffusion(kappa=1.1)
x_1 = diffuser(x_0, t=torch.ones(batch_size))
print('x_0 std =', x_0.std().item())
print('x_1 std =', x_1.std().item())
```
```python-repl
>>> x_0 std = 0.10194464772939682
>>> x_1 std = 1.0630394220352173
```
as well as on the ${\rm SU}(N)$ manifold:
```python
from sun_diffusion.diffusion import PowerDiffusionSUN

batch_size = 512
U_0 = random_sun_element(batch_size, Nc=2, scale=0.1)

# Diffuse U_0 -> U_1 on SU(2)
diffuser = PowerDiffusionSUN(kappa=3.0, alpha=1.0)
U_1, xs, V = diffuser(U_0, torch.ones(batch_size))
```
See [`sun_diffusion.diffusion`](https://github.com/ovega14/sun_diffusion/blob/main/sun_diffusion/diffusion.py) for more diffusion processes and implementation details.

Sampling from the ${\rm SU}(N)$ heat kernel over the diagonal subalgebra of eigenangles is also simple, and can easily be combined with this package's matrix algebra utilities to produce group elements:
```python
from sun_diffusion.heat import sample_sun_hk
from sun_diffusion.linalg import adjoint
from sun_diffusion.sun import random_un_haar_element, embed_diag, matrix_exp

# Batch of HK eigenangles
batch_size = 2
Nc = 2
xs = sample_sun_hk(batch_size, Nc=Nc, width=torch.rand(batch_size), n_iter=10)

# Promote to Algebra -> project to SU(N)
X = embed_diag(torch.tensor(xs))
V = random_un_haar_element(batch_size, Nc=Nc)
U = V @ matrix_exp(X) @ adjoint(V)
print(U)
```
```python-repl
>>> tensor([[[ 0.8995-0.3124j,  0.0070+0.3055j],
         [-0.0070+0.3055j,  0.8995+0.3124j]],

        [[ 0.9748+0.1920j,  0.0399+0.1061j],
         [-0.0399+0.1061j,  0.9748-0.1920j]]])
```
See the [`sun_diffusion.heat`](https://github.com/ovega14/sun_diffusion/blob/main/sun_diffusion/heat.py), [`sun_diffusion.linalg`](https://github.com/ovega14/sun_diffusion/blob/main/sun_diffusion/linalg.py) and [`sun_diffusion.sun`](https://github.com/ovega14/sun_diffusion/blob/main/sun_diffusion/sun.py) modules for more.
 
## Conventions
Some notes on mathematical conventions (which can definitely be changed):
- The exponential map $\exp: \mathfrak{su}(N) \to {\rm SU}(N)$ is defined conventionally as $$U := \exp(iA)$$, where $A = A^\dagger$ is a hermitian matrix. In the math literature, one absorbs the factor of $i$ into the algebra-valued matrix $A$ and calls this an anti-hermitian matrix, but the physics convention makes explicit the imaginary unit in $\exp$. We adopt this convention automatically, so our functions that map between group and algebra, namely `sun.matrix_exp()` and `sun.matrix_log()`, expect a Hermitian matrix to be input.
