# BivMatFun

```BivMatFun``` is a Julia package to evaluate bivariate-matrix functions. 
To install the package into your Julia environment you may run:
```julia
import Pkg;
Pkg.add(url = "https://github.com/numpi/BivMatFun.git")
```
or, alternatively, enter the ```pkg``` mode by typing ```]```, and then 
running:
```julia
pkg> add https://github.com/numpi/BivMatFun.git
```

The function ```fun2m``` can be used to evaluate a bivariate 
matrix function as follows:
```julia
using BivMatFun;
X = fun2m(f, A, B, C)
```
where ```f(x,y,i,j)``` computes the (i,j)-th derivative of the desired
function. Unless the method based on Taylor expansions is specifically
chosen by setting the optional parameter ```method = BivMatFun.Taylor```, 
the ```(i,j)``` parameters can be ignored, as they will always be equal 
to ```(0,0)```· 

## Tests

You may run all the tests, which repeat the experiments included in [1].
In order to do that, run the following commands (as above, in `pkg` mode, which can be
enabled by typing `]`):
```julia
pkg> activate .
pkg> test
```
Please note that the tests run the solver several times for getting accurate
timings, and compute the reference solution using high precision arithmetic. 
The latter operation is quite computationally demanding, so they may require 
several minutes to complete. 

## References

* S. Massei, L. Robol, _Recursive Block Diagonalization for evaluating bivariate functions of matrices_, in preparation, 2021.
