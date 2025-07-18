# Intro to Julia 

[Julia Intro for QNumerics Summer School](https://rayegun.github.io/JuliaIntro/)
by Raye Kimmerer.

## Other resources

[Julia cheat sheet](https://cheatsheet.juliadocs.org/)

[Modern Julia Workflows](https://modernjuliaworkflows.org/)

[Performant Serial Julia](https://mitmath.github.io/Parallel-Computing-Spoke/notebooks/PerformantSerialJulia.html)

[What scientists must know about hardware to write fast code](https://github.com/jakobnissen/hardware_introduction)

## Linear Algebra 

[LinearAlgebra docs](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)

```@example la
using LinearAlgebra
using SparseArrays
using InteractiveUtils

A = rand(100,100)
```
Transpose and other operations are non-allocating, and instead create a wrapper
of the original array. This prevents unecessary allocation and allows more 
opportunity to dispatch on array types.

```@example la
B = transpose(A)
```

```@example la
C = copy(B)
```
When we copy the array it allocates and is no longer a wrapper

```@example la
Diagonal(A)
```

```@example la
subtypes(LinearAlgebra.Factorization)
```

```@example la
issymmetric(A*A')
```
How do we communicate this to the dispatch system?

```@example la
D = Symmetric(A*A')
```
Now we can dispatch on the fact that this matrix is symmetric, allowing us to
select more efficient algorithms for this case.

```@example la
Symmetric(A)
```
This is 'casting' to a symmetric matrix, it will happily create a symmetric
matrix out of a non-symmetric one.

## Sparse Arrays

```@example la
A = sprand(100,100,0.1)
```

