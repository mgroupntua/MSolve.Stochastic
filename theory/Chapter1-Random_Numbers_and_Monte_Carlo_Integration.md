% include mathjax.html %
# Random Numbers: Meaning and Generation

Random numbers (RNs) are used in a variety of applications involving computer generation of statistical data. In this section, we explain the underlying ideas concentrating on the meaning and generation of random numbers. We start with a simple illustration of the role of statistics in the numerical solution of deterministic problems. 
## Monte Carlo Integration
We wish to evaluate the integral

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;I=\int_{0}^{1}g(x)dx" title="I=\int_{0}^{1}g(x)dx" />(1)

For this purpose, we introduce a random variable x with uniform distribution in the interval (0, 1) and we form the random variable y = g(x). As we know,

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;E\left&space;\{&space;g(x)&space;\right&space;\}=\int_{0}^{1}g(x)f_{x}(x)dx=\int_{0}^{1}g(x)dx" title="E\left \{ g(x) \right \}=\int_{0}^{1}g(x)f_{x}(x)dx=\int_{0}^{1}g(x)dx" />(2)

We have thus expressed the unknown I as the expected value of the random variable y. This result involves only concepts; it does not yield a numerical method for evaluating I. Suppose, however, that the random variable x models a physical quantity in a real experiment. We can then estimate I using the relative frequency interpretation of expected values: We repeat the experiment a large number of times and observe the values Xi of x; we compute the corresponding values Yi =g(Xi) of y and form their average. This yields

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;I=E\left&space;\{&space;g(x)&space;\right&space;\}=\frac{1}{n}\sum&space;g(x^{i})" title="I=E\left \{ g(x) \right \}=\frac{1}{n}\sum g(x^{i})" />(3)

This suggests the method described next for determining I: The data xi, no matter how they are obtained, are random numbers; that is, they are numbers having certain properties. If, therefore, we can numerically generate such numbers, we have a method for determining I. To carry out this method we must develop computer programs for generating random numbers.

## Generation of Random Number Sequences
Random numbers used in Monte Carlo calculations are generated mainly by computer programs; however, they can also be generated as observations of random data obtained from real experiments: The tosses of a fair coin generate a random sequence of 0's (heads) and 1's (tails); the distance between radioactive emissions generates a random sequence of exponentially distributed samples. We accept number sequences so generated as random because of our long experience with such experiments. Random number sequences experimentally generated are not however, suitable for computer use, for obvious reasons. An efficient source of random numbers is a computer program with small memory, involving simple arithmetic operations.
Our objective is to generate random number sequences with arbitrary distributions. In the present state of the art, however, this cannot be done directly. The available algorithms only generate sequences consisting of integers zi uniformly distributed in an interval (0, m). The generation of a sequence xi with an arbitrary distribution is obtained indirectly by a variety of methods involving the uniform sequence zi.
The most general algorithm for generating a random number sequence zi is an equation of the form

<img src="https://latex.codecogs.com/gif.latex?z_n=f(z_n,...,z_{n-r})\:&space;mod\:&space;m" title="z_n=f(z_n,...,z_{n-r})\: mod\: m" />(4)

where <img src="https://latex.codecogs.com/gif.latex?z_n=f(z_n,...,z_{n-r})" title="z_n=f(z_n,...,z_{n-r})" />   is a function depending on the r most recent past values of zn. In this notation, zn is the remainder of the division of the number   is a function depending on the r most recent past values of zn. In this notation, zn is the remainder of the division of the number by <img src="https://latex.codecogs.com/gif.latex?m" title="m" />This is a nonlinear recursion expressing zn in terms of the constant m, the function f And the initial conditions z1,…,zr-1. The quality of the generator depends on the form of the function f. It might appear that good random number sequences result if this function is complicated. Experience has shown, however. that this is not the case. Most algorithms in use are linear recursions of order 1. MSolve uses the Nuget Troschuetz random number generator package (https://www.nuget.org/packages/Troschuetz.Random/) for the generation of random numbers of arbitrary distribution. 
