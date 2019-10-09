# Fourier Series and Karhunen-Loeve Expansions

A process x(t) is MS periodic with period T if <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;E\left&space;\{&space;\left&space;|&space;\mathbf{x(t&plus;T)-x(t)}&space;\right&space;|^{2}&space;\right&space;\}=0" title="E\left \{ \left | \mathbf{x(t+T)-x(t)} \right |^{2} \right \}=0" />  for all t. A Wide sense stationary (WSS) process is MS periodic if its autocorrelation R(τ) is periodic with period T  = 2π/ω0. Expanding R(r) into a Fourier series, we obtain

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;R\left&space;(&space;\tau&space;\right&space;)=\sum_{n=-\bowtie&space;}^{\bowtie&space;}\gamma&space;_{n}e^{jn\omega&space;_{0}\tau&space;}\;" title="R\left ( \tau \right )=\sum_{n=-\bowtie }^{\bowtie }\gamma _{n}e^{jn\omega _{0}\tau }\;" />   where  <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;\gamma&space;_{n}=\frac{1}{T}\int_{0}^{T}R(\tau&space;)e^{-jn\omega_0\tau&space;}d\tau" title="\gamma _{n}=\frac{1}{T}\int_{0}^{T}R(\tau )e^{-jn\omega_0\tau }d\tau" />(1)

Given a WSS periodic process x(t) with period T. we form the sum

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;\hat{x}(t)=\sum_{n=-\bowtie&space;}^{\bowtie}c_{n}e^{jn\omega&space;_{0}\tau&space;}" title="\hat{x}(t)=\sum_{n=-\bowtie }^{\bowtie}c_{n}e^{jn\omega _{0}\tau }" /> where <img src="https://latex.codecogs.com/gif.latex?\inline&space;c^{_{n}}=\frac{1}{T}\int_{0}^{T}\mathbf{x}(\tau&space;)e^{-jn\omega&space;_{0}\tau&space;}d\tau" title="c^{_{n}}=\frac{1}{T}\int_{0}^{T}\mathbf{x}(\tau )e^{-jn\omega _{0}\tau }d\tau" />(2)

This sum equals x(t) in the MS sense:

<img src="https://latex.codecogs.com/gif.latex?\inline&space;E\left&space;\{&space;\left&space;|&space;\mathbf{x}(t)-\hat{\mathbf{x}}(t))&space;\right&space;|^{2}&space;\right&space;\}=0" title="E\left \{ \left | \mathbf{x}(t)-\hat{\mathbf{x}}(t)) \right |^{2} \right \}=0" />(3)

Furthermore, the random variables cn are uncorrelated with zero mean for n ≠ 0, and
their variance equals γn:

<img src="https://latex.codecogs.com/gif.latex?\inline&space;E\left&space;\{&space;\mathbf{c_{\mathbf{n}}}&space;\right&space;\}=\left\{\begin{matrix}&space;n_{x}&space;&&space;n=0\\0&space;&n\neq&space;0&space;\end{matrix}\right." title="E\left \{ \mathbf{c_{\mathbf{n}}} \right \}=\left\{\begin{matrix} n_{x} & n=0\\0 &n\neq 0 \end{matrix}\right." /> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;E\left&space;\{&space;\mathbf{c_{\mathbf{n}}}\mathbf{c_{\mathbf{m}}^{*}}&space;\right&space;\}=\left\{\begin{matrix}&space;\gamma&space;_{n}&space;&&space;n=m\\0&space;&n\neq&space;m&space;\end{matrix}\right." title="E\left \{ \mathbf{c_{\mathbf{n}}}\mathbf{c_{\mathbf{m}}^{*}} \right \}=\left\{\begin{matrix} \gamma _{n} & n=m\\0 &n\neq m \end{matrix}\right." /> (4)
 

## The Karhunen-Loeve Expansion
The Fourier series is a special case of the expansion of a process x(t) into a series of the
form

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathbf{\hat{x}}(t)=\sum_{n=1}^{\infty}\mathbf{c}_n\phi_n(t)\;\;\;\;\;0<t<T" title="\mathbf{\hat{x}}(t)=\sum_{n=1}^{\infty}\mathbf{c}_n\phi_n(t)\;\;\;\;\;0<t<T" />(5)

For this purpose, we introduce a random variable x with uniform distribution in the interval (0, 1) and we form the random variable y = g(x). As we know,


<img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;E\left&space;\{&space;g(x)&space;\right&space;\}=\int_{0}^{1}g(x)f_{x}(x)dx=\int_{0}^{1}g(x)dx" title="E\left \{ g(x) \right \}=\int_{0}^{1}g(x)f_{x}(x)dx=\int_{0}^{1}g(x)dx" />(6)

where φn(t) is a set of orthonormal functions in the interval (0, T);

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\int_{0}^{T}\phi&space;_{n}(t)\phi_{m}^{*}(t)dt=\delta[n-m]" title="\int_{0}^{T}\phi _{n}(t)\phi_{m}^{*}(t)dt=\delta[n-m]" />(7)

and the coefficients cn are random variables given by

<img src="https://latex.codecogs.com/gif.latex?\inline&space;c_{n}=\int_{0}^{T}\mathbf{x}(t)\phi&space;_{n}^{*}dt" title="c_{n}=\int_{0}^{T}\mathbf{x}(t)\phi _{n}^{*}dt" />(8)


In this development, we consider the problem of determining a set of ortbononnal functions
φn(t) such that: (a) the sum in Eq.9  equals x(t); (b) the coefficients cn are
orthogonal.
To solve this problem, we form the integral equation

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\int_{0}^{T}R(t_{1},t_{2})&space;\phi(t_2)dt_2=\lambda&space;\phi(t_1)&space;\;\;\;\;0<t_1<T" title="\int_{0}^{T}R(t_{1},t_{2}) \phi(t_2)dt_2=\lambda \phi(t_1) \;\;\;\;0<t_1<T" />(9)

where R(t1,t2) is the autocorrelation of the process x(t). It is well known from the theory of integral equations that the eigenfunctions cn(t) of Eq.12 are orthonormal as in Eq.10 and they satisfy the identity

<img src="https://latex.codecogs.com/gif.latex?\inline&space;R(t,t)=\sum_{n=1}^{\infty&space;}\lambda_n|\phi_n(t)|^2" title="R(t,t)=\sum_{n=1}^{\infty }\lambda_n|\phi_n(t)|^2" />(10)

The sum in Eq.9 is called the Karhunen-Loeve (K-L) expansion of the process x(t). In this expansion, x(t) need not be stationary. If it is stationary then the origin can be chosen arbitrarily. For further reading on theoretical concepts the reader is referred to Athanasios Papoulis, S. Unnikrishna Pillai - Probability, Random Variables and Stochastic Processes-McGraw-Hill Europe (2002).
