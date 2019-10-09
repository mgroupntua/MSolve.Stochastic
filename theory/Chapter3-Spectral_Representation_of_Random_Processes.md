# Spectral Representation of Random Processes

The Fourier transform of a stochastic process x(t) is a stochastic process X(ω) given by

<img src="https://latex.codecogs.com/gif.latex?\inline&space;X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega&space;t}dt" title="X(\omega)=\int_{-\infty}^{\infty}x(t)e^{-j\omega t}dt" />(1)

The integral is interpreted as an MS limit. Reasoning as in Eq.3, Ch.2, we can show that
(inversion formula)

<img src="https://latex.codecogs.com/gif.latex?\mathbf{x}(t)=\frac{1}{2\pi}&space;\int_{-\infty}^{\infty}&space;\mathbf{X}(\omega)e^{-j\omega&space;t}d\omega" title="\mathbf{x}(t)=\frac{1}{2\pi} \int_{-\infty}^{\infty} \mathbf{X}(\omega)e^{-j\omega t}d\omega" />(2)

in the MS sense.

Let f0(t) be a ID-IV stationary stochastic process with mean value equal to zero, autocorrelation func-tion Rf0f0(τ) and two-sided power spectral density function Sf0f0(ω). Then the following relations hold:

<img src="https://latex.codecogs.com/gif.latex?E[f_0&space;(t)]=0" title="E[f_0 (t)]=0" />(3)

<img src="https://latex.codecogs.com/gif.latex?E[f_0&space;(t&plus;\tau)f_0(t)]=R_{f_0&space;f_0}(\tau)" title="E[f_0 (t+\tau)f_0(t)]=R_{f_0 f_0}(\tau)" />(4)

<img src="https://latex.codecogs.com/gif.latex?\inline&space;S_{f_0&space;f_0}(\omega)=\frac{1}{2\pi}&space;\int_{-\infty}^{\infty}R_{f_0&space;f_0}(\omega)e^{-j\omega&space;\tau}d&space;\tau" title="S_{f_0 f_0}(\omega)=\frac{1}{2\pi} \int_{-\infty}^{\infty}R_{f_0 f_0}(\omega)e^{-j\omega \tau}d \tau" />(5)

<img src="https://latex.codecogs.com/gif.latex?\inline&space;R_{f_0&space;f_0}(\tau)=\int_{-\infty}^{\infty}S_{f_0&space;f_0}(\omega)e^{j\omega&space;\tau}d&space;\omega" title="R_{f_0 f_0}(\tau)=\int_{-\infty}^{\infty}S_{f_0 f_0}(\omega)e^{j\omega \tau}d \omega" />(6)


where the last two equations constitute the well-known Wiener-Khintchine transform pair. 
The following theorem is fundamental in the theory of 1DIV stationary stochastic processes with mean value equal to zero (e.g. Yaglom 1962, Cramer and Leadbetter 1967). 

*To every real-valued ID-IV stationary stochastic process f0t) with mean value equal to zero and two-sided power spectral density function SF0F0(ω), two mutually orthogonal real processes u(ω) and v(ω) with orthogonal increments du(ω) and dv(ω) can be assigned such that:*

<img src="https://latex.codecogs.com/gif.latex?\inline&space;f_0(t)=\int_{0}^{\infty}[cos(\omega&space;t)du(\omega)&plus;sin(\omega&space;t)dv(\omega)]" title="f_0(t)=\int_{0}^{\infty}[cos(\omega t)du(\omega)+sin(\omega t)dv(\omega)]" />(7)

Under certain requirements for the processes u(ω) and v(ω) Eq.7 can be rewritten in the following form: 

<img src="https://latex.codecogs.com/gif.latex?\inline&space;f_0(t)=\sum_{0}^{\infty}[cos(\omega_\kappa&space;t)du(\omega_\kappa)&plus;sin(\omega_\kappa&space;t)dv(\omega_\kappa)]" title="f_0(t)=\sum_{0}^{\infty}[cos(\omega_\kappa t)du(\omega_\kappa)+sin(\omega_\kappa t)dv(\omega_\kappa)]" />(8)

where <img src="https://latex.codecogs.com/gif.latex?\inline&space;\omega_k&space;=&space;k\Delta&space;\omega" title="\omega_k = k\Delta \omega" /> with sufficiently small but finite *Δω*, and if <img src="https://latex.codecogs.com/gif.latex?\inline&space;du(\omega_k)=X_k&space;,\;&space;\;&space;dv(\omega_k)=Y_k" title="du(\omega_k)=X_k ,\; \; dv(\omega_k)=Y_k" /> are independent random variables with mean value equal to zero and standard deviation (2Sf0f0(ωk)Δω)1/2 , then

<img src="https://latex.codecogs.com/gif.latex?\inline&space;f_0&space;(t)=\sum_{k=0}^{\infty}[cos(\omega_\kappa&space;t)X_k&space;&plus;&space;sin(\omega_k&space;t)Y_k]" title="f_0 (t)=\sum_{k=0}^{\infty}[cos(\omega_\kappa t)X_k + sin(\omega_k t)Y_k]" />(9)

where

<img src="https://latex.codecogs.com/gif.latex?\inline&space;X_k&space;=&space;\sqrt2&space;A_k&space;cos\Phi_k" title="X_k = \sqrt2 A_k cos\Phi_k" />(10)

<img src="https://latex.codecogs.com/gif.latex?\inline&space;Y_k=-\sqrt2&space;A_k&space;sin\Phi_k" title="Y_k=-\sqrt2 A_k sin\Phi_k" />(11)

<img src="https://latex.codecogs.com/gif.latex?\inline&space;A_k&space;=&space;(2S_{f_0&space;f_0}{(\omega_k)}\Delta&space;\omega)^{1/2}" title="A_k = (2S_{f_0 f_0}{(\omega_k)}\Delta \omega)^{1/2}" />(12)

and the <img src="https://latex.codecogs.com/gif.latex?\inline&space;\Phi_k" title="\Phi_k" />’s are independent random phase angles uniformly distributed ine the range [0,2π]. *For further reading the reader is referred to Shinozuka and Deodatis 1991, Simulation of Stochastic Processes with Spectral Representation.*


For a 2D-1V homogeneous stochastic field <img src="https://latex.codecogs.com/gif.latex?\inline&space;f_0&space;(x_1&space;,&space;x_2)" title="f_0 (x_1 , x_2)" /> case, the following series as <img src="https://latex.codecogs.com/gif.latex?\inline&space;N_1,N_2\rightarrow&space;\infty" title="N_1,N_2\rightarrow \infty" /> can be used for its simulation:

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\begin{matrix}&space;f(x_1,x_2)=\sqrt2\sum_{n_1=0}^{N_1-1}\sum_{n_2=0}^{N_2-1}[A_{n_1n_2}cos(\omega_{1n_1}x_1&space;&plus;&space;\omega_{2n_2}x_2&space;&plus;&space;\Phi_{n_1n_2}^{(1)})\\&space;&plus;\widetilde{A}_{n_1n_2}cos(\omega_{1n_1}x_1-\omega_{2n_2}&plus;\Phi_{n_1n_2}^2)]&space;\end{matrix}" title="\begin{matrix} f(x_1,x_2)=\sqrt2\sum_{n_1=0}^{N_1-1}\sum_{n_2=0}^{N_2-1}[A_{n_1n_2}cos(\omega_{1n_1}x_1 + \omega_{2n_2}x_2 + \Phi_{n_1n_2}^{(1)})\\ +\widetilde{A}_{n_1n_2}cos(\omega_{1n_1}x_1-\omega_{2n_2}+\Phi_{n_1n_2}^2)] \end{matrix}" />(13)

where:

<img src="https://latex.codecogs.com/gif.latex?\inline&space;A_{n_1n_2}=\sqrt{2S_{f_0&space;f_0}&space;(\omega_{1n_1},\omega_{2n_2})\Delta\omega_1\Delta\omega_2}" title="A_{n_1n_2}=\sqrt{2S_{f_0 f_0} (\omega_{1n_1},\omega_{2n_2})\Delta\omega_1\Delta\omega_2}" />(14)

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\widetilde{A}_{n_1n_2}=\sqrt{2S_{f_0f_0}(\omega_{1n_1},-\omega_{2n_2})\Delta\omega_1\Delta\omega_2}" title="\widetilde{A}_{n_1n_2}=\sqrt{2S_{f_0f_0}(\omega_{1n_1},-\omega_{2n_2})\Delta\omega_1\Delta\omega_2}" />(15)

For further reading: *Shinozuka and Deodatis 1996, Simulation of multi-dimensional Gaussian stochastic fields by spectral representation.*
