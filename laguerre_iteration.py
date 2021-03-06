import numpy as np
import sympy
import cmath
import matplotlib.pyplot as plt

def Lagg(f, x0, n, k, tol):
    '''
    :param f: function you want to find the roots with variable z (it can be used for complex & real functions)
    :param x_k: the iniitial guess
    :param n: order of polynomial ( e.g. n=1 for p(z) = z**2 -1 )
    :param k: up to what iteration are we finding x, x_k
    :return: x_k_1, i where "x_k_1" is the root and "i" is the number of iterations it took
    '''
    z = sympy.symbols('z')
    f_pr = sympy.lambdify(z, sympy.diff(f(z),z))
    mu = lambda z: f_pr(z)/f(z)
    mu_pr = sympy.lambdify(z, sympy.diff(mu(z),z))

    x_k = x0
    for i in range(k+1):
        s = 1
        x1 = np.abs(-f_pr(x_k)/f(x_k)+s*np.sqrt((n-1)*(-n*mu_pr(x_k) - mu(x_k)**2)))
        x2 = np.abs(-f_pr(x_k)/f(x_k)-s*np.sqrt((n-1)*(-n*mu_pr(x_k) - mu(x_k)**2)))
        if x2 > x1:
            s = -1
        if x2 < x1:
            s = 1

        x_k_1 = x_k + n / ( - mu(x_k) + s * cmath.sqrt( (n-1) * (- n * mu_pr(x_k) - mu(x_k)*mu(x_k) ) ) )

        if np.abs(x_k_1 - x_k) < tol:
            break

        x_k = x_k_1
        print(x_k,s)

    return x_k, i
