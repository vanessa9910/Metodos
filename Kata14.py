#Los datos correspondientes  al censo de una población (en miles de habitantes) se recogen en la siguiente tabla:
#Año 1950 1960 1970 1980 1990 2000
#Número habitantes 123.5 131.2 150.7 141.3 203.2 240.5
#a) Utilizar interpolación polinómica para estimar el número de habitantes en el año 1965.

from sympy import *
import math

import matplotlib.pyplot as plt

import numpy as np

# 77,68,70,59,58,64,72,57,63]

# ,425,346,267,368,295,487,481,374,252]

x = [1950,1960,1970,1980,1990,2000]

y = [123.5,131.2,150.7,141.3,203.2,240.5]



pL = ''

for k in range(len(y)):

    pL += str(y[k]) +'*'

    Lxk = 1

    for i in range(len(x)):

        if (i == k):

            continue

        pL += '(x - %f)*'%(x[i])

        Lxk *= (x[k]-x[i])

    pL = pL[:-1]

    pL += '/%f+'%(Lxk)

pL = pL[:-1]



expr = sympify(pL)

#expr = expand(expr)

print(expand(expr))

plt.plot(x,y,'go')



x2 = np.linspace(1950,2000,100)

x = symbols('x')

y2 = [expr.subs(x,xi) for xi in x2]

plt.plot(x2,y2)

plt.grid()
re = (-2.80833333333333e-5*1965**5 + 0.277207916666667*1965**4 - 1094.4933*1965**3 + 2160633.56120833*1965**2 - 2132601323.83917*1965 + 841954803085.5)
res = math.ceil(re)
#print (re)
print("El número de habitantes en 1965 era aproximadamente de: %d "%res)

#b) Utilizar el método de Lagrange para estimar el número de habitantes en el año 1965.
