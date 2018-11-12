#Dado: y' = 2xy con y(1) = 1 
#Encontrar por el método de Euler el valor de y(1.5) utilizando 
#a) h = 0.1 
import math
import numpy as np
from sympy import *
import matplotlib.pyplot as plt


def f(x,y):
    return 2*x*y


def euler(y0,x,h,f):
    y = []
    y.append(y0)
    for i in range (1,len(x)):
        y.append(y[i-1] + h*f(x[i-1],y[i-1])) 
    return y


a = 1   #punto inicial
b = 1.5   #punto final saber el valor
h = 0.1
n = abs(a-b)/h + 1  #puntos

x = np.linspace(a,b,n)
y = euler(1,x,h,f)

plt.plot(x,y,'g')
plt.grid()
res = (y[-1])
print ("El resultado de y'=2xy en el punto y(1.5) con h=0.1 es: %f"%res)

#b) h = 0.05

import math
import numpy as np
from sympy import *
import matplotlib.pyplot as plt


def f(x,y):
    return 2*x*y


def euler(y0,x,h,f):
    y = []
    y.append(y0)
    for i in range (1,len(x)):
        y.append(y[i-1] + h*f(x[i-1],y[i-1])) 
    return y


a = 1   #punto inicial
b = 1.5   #punto final saber el valor
h = 0.05
n = abs(a-b)/h + 1  #puntos

x = np.linspace(a,b,n)
y = euler(1,x,h,f)

plt.plot(x,y,'g')
plt.grid()
res = (y[-1])

print ("El resultado de y'=2xy en el punto y(1.5) con h=0.05 es: %f"%res)

