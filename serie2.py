#1
def createMatriz(m,n,v):
    C = []
    for i in range(m):
        C.append([])
        for j in range(n):
            C[i].append(v)
    return C

def getDimensiones(A):
    return(len(A), len(A[0]))

def sumMatrices(A,B):
    Am,An = getDimensiones(A)
    Bm,Bn = getDimensiones(B)
    if Am != Bm or An!= Bn:
        print("Las dimensiones son diferentes")
        return[]
    C = createMatriz(Am,An,0)
    for i in range(Am):
        for j in range(An):
            C[i][j] = A[i][j] + B[i][j]
    return C
def mulMatrices(A,B):
    Am,An = getDimensiones(A)
    Bm,Bn = getDimensiones(B)
    if An != Bm:
        print("Las matrices no son conformables")
        return []
    C = createMatriz(Am,Bn,0)
    for i in range(Am):
        for j in range(Bn):
            for k in range(An):
                C[i][j] += A[i][k] * B[k][j]
    return C
    
def getMenorMatriz(A,r,c):
    m,n = getDimensiones(A)
    C = createMatriz(m-1,n-1,0)
    for i in range (m):
        if i == r:
            continue
        for j in range(n):
            if j == c:
                continue
            Ci = i
            if i > r:
                   Ci = i-1
            Cj = j
            if j > c:
                Cj = j-1
            C[Ci][Cj] = A[i][j]

    return C

def detMatriz(A):
    m,n = getDimensiones(A)
    if m != n:
        print("La matriz no es cuadrada")
        return -1
    if m==1:
        return m
    if m==2:
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]
    det = 0
    for j in range(n):
        det+=(-1)**(j)*A[0][j]*detMatriz(getMenorMatriz(A,0,j))
    return det
def getMatrizAdyacente(A):
    m,n = getDimensiones(A)
    C = createMatriz(m,n,0)
    for i in range(m):
        for j in range(n):
            C[i][j] = (-1)**(i+j)*detMatriz(getMenorMatriz(A,i,j))
    return C
def getMatrizTranspuesta(A):
    m,n = getDimensiones(A)
    C = createMatriz(n,m,0)
    for i in range(m):
        for j in range(n):
            C[j][i] = A[i][j]
    return C

def getMatrizInversa(A):
    detA = detMatriz(A)
    if detA == 0:
        print("La matriz no tiene inversa")
        return 0
    At = getMatrizTranspuesta(A)
    adyAt = getMatrizAdyacente(At )
    m,n = getDimensiones(A)
    C = createMatriz(m,n,0)
    for i in range(m):
        for j in range(n):
            C[i][j] = (1/detA)*adyAt[i][j]
    return C

# h + Cma  + Cme = 24
#0h + 3Cma - 4me = 0
#-2h + Cma + 2me = 0
A = createMatriz(3,3,0)
A[0] = [1,1,1]
A[1] = [0,3,-4]
A[2] = [-2,1,2]

R = createMatriz(3,1,0)
R[0] = [24]
R[1] = [0]
R[2] = [0]

inversa= getMatrizInversa(A)
print(mulMatrices(inversa, R))

#h = 10
#Cma = 8
#Cme = 6

#2
# s + n  + l  = 20
# 2s + n + 0  = 27
# s + 0 + 3l = 19
def createMatriz(m,n,v):
    C = []
    for i in range(m):
        C.append([])
        for j in range(n):
            C[i].append(v)
    return C

U = createMatriz(3,3,0)
U[0] = [1,1,1]
U[1] = [2,1,0]
U[2] = [1,0,3]

L =  createMatriz(3,3,0)
L[0] = [1,0,0]
L[1] = [0,1,0]
L[2] = [0,0,1]

C = createMatriz(3,1,0)
C[0] = [20]
C[1] = [27]
C[2] = [19]

for i in range(3):
    if U[i][i] == 0:
        print("La Matriz no tiene LU")
        break
    for j in range(i+1,3):
        c = -1*U[j][i] / U[i][i]
        L[j][i] = -1*c
        for k in range(3):
            U[j][k] += c*U[i][k]

# LZ = C
Z = createMatriz(3,1,0)
for i in range(3):
    Z[i][0] = C[i][0]
    for j in range(3):
        if i == j:
            break
        Z[i][0] -= L[i][j] * Z[j][0]

# UB = Z
B = createMatriz(3,1,0)
for i in range(2,-1,-1):
    B[i][0] = Z[i][0]
    for j in range(2,-1,-1):
        if i == j:
            break
        B[i][0] -= U[i][j] * B[j][0]
    B[i][0] = B[i][0]/U[i][i]
print(B)
# Habitaciones sencillas: 10
# Habitaciones normal : 7
# Habitaciones de lujo: 3

#3

# P + 0 -3h  = 100
# -2P + L+ 0  = -200
#  P + L + h = 1100

def createMatriz(m,n,v):
    C = []
    for i in range(m):
        C.append([])
        for j in range(n):
            C[i].append(v)
    return C

MA = createMatriz(3,4,0)
MA[0] = [1,0,-3,100]
MA[1] = [-2,1,0,-200]
MA[2] = [1,1,1,1100]

MAm = 3
MAn = 4
for i in range(MAm):
    pivote = MA[i][i]
    if pivote == 0:
        for j in range(i+1, MAm):
            pivote = MA[j][i]
            if pivote != 0:
                T = MA[i]
                MA[i] = MA[j]
                MA[j] = T
                break
    for k in range(MAn):
        MA[i][k] = (1/pivote)*MA[i][k]
    for j in range(i+1, MAm):
        c = -1*MA[j][i]
        T = createMatriz(1,MAn,0)
        for k in range(MAn):
            T[0][k] = c*MA[i][k]
        for k in range(MAn):
            MA[j][k] += T[0][k]


B = createMatriz(3,1,0)
for i in range(MAm-1,-1,-1):
    B[i][0] = MA[i][MAn-1]
    for j in range(MAn-2,-1,-1):
        if i == j:
            break
        B[i][0] -= MA[i][j]*B[j][0]
        
print(B)

#Paco = 400
#Luis = 600
#Hugo = 100

#4 no se puede resolver

#5
# pm + p + c = 120
# pm + p - 2c = 0
# 3pm + 5p + 3c = 420

def createMatriz(m,n,v):
    C = []
    for i in range(m):
        C.append([])
        for j in range(n):
            C[i].append(v)
    return C

MA = createMatriz(3,4,0)
MA[0] = [1,1,1,120]
MA[1] = [1,1,-2,0]
MA[2] = [3,5,3,420]

MAm = 3
MAn = 4
for i in range(MAm):
    pivote = MA[i][i]
    if pivote == 0:
        for j in range(i+1, MAm):
            pivote = MA[j][i]
            if pivote != 0:
                T = MA[i]
                MA[i] = MA[j]
                MA[j] = T
                break
    for k in range(MAn):
        MA[i][k] = (1/pivote)*MA[i][k]
    for j in range(i+1, MAm):
        c = -1*MA[j][i]
        T = createMatriz(1,MAn,0)
        for k in range(MAn):
            T[0][k] = c*MA[i][k]
        for k in range(MAn):
            MA[j][k] += T[0][k]

B = createMatriz(3,1,0)
for i in range(MAm-1,-1,-1):
    B[i][0] = MA[i][MAn-1]
    for j in range(MAn-2,-1,-1):
        if i == j:
            break
        B[i][0] -= MA[i][j]*B[j][0]
        
print(B)
#50 pan de muerto
#30 panques
#40 cocoles

#6
def createMatriz(m,n,v):
    C = []
    for i in range(m):
        C.append([])
        for j in range(n):
            C[i].append(v)
    return C

def getDimensiones(A):
    return(len(A), len(A[0]))

def sumMatrices(A,B):
    Am,An = getDimensiones(A)
    Bm,Bn = getDimensiones(B)
    if Am != Bm or An!= Bn:
        print("Las dimensiones son diferentes")
        return[]
    C = createMatriz(Am,An,0)
    for i in range(Am):
        for j in range(An):
            C[i][j] = A[i][j] + B[i][j]
    return C
def mulMatrices(A,B):
    Am,An = getDimensiones(A)
    Bm,Bn = getDimensiones(B)
    if An != Bm:
        print("Las matrices no son conformables")
        return []
    C = createMatriz(Am,Bn,0)
    for i in range(Am):
        for j in range(Bn):
            for k in range(An):
                C[i][j] += A[i][k] * B[k][j]
    return C
    
def getMenorMatriz(A,r,c):
    m,n = getDimensiones(A)
    C = createMatriz(m-1,n-1,0)
    for i in range (m):
        if i == r:
            continue
        for j in range(n):
            if j == c:
                continue
            Ci = i
            if i > r:
                   Ci = i-1
            Cj = j
            if j > c:
                Cj = j-1
            C[Ci][Cj] = A[i][j]

    return C

def detMatriz(A):
    m,n = getDimensiones(A)
    if m != n:
        print("La matriz no es cuadrada")
        return -1
    if m==1:
        return m
    if m==2:
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]
    det = 0
    for j in range(n):
        det+=(-1)**(j)*A[0][j]*detMatriz(getMenorMatriz(A,0,j))
    return det
def getMatrizAdyacente(A):
    m,n = getDimensiones(A)
    C = createMatriz(m,n,0)
    for i in range(m):
        for j in range(n):
            C[i][j] = (-1)**(i+j)*detMatriz(getMenorMatriz(A,i,j))
    return C
def getMatrizTranspuesta(A):
    m,n = getDimensiones(A)
    C = createMatriz(n,m,0)
    for i in range(m):
        for j in range(n):
            C[j][i] = A[i][j]
    return C

def getMatrizInversa(A):
    detA = detMatriz(A)
    if detA == 0:
        print("La matriz no tiene inversa")
        return 0
    At = getMatrizTranspuesta(A)
    adyAt = getMatrizAdyacente(At )
    m,n = getDimensiones(A)
    C = createMatriz(m,n,0)
    for i in range(m):
        for j in range(n):
            C[i][j] = (1/detA)*adyAt[i][j]
    return C

# A + B  - C = 3
# A + 0 + C = 17
# 0 + 2B + C = 22
A = createMatriz(3,3,0)
A[0] = [1,1,-1]
A[1] = [1,0,1]
A[2] = [0,2,1]

R = createMatriz(3,1,0)
R[0] = [3]
R[1] = [17]
R[2] = [22]

inversa= getMatrizInversa(A)
print(mulMatrices(inversa, R))
#Antonio: 7 años
#Brenda:  6 años
#Cinthia: 10 años

#7
# l + m  + mi  = 20
# l -m  + 0   = 5
# -l + 0 + mi = 4
def createMatriz(m,n,v):
    C = []
    for i in range(m):
        C.append([])
        for j in range(n):
            C[i].append(v)
    return C

U = createMatriz(3,3,0)
U[0] = [1,1,1]
U[1] = [1,-1,0]
U[2] = [-1,0,1]

L =  createMatriz(3,3,0)
L[0] = [1,0,0]
L[1] = [0,1,0]
L[2] = [0,0,1]

C = createMatriz(3,1,0)
C[0] = [20]
C[1] = [5]
C[2] = [4]

for i in range(3):
    if U[i][i] == 0:
        print("La Matriz no tiene LU")
        break
    for j in range(i+1,3):
        c = -1*U[j][i] / U[i][i]
        L[j][i] = -1*c
        for k in range(3):
            U[j][k] += c*U[i][k]

# LZ = C
Z = createMatriz(3,1,0)
for i in range(3):
    Z[i][0] = C[i][0]
    for j in range(3):
        if i == j:
            break
        Z[i][0] -= L[i][j] * Z[j][0]


# UB = Z
B = createMatriz(3,1,0)
for i in range(2,-1,-1):
    B[i][0] = Z[i][0]
    for j in range(2,-1,-1):
        if i == j:
            break
        B[i][0] -= U[i][j] * B[j][0]
    B[i][0] = B[i][0]/U[i][i]
print(B)
#lunes: 7 cosméticos 
#martes: 2 cosméticos 
#miercoles: 11 cosméticos 

#8
#Gauss - Seidel
#R - 2C + 0 =0          C= x1
#-3R +2C + d =0         d= x2
#5R + 15C + 20d= 210    R= x3 

def getX1(x3):
    return (x3)/2
def getX2(x1,x3):
    return (210-5*x3-15*x1)/20
def getX3(x1,x2):
    return (-x2-2*x1)/-3
x1=0
x2=0
x3=0
E = 0.00000001
for i in range (100):
    x1i = getX1(x3)
    x2i = getX2(x1i,x3)
    x3i = getX3(x1i,x2i)
    Ex1 = abs(x1 - x1i)
    Ex2 = abs(x2 - x2i)
    Ex3 = abs(x3 - x3i)
    x1 = x1i
    x2 = x2i
    x3 = x3i
    if Ex1 < E and Ex2 < E and Ex3 < E:
        break
print ("La solucion es: ", x1,x2,x3)
#resistores: 4
#diodos: 8
#capacitores: 2

#9
#def createMatriz(m,n,v):
    C = []
    for i in range(m):
        C.append([])
        for j in range(n):
            C[i].append(v)
    return C

def getDimensiones(A):
    return (len(A),len(A[0]))

def sumMatrices(A,B):
    Am,An = getDimensiones(A)
    Bm,Bn = getDimensiones(B)
    if Am != Bm or An != Bn:
        print("Las dimensiones son diferentes")
        return []
    C = createMatriz(Am,An,0)
    for i in range(Am):
        for j in range(An):
            C[i][j] = A[i][j] + B[i][j]
    return C

def restaMatrices(A,B):
    Am,An = getDimensiones(A)
    Bm,Bn = getDimensiones(B)
    if Am != Bm or An != Bn:
        print("Las dimensiones son diferentes")
        return []
    C = createMatriz(Am,An,0)
    for i in range(Am):
        for j in range(An):
            C[i][j] = A[i][j] - B[i][j]
    return C

def mulMatrices(A,B):
    Am,An = getDimensiones(A)
    Bm,Bn = getDimensiones(B)
    if An != Bm:
        print("Las matrices no son conformables")
        return []
    C = createMatriz(Am,Bn,0)
    for i in range(Am):
        for j in range(Bn):
            for k in range(An):
                C[i][j] += A[i][k] * B[k][j]
    return C
def getMenorMatriz(A,r,c):
    m,n = getDimensiones(A)
    C = createMatriz(m-1,n-1,0)
    for i in range(m):
        if i == r:
            continue
        for j in range(n):
            if j == c:
                continue
            Ci = i
            if i > r:
                Ci = i - 1
            Cj = j
            if j > c:
                Cj = j -1
            C[Ci][Cj] = A[i][j]
    return C

def detMatriz(A):
    m,n = getDimensiones(A)
    if m != n:
        print("La matriz no es cuadrada")
        return -1
    if m == 1:
        return m
    if m == 2:
        return  A[0][0]*A[1][1] - A[0][1]*A[1][0]
    det = 0
    for j in range(n):
        det += (-1)**(j)*A[0][j]*detMatriz(getMenorMatriz(A,0,j))
    return det

def getMatrizAdyacente(A):
    m,n = getDimensiones(A)
    C = createMatriz(m,n,0)
    for i in range(m):
        for j in range(n):
            C[i][j] = (-1)**(i+j)*detMatriz(getMenorMatriz(A,i,j))
    return C

def getMatrizTranspuesta(A):
    m,n = getDimensiones(A)
    C = createMatriz(n,m,0)
    for i in range(m):
        for j in range(n):
            C[j][i] = A[i][j]
    return C

def getMatrizInversa(A):
    detA = detMatriz(A)
    if detA == 0:
        print("La matriz no tiene inversa")
        return 0
    At =  getMatrizTranspuesta(A)
    adyAt =  getMatrizAdyacente(At)
    m,n = getDimensiones(A)
    C = createMatriz(m,n,0)
    for i in range(m):
        for j in range(n):
            C[i][j] = (1/detA)*adyAt[i][j]
    return C

####

####
# Esta parte se debe cambiar y corresponde
# Al sistema a resolver
def u(x,y):
    return x*2 + y*2 -100

def v(x,y):
    return x*y -40

def dudx(x,y):
    return 2*x 

def dudy(x,y):
    return 2*y

def dvdx(x,y):
    return y

def dvdy(x,y):
    return x

###
#Esta es la matriz Jacobiana
# Corresponde a las derivadas parciales
A = [[dudx, dudy],[dvdx, dvdy]]
###
# Esta es una matriz con las funciones originales
D = [[u],[v]]

####
#Valores iniciales
C = [[10],[3.42]]
####
# Error deseado
E = 0.00000001

for i in range(50):
    Am,An = getDimensiones(A)
    ## Calcular D en el punto C
    Di =createMatriz(Am,1,0)
    for k in range(Am):
        for j in range(1):
            Di[k][j] = D[k][j](C[0][0],C[1][0])
    Ai = createMatriz(Am,An,0)
    for k in range(Am):
        for j in range(An):
            Ai[k][j] = A[k][j](C[0][0],C[1][0])
    invAi = getMatrizInversa(Ai)
    Bi = restaMatrices(C,mulMatrices(invAi,Di))
    Ce = restaMatrices(C,Bi)
    if abs(Ce[0][0]) < E and abs(Ce[1][0]) < E:
        C = Bi
        break
    C = Bi

print("Los valores son",C)

# EJERCICIO 10

## Funciones para matrices
def createMatriz(m,n,v):
    C = []
    for i in range(m):
        C.append([])
        for j in range(n):
            C[i].append(v)
    return C

def getDimensiones(A):
    return (len(A),len(A[0]))

def sumMatrices(A,B):
    Am,An = getDimensiones(A)
    Bm,Bn = getDimensiones(B)
    if Am != Bm or An != Bn:
        print("Las dimensiones son diferentes")
        return []
    C = createMatriz(Am,An,0)
    for i in range(Am):
        for j in range(An):
            C[i][j] = A[i][j] + B[i][j]
    return C

def restaMatrices(A,B):
    Am,An = getDimensiones(A)
    Bm,Bn = getDimensiones(B)
    if Am != Bm or An != Bn:
        print("Las dimensiones son diferentes")
        return []
    C = createMatriz(Am,An,0)
    for i in range(Am):
        for j in range(An):
            C[i][j] = A[i][j] - B[i][j]
    return C

def mulMatrices(A,B):
    Am,An = getDimensiones(A)
    Bm,Bn = getDimensiones(B)
    if An != Bm:
        print("Las matrices no son conformables")
        return []
    C = createMatriz(Am,Bn,0)
    for i in range(Am):
        for j in range(Bn):
            for k in range(An):
                C[i][j] += A[i][k] * B[k][j]
    return C
def getMenorMatriz(A,r,c):
    m,n = getDimensiones(A)
    C = createMatriz(m-1,n-1,0)
    for i in range(m):
        if i == r:
            continue
        for j in range(n):
            if j == c:
                continue
            Ci = i
            if i > r:
                Ci = i - 1
            Cj = j
            if j > c:
                Cj = j -1
            C[Ci][Cj] = A[i][j]
    return C

def detMatriz(A):
    m,n = getDimensiones(A)
    if m != n:
        print("La matriz no es cuadrada")
        return -1
    if m == 1:
        return m
    if m == 2:
        return  A[0][0]*A[1][1] - A[0][1]*A[1][0]
    det = 0
    for j in range(n):
        det += (-1)**(j)*A[0][j]*detMatriz(getMenorMatriz(A,0,j))
    return det

def getMatrizAdyacente(A):
    m,n = getDimensiones(A)
    C = createMatriz(m,n,0)
    for i in range(m):
        for j in range(n):
            C[i][j] = (-1)**(i+j)*detMatriz(getMenorMatriz(A,i,j))
    return C

def getMatrizTranspuesta(A):
    m,n = getDimensiones(A)
    C = createMatriz(n,m,0)
    for i in range(m):
        for j in range(n):
            C[j][i] = A[i][j]
    return C

def getMatrizInversa(A):
    detA = detMatriz(A)
    if detA == 0:
        print("La matriz no tiene inversa")
        return 0
    At =  getMatrizTranspuesta(A)
    adyAt =  getMatrizAdyacente(At)
    m,n = getDimensiones(A)
    C = createMatriz(m,n,0)
    for i in range(m):
        for j in range(n):
            C[i][j] = (1/detA)*adyAt[i][j]
    return C

####
# Sistema a resolver
# 4*x1 - x2 + x3 - x1*x4
# -x1 + 3*x2 - 2*x3 - x2*x4
# x1 - 2*x2 + 3*x3 - x3*x4
# 2*x1 + 2*x2 + 2*x3 - 1

####
# Esta parte se debe cambiar y corresponde
# Al sistema a resolver
def n(x1,x2,x3,x4):
    return 4*x1 - x2 + x3 - x1*x4

def dnd1(x1,x2,x3,x4):
    return 4-x4

def dnd2(x1,x2,x3,x4):
    return -1

def dnd3(x1,x2,x3,x4):
    return 1

def dnd4(x1,x2,x3,x4):
    return -x1

def m(x1,x2,x3,x4):
    return -x1 + 3*x2 - 2*x3 - x2*x4

def dmd1(x1,x2,x3,x4):
    return -1

def dmd2(x1,x2,x3,x4):
    return 3-x4

def dmd3(x1,x2,x3,x4):
    return -2

def dmd4(x1,x2,x3,x4):
    return -x2

def o(x1,x2,x3,x4):
    return x1 - 2*x2 + 3*x3 - x3*x4

def dod1(x1,x2,x3,x4):
    return 1

def dod2(x1,x2,x3,x4):
    return -2

def dod3(x1,x2,x3,x4):
    return 3-x4

def dod4(x1,x2,x3,x4):
    return -x3

def p(x1,x2,x3,x4):
    return 2*x1 + 2*x2 + 2*x3 - 1

def dpd1(x1,x2,x3,x4):
    return 2

def dpd2(x1,x2,x3,x4):
    return 2

def dpd3(x1,x2,x3,x4):
    return 2

def dpd4(x1,x2,x3,x4):
    return 0

###
#Esta es la matriz Jacobiana
# Corresponde a las derivadas parciales
A = [[dnd1, dnd2, dnd3, dnd4],[dmd1, dmd2, dmd3, dmd4], [dod1, dod2, dod3, dod4], [dpd1, dpd2, dpd3, dpd4]]
###
# Esta es una matriz con las funciones originales
D = [[n],[m], [o], [p]]

####
#Valores iniciales
C = [[1],[1],[1],[1]]
####
# Error deseado
itera=0

for i in range(100):
    itera = itera + 1
    Am,An = getDimensiones(A)
    ## Calcular D en el punto C
    Di =createMatriz(Am,1,0)
    for k in range(Am):
        for j in range(1):
            Di[k][j] = D[k][j](C[0][0],C[1][0],C[2][0],C[3][0])
    Ai = createMatriz(Am,An,0)
    for k in range(Am):
        for j in range(An):
            Ai[k][j] = A[k][j](C[0][0],C[1][0],C[2][0],C[3][0])
    invAi = getMatrizInversa(Ai)
    Bi = restaMatrices(C,mulMatrices(invAi,Di))
    Ce = restaMatrices(C,Bi)
    print ("Error en iteración ",itera," = ",Ce)    
    if itera==4:
        C = Bi
        break
    C = Bi

print("Los valores son",C)
