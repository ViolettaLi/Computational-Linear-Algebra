from numpy import linalg
import numpy as np
from numpy import random
from cla_utils.exercises6 import LU_inplace_new
from cla_utils.exercises6 import solve_L
from cla_utils.exercises6 import solve_U


# 3c
# Specific Algorithm to find out the eigenvalue of a 2*2 matrix A_1
def soleiv1(A):
    a = 1
    b = -2
    c = 1
    lam1 = (-1 * b + np.sqrt(b**2 - 4*a * c))/2
    lam2 = (-1 * b - np.sqrt(b**2 - 4*a*c))/2
    return lam1, lam2


#Specific Algorithm to find out the eigenvalue of a 2*2 matrix A_2
def soleiv2(A):
    a = 1
    b = -(2 + 1e-14)
    c = 1+1e-14
    lam21 = (-1 * b + np.sqrt(b**2 - 4*a * c))/2
    lam22 = (-1 * b - np.sqrt(b**2 - 4*a*c))/2
    return lam21, lam22

#Use the above function to find out the results(eigencalue) of A1 and A2 in the floating point implementation
A1 = np.array([[1, 0], [0, 1]])
A2 = np.array([[1+10**(-14), 0], [0, 1]])
lamA11, lamA12 = soleiv1(A1)
lamA21, lamA22 = soleiv2(A2)


# General Algorithm to find out the eigenvalue of a 2*2 matrix A
def soleiv(A):
    #Algorithm to find out the eigenvalue of a 2*2 matrix A
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]  
    #compute eigenvalues by characteristic polynomial
    char = b * c - a * d / 2 + (d**2 + a**2)/4
    lam1 = (np.sqrt(char) if char >= 0 else 1j * np.sqrt(abs(char))) + (d + a)/2
    lam2 = (-np.sqrt(char) if char >= 0 else  -1j * np.sqrt(abs(char))) + (d + a)/2 
    return lam1, lam2
    #A1 = np.array([[1, 0], [0, 1]])
    #A2 = np.array([[1+10**(-14), 0], [0, 1]])
    #Use the above function to find out the results(eigencalue) of A1 and A2 in the floating point implementation
    #lamA11, lamA12 = soleiv(A1)
    #lamA21, lamA22 = soleiv(A2)



#4a
def creatA(n, epsilon):
    # create B matrix to store matrix B
    B = np.zeros((4 * n + 1, 4 * n + 1))
    I = np.identity(4 * n + 1)
    for i in range(n):
        #create the random 5*5 matrix vijk
        vijk = random.randn(5, 5)
        #Put the value to the diagonal of matrix by storing it in Bi
        Bi = np.zeros((4 * n + 1, 4 * n + 1))
        Bi[4 * i:4 * i + 5, 4 * i:4 * i + 5] = vijk
        B = B + Bi        
    # Because epsilon in range(0, 0.1), I  can assume that epsilon = 0.05
    A = I + epsilon * B
    return A


def LU_check(A):
    m, k = A.shape
    #Find out the L and U by one-loop LU_inplace in exercise6
    new_A = LU_inplace_new(A)
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = new_A[i1]
    #L = np.identity(m) + np.tril(new_A, -1)
    U = np.triu(new_A, -1)
    L = np.identity(m) + np.tril(A, -1)
    U = np.triu(A, -1)
    return L, U

#See the observation by computing some A, L and U.
A = creatA(3, 0.05)
L, U = LU_check(A)

# 4c
def CWLU_inplace(A):
    m, n = A.shape
    for k in range(m - 1):
        #num = min(m, k + 5 - k % 4)
        num = k + 5 - k % 4
        A[k + 1:num, k] = A[k + 1:num, k] / A[k ,k]
        A[k + 1:num, k + 1:num] = A[k + 1:num, k + 1:num] - np.outer(A[k + 1:num, k], A[k, k + 1:num])


# 4e
def solve_bandtri(A, b):
    #step 1: do elimination
    #A and b need to do the row elimination together so I combine them together
    combineAb = np.c_[A, b]
    m, m=A.shape
    n = int((m - 1)/4)
    for k in range(n):
        p = 4 * k
       # do row elimination in middle 3 columns in each block of 5*5 matrix
        for j in range(p + 1, p + 4):
            #do row elimination in each entries excpet when i + j = 6
            for i in [x for x in range(p, p + 5) if x != j]:
                if i % 4 == 0:
                    begin = i - 4
                    end = i + 5
                else:
                    begin = p
                    end = p + 5
                index = np.append(np.arange(max(begin, 0), min(end, m)), m)
                combineAb[i, index] = combineAb[i, index]/(combineAb[i, j]/combineAb[j, j]) - combineAb[j, index]
    #step2: reduce dimension and calculate x_1, x_5, x_9, x_13, etc by tildeA and tildeb such that tildeA*x(some entries) = tildeb.
    tildeA = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        a = min(i + 1, n)
        tildeA[i, i] = combineAb[4*i, 4*i]
        tildeA[i, a] = combineAb[4*i, 4*a]
        tildeA[a, i] = combineAb[4*a, 4*i]
    ran = np.arange(0, m, 4)
    tildeb = combineAb[:, m][ran]
    for i in range(n):
        rate = tildeA[i + 1,i]/tildeA[i, i]
        tildeb[i + 1] = tildeb[i + 1] - rate * tildeb[i]
        tildeA[i + 1, i + 1] = tildeA[i + 1, i + 1] - rate * tildeA[i, i + 1]
    tildex = tildeb / tildeA[n, n]
    for i in range(n, -1, -1):
        tildex[i - 1] = (tildeb[i - 1] - tildeA[i - 1, i] * tildex[i]) / tildeA[i - 1, i - 1]
    xn = b.shape
    x = np.zeros(xn)
    x[ran] = tildex
    #step3: calculate the other entries of x
    for i in range(n):
        num = 4 * i + 1
        diag = np.diag(combineAb[num:num + 3, num:num + 3])
        d = combineAb[:, m][num: num + 3] - np.dot(combineAb[num: num + 3, num - 1], x[num - 1]) - np.dot(combineAb[num:num + 3, num + 3], x[num + 3])
        x[num:num + 3] = d/diag 
    return x        


# 5b
def CWLU_inplace5(A):
    m, m = A.shape
    n = int(np.sqrt(m)) + 1
    for k in range((n - 1)**2 - 1):
        num = k + n - 1
        A[k + 1:num + 1, k] = A[k + 1:num + 1, k] / A[k ,k]
        A[k + 1:num + 1, k + 1:num + 1] = A[k + 1:num + 1, k + 1:num + 1] - np.outer(A[k + 1:num + 1, k], A[k, k + 1:num + 1])


# 5d
def solve(n, alpha, s0, r0, mu, c, u_k):
    m = (n - 1)**2
    delta = 1/n
    x = np.array([(i + 1) * delta for i in range(n -1)])
    y = np.array([(i + 1) * delta for i in range(n -1)])
    B1 = - alpha * np.outer(np.sin(x * np.pi), np.cos(y * np.pi))
    B2 = alpha * np.outer(np.cos(x * np.pi), np.sin(y * np.pi))
    S = s0 * np.exp(-(x - 1/4)**2/r0**2 - (y - 1/4)**2/r0**2)
    ap = B1/(2*delta) - mu/delta**2
    fp = -(B1/(2*delta) + mu/delta**2)
    e = 4 * mu/(delta**2) + c
    cp = - B2/(2*delta) + mu/delta**2
    dp = B2/(2*delta) + mu/delta**2
    # First matrix A1
    A1 = e * np.indentity(m)
    A1[n - 1:, :m - n + 1] = A1[n - 1:, :m - n + 1] + np.diag(fp.ravel()[n - 1:])
    A1[:m - n + 1, n - 1:] = A1[:m - n + 1, n - 1:] + np.diag(ap.ravel()[:m - n + 1])
    #Find C1
    C1 = np.indentity(m)
    for i in range(m):
        if i % (n-1) != n-2:
            C1[i][i+1] = cp.ravel()[i]
            C1[i+1][i] = dp.ravel()[i+1]
    Vk = np.reshape(u_k, -1)
    b1 = S + C1 * Vk
    b0 = 1 * b1
    # Use LU decomposition to solve Ax = b
    A1 = LU_inplace_new(A1)
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = A1[i1]
    U = np.triu(A1)
    y = solve_L(L, b0)
    x = solve_U(U, y)
    vnew = x.reshape(m,)
    A2 = e * np.identity(m)
    for i in range(m):
        if i % (n - 1) != n - 2:
            A2[i][i + 1] = -cp.ravel()[i]
            A2[i+1][i] = -dp.ravel()[i + 1]
    C2 = np.identity(m)
    C2[n-1:, :m - n + 1] = C2[n-1:, :m - n + 1] + np.diag(fp.ravel()[n - 1:])
    C2[:m - n + 1, n-1:] = C2[:m - n + 1, n-1:] + np.diag(ap.ravel()[:m - n + 1])
    b2 = S - np.dot(C2, vnew)
    b2_new = 1 * b2
    # Use LU decomposition to solve Ax = b
    A2 = LU_inplace_new(A2)
    L2 = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L2[i1] = A2[i1]
    U2 = np.triu(A2)
    y2 = solve_L(L2, b2_new)
    x2 = solve_U(U2, y2)
    v_now = x2.reshape(m,)
    u_half = vnew.reshape(n-1, n-1)
    u_whole = v_now.reshape(n-1, n-1)
    return u_half, u_whole




    
