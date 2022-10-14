#from re import A
import numpy as np
import numpy.random as random
from numpy import linalg
from cla_utils.exercises3 import householder_qr
from cla_utils.exercises9 import pure_QR
from cla_utils.exercises8 import hessenberg
from cla_utils.exercises2 import GS_modified
from cw2.CW2 import solve_bandtri

#2(a)
# To create A
def create_A(n):
    A = np.zeros((2 * n, 2 * n), dtype = complex)
    for i in range(2 * n - 1):
        # The struncture by question
        A[i, i + 1] = 1
        A[i + 1, i] = -1
    return A


# To use pure QR factorisation I will import the function householder_qr to find out the Ak after applying pure_QR
def pure_QR_A(n, maxit, tol):
    A = np.zeros((2 * n, 2 * n), dtype = complex)
    for i in range(2 * n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = -1
    A0 = 1.0 * A
    """
    for k in range(1, maxit + 1):
        r = np.linalg.norm(A0[np.tril_indices(2 * n, -1)])/(2 * n)**2
        if r  > tol:
            Q0, R0 = householder_qr(A0)
            A0 = np.dot(R0, Q0)
            r = np.linalg.norm(A0[np.tril_indices(2 * n, -1)])/(2 * n)**2"""
    A0 = pure_QR(A0, maxit, tol)
    return A0


# Example when n = 2
"""
Ak_2a = np.round((pure_QR_A(2,10000,1e-5)),2)
"""


#2(c)
def pure_QR_A_ev(n, maxit, tol):
    A0 = create_A(n)
    # By 2(a) I can get Ak by function pure_QR_A
    A0 = pure_QR_A(n, maxit, tol)
    # Find out the eigen values
    eigv1 = np.diag(A0, -1) * 1j
    eigv1 = np.append(eigv1, 0)
    eigv2 = np.diag(A0, 1) * 1j
    # Make 2 arrays together to get an array contains all eigen values
    for i in range(n):
        eigv1[i * 2 + 1] = eigv2[i * 2]
    return eigv1


# Numerical experiments for different values of n
"""
# For n = 2
e21 = np.round(pure_QR_A_ev(2,10000,1e-5), 2)
e22 = np.round(np.linalg.eig(create_A(2))[0],2)
# For n = 3
e31 = np.round(pure_QR_A_ev(3,10000,1e-5), 2)
e32 = np.round(np.linalg.eig(create_A(3))[0],2)
# For n = 4
e41 = np.round(pure_QR_A_ev(4,10000,1e-5), 2)
e42 = np.round(np.linalg.eig(create_A(4))[0],2)
"""


#2(d)
def pure_QR_A_ev_new(n, maxit, tol):
    A = np.zeros((2 * n, 2 * n), dtype = complex)
    # To create A
    for i in range(2 * n - 1):
        A[i, i + 1] = 2
        A[i + 1, i] = -1
    A0 = 1.0 * A
    # Calculate Ak by pure_QR
    for k in range(1, maxit + 1):
        r = np.linalg.norm(A0[np.tril_indices(2 * n, -1)])/(2 * n)**2
        if r  > tol:
            Q0, R0 = householder_qr(A0)
            A0 = np.dot(R0, Q0)
            r = np.linalg.norm(A0[np.tril_indices(2 * n, -1)])/(2 * n)**2
    # Put two tri-diagonals into arraies
    eigv1 = - np.diag(A0, -1)
    eigv2 = - np.diag(A0, 1)
    # Create an empty array to contain eigenvalues
    eigv = np.zeros(2 * n, dtype = A0.dtype)
    # Use for loop to put all eigenvalues into the array
    for i in range(n):
        root_square = eigv1[2 * i] * eigv2[2 * i]
        eigv[2 * i] = np.sqrt(root_square)
        eigv[2 * i + 1] = - np.sqrt(root_square)
    return eigv


def create_A_d(n):
    A = np.zeros((2 * n, 2 * n), dtype = complex)
    for i in range(2 * n - 1):
        A[i, i + 1] = 2
        A[i + 1, i] = -1
    return A


# Numerical experiments for different values of n
"""
# For n = 2
e21n = np.round(pure_QR_A_ev_new(2,10000,1e-5), 9)
e22n = np.round(np.linalg.eig(create_A_d(2))[0],9)
# For n = 3
e31n = np.round(pure_QR_A_ev_new(3,10000,1e-5), 9)
e32 = np.round(np.linalg.eig(create_A_d(3))[0],9)
# For n = 4
e41n = np.round(pure_QR_A_ev_new(4,10000,1e-5), 9)
e42n = np.round(np.linalg.eig(create_A_d(4))[0],9)
"""


# 3(b)
# create the 5 × 5 matrix Aij = 1/(i + j + 1)
def create_Aij(m):
    A = np.zeros((5, 5), dtype = complex)
    for i in range(m):
        for j in range(m):
            # satisfy the condition given in question
            A[i, j] = 1/(i + j + 1)
    return A


#  Apply my program to the 5 × 5 matrix Aij = 1/(i + j + 1)
"""
A_3b = create_Aij(5)
A_3b_H = hessenberg(A_3b)
e1 = pure_QR_1(A_3b_H, 10000, 1e-5, sym = True)
# The real eigenvalues of matrix A_3b
eigv3b = np.linalg.eig(A_3b)[0]
# The estimated eigenvalues of matrix A by choose the diagonal from the matrix after pure_QR
eigv3bd = np.diag(e1)
print(eigv3b - eigv3bd)
"""


# 3(c)
"""
A = create_Aij(m)
"""
def ev_3c_check(m, maxit, tol, sym1, return_t1, shift1):
    A = create_Aij(m)
    # turns A into the hessenberg form
    T = hessenberg(A)
    # create an empty array to contain the eigenvalues which will be calculated in loops
    eigv_array = np.array([])
    # create an empty array to contain the t values which will be calculated in loops
    t_array = np.array([])
    for k in range(m, 0, -1):
        # because pure_QR has 2 possible output I need to use if else function to write down matrix after applying pure_QR and array of t.
        if return_t1:
            T, t, i = pure_QR(T, maxit, tol, sym = sym1, return_t = return_t1, shift = shift1)
            # Put the t values into the array t_array
            if k > 1:
                t_array = np.append(t_array, t)
        # when return_t1 = False, the pure_QR only has 1 output, A0(T here)
        else:
            T = pure_QR(T, maxit, tol, sym = sym1, return_t = return_t1)
        # Compute eigenvalue by taking T_(k, k) in loops
        eigv = T[k - 1, k - 1]
        # Put the eigenvalues into the array eigv_array
        eigv_array = np.append(eigv_array, eigv)
        # Compute new T with smaller size for next loop
        T = T[:k - 1, :k - 1]
    return eigv_array, t_array


# Plot the graph in 3c
t_array = ev_3c_check(5, 10000, 1e-12, True, True, True)[1]
import matplotlib.pyplot as plt
plt.plot(t_array)
plt.title('t values')
plt.xlabel('order of the value in the t_array')
plt.ylabel('T_{m, m-1}')
plt.show()
plt.show()



# 3(d)
# plot the graph in 3(d)
t_array = ev_3c_check(5, 10000, 1e-12, True, True, True)[1]
import matplotlib.pyplot as plt
plt.plot(t_array)
plt.title('t values')
plt.xlabel('order of the value in the t_array')
plt.ylabel('T_{m, m-1}')
plt.show()
plt.show()


# 3(e)
# define the function which can return the t_array
def ev_3e_check(m, maxit, tol, sym1, return_t1, shift1):
    # Create A = D + O
    D = np.diag(np.linspace(m,1,m))
    O = np.ones((m, m), dtype = D.dtype)
    A = D + O
    # turns A into the hessenberg form
    T = hessenberg(A)
    # create an empty array to contain the eigenvalues which will be calculated in loops
    eigv_array = np.array([])
    # create an empty array to contain the t values which will be calculated in loops
    t_array = np.array([])
    for k in range(m, 0, -1):
        # because pure_QR has 2 possible output I need to use if else function to write down matrix after applying pure_QR and array of t.
        if return_t1:
            T, t, i = pure_QR(T, maxit, tol, sym = sym1, return_t = return_t1, shift = shift1)
            # Put the t values into the array t_array
            if k > 1:
                t_array = np.append(t_array, t)
        # when return_t1 = False, the pure_QR only has 1 output, A0(T here)
        else:
            T = pure_QR(T, maxit, tol, sym = sym1, return_t = return_t1)
        # Compute eigenvalue by taking T_(k, k) in loops
        eigv = T[k - 1, k - 1]
        # Put the eigenvalues into the array eigv_array
        eigv_array = np.append(eigv_array, eigv)
        T = T[:k - 1, :k - 1]
    # Compute new T with smaller size for next loop
    return eigv_array, t_array


# graph corresponding to no shift in 3(e)
t_array_noshifts = ev_3e_check(15, 10000, 1e-12, True, True, False)[1]
import matplotlib.pyplot as plt
plt.plot(t_array_noshifts)
plt.title('when A = D + O, t values corresponding to no shift')
plt.xlabel('order of the value in the t_array')
plt.ylabel('T_{m, m-1}')
plt.show()


# graph corresponding to shifts in 3(e)
t_array_shifts = ev_3e_check(15, 10000, 1e-12, True, True, True)[1]
import matplotlib.pyplot as plt
plt.plot(t_array_shifts)
plt.title('when A = D + O, t values corresponding to shifts')
plt.xlabel('order of the value in the t_array')
plt.ylabel('T_{m, m-1}')
plt.ylim(-0.01, 0.28)
plt.show()


# plot log(t) to no shift in 3(e)
plt.plot(np.log(t_array_noshifts))
plt.title('when A = D + O, log(t values) corresponding to no shift')
plt.xlabel('order of the value in the t_array')
plt.ylabel('log(T_{m, m-1})')
plt.show()
plt.show()


# plot log(t) to shifts in 3(e)
plt.plot(np.log(t_array_shifts))
plt.title('when A = D + O, log(t values) corresponding to shifts')
plt.xlabel('order of the value in the t_array')
plt.ylabel('log(T_{m, m-1})')
plt.show()
plt.show()


# 4(a)
# serialised normal two-dimensional matrix u row by row
"""
# delete the boundary
def u_to_v_nb(u):
    n, n = u.shape
    # because the coundary of u, i.e. u0,0; u0,1; u1,n are all zero, I need to delete the boundary of u
    u = u[1:n - 1, 1:n- 1]
    # flatten is a function in numpy which can 
    v = u.flatten()
    return v
"""
# not delete the boundary
def u_to_v_wb(u):
    n, n = u.shape
    # because the coundary of u, i.e. u0,0; u0,1; u1,n are all zero, I need to delete the boundary of u
    # flatten is a function in numpy which can 
    v = u.flatten()
    return v


# serialised 2d matrix dx with size (n+1)*n (different dimensions) into 1d array row by row
def dx_to_v(dx):
    """
    m = len(dx[:,0])
    v = np.array([])
    for i in range(m):
        dxi =dx[i]
        v = np.append(v, dxi)
    """
    v = dx.flatten()
    return v


# serialised 2d matrix dy with size n*(n+1) (different dimensions) into 1d array row by row
def dy_to_v(dy):
    """
    m = len(dy[:,0])
    v = np.array([])
    for i in range(m):
        dxi =dy[i]
        v = np.append(v, dxi)
    """
    v = dy.flatten()
    return v

"""
# Inverse from 1d array into the normal two-dimensional array of u
# add the boundary 0
def v_to_u_nb(v):
    # The original form is a normal two-dimensional matrix, the size will be 2*m and calculate m here
    n = int(np.sqrt(v.shape))
    u1 = v.reshape(n,n)
    u = np.zeros((n+2, n+2), dtype=v.dtype)
    u[1:n+1, 1:n+1] = u1
    return u
"""

# Inverse from 1d array into the normal two-dimensional array of u
# not add the boundary
def v_to_u_wb(v):
    m=len(v)
    n = int(np.sqrt(m))
    u = np.zeros((n, n), dtype = v.dtype)
    for i in range(n):
        uk = v[n * i:n * (i + 1)]
        u = np.vstack((u, uk))
    return u

# Inverse from 1d array into the 2d matrix with different sizes (different dimensions)
def v_to_d(v, x):
    """
    input
    v: the 1d array
    x: change into dx or dy if change to dx x = True, if change into dy x =False
    """
    # m is the dimension of the matrix that we need, m must be the division of len(v)
    # The original form is a normal m dimensional matrix, the size will be m*n and calculate n here
    m = int(np.sqrt(len(v)))
    n = m + 1
    if x:
        u = np.zeros(m, dtype = int)
        for i in range(n):
            vi = v[m * i:m * (i + 1)]
            u = np.vstack((u, vi))
            print(i)
            print(u)
            print(vi)
    else:
        u = np.zeros(n, dtype = int)
        for i in range(m):
            vi = v[n * i:n * (i + 1)]
            u = np.vstack((u, vi))
            print(i)
            print(u)
            print(vi)
    u = u[1:]
    return u


# 4(b)
def H_apply(v, lambda0, mu0):
    u = v_to_u_wb(v)
    m, n = u.shape
    # create the row vector and column vector to combine with ui-1,j, ui+1,j, ui,j+1, ui,j-1
    u0r = np.zeros(n + 2, dtype = u.dtype)
    u0c = np.array([np.zeros(m + 2, dtype = u.dtype)]).T
    # u1i,j = ui,j, and add the boundary entries like u0,0
    u1 = u.copy()
    u1 = np.hstack((np.array([np.zeros(m)]).T, u1))
    u1 = np.hstack((u1, np.array([np.zeros(m)]).T))
    u1 = np.vstack((np.zeros(n + 2), u1))
    u1 =np.vstack((u1,np.zeros(n + 2)))
    # add in row at the beginning because when i = 1, u2i,j = ui-1,j = u0,j = 0
    u2 = np.vstack((u0r, u1[:m + 1, :]))
    # add in row at the end because when i = n, u3i,j = ui+1,j = un+1,j = 0
    u3 = np.vstack((u1[1:, :], u0r))
    # add in coulum at the beginning because when j = 1, u4i,j = ui,j-1 = ui,0 = 0
    u4 = np.hstack((u0c, u1[:, :n + 1]))
    # add in coulum at the end because when j = n, u4i,j = ui,j+1 = ui,n+1 = 0
    u5 = np.hstack((u1[:, 1:], u0c))
    L = 4 * u1 - u2 - u3 - u4 - u5
    L = lambda0 * L + mu0 * u1
    l = u_to_v_wb(L)
    return l


"""
def H_apply_wn(v, lambda0, mu0, laplacian = False):
    u = v_to_u_wb(v)
    m, n = u.shape
    u0 = u.copy()
    u1 = u[:, 1:]
    u2 = u[:, :-1]
    u3 = u[1:, :]
    u4 = u[:-1, :]
    L = np.zeros((m, n), dtype = u.dtype)
    L = L + 4 * u0
    L[:, :-1] = L[:, :-1] - u1
    L[:, 1:] = L[:, 1:] - u2
    L[:-1, :] = L[:-1, :] - u3
    L[1:, :] = L[1:, :] - u4
    l = lambda0 * L + mu0 * u0
    if laplacian:
        return u_to_v_wb(l), L
    else:
        return u_to_v_wb(l)
"""


