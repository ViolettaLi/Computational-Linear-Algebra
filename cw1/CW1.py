from re import S
import numpy as np
from cla_utils.exercises2 import orthog_cpts, solveQ, orthog_proj, orthog_space, GS_classical, GS_modified, GS_modified_get_R, GS_modified_R
from cla_utils.exercises3 import householder, householder_ls, householder_qr, householder_solve
from scipy.linalg import solve_triangular

# 2(a)
C = np.loadtxt('C.dat', delimiter=',')
Q, R = householder_qr(C)
#default precision=8
"""
We need to use the following code to find the property of R
"""
np.set_printoptions(suppress=True)



# 2(b)
Q_basis = Q[:, :3]
R_coefficient = R[:3, :]


def compress(C):
    Q, R = householder_qr(C)
    Q_basis = Q[:, :3]
    R_coefficient = R[:3, :]
    """C_new are used for test"""
    C_new = Q_basis.dot(R_coefficient)
    return Q_basis, R_coefficient, C_new

"""If |C_test-C| < 1.0e-8, this storage method is ok."""
#assert (np.linalg.norm(C_test-C) < 1.0e-8)

# 3(a)
l = np.arange(0., 1.0001, 1./51)
f = np.zeros((52,1))
f[0] = 1
f[50] = 1
A = np.zeros((52,13), dtype = float)
for i in range(52):
    for j in range(13):
        A[i, j] = l[i] ** j

A_1 = 1.0 * A
A_2 = 1.0 * A
A_3 = 1.0 * A

# Classical  Gram-Schmidt Method
def GS_classicalnew(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    Q = A
    Q_conjugate = np.conjugate(Q)
    m = len(A[0])
    R = np.zeros((m, m), dtype = A.dtype)
    for j in range(1,m+1):
        k = j-1
        R[:k, j-1] = np.einsum('ij,i->j', Q_conjugate, A[:, j-1])[:k]
        v_j = A[:, j-1] - np.inner(R[:, j-1], Q)
        R[j-1, j-1] = np.sqrt((np.conjugate(v_j).dot((v_j))))
        Q[:, j-1] = v_j / R[j-1, j-1]
    return Q, R
"""
Once Q and R are worked out, let's say that the coeffients I want is x, 
then QRx=f, Rx=Q*f and a solve_triangular can do the work.
"""
Q_1, R_1 = GS_classicalnew(A_1)
Q_1_star = np.conjugate(Q_1.T)
x_1 = solve_triangular(R_1, Q_1_star.dot(f))
print(x_1)

# Modified  Gram-Schmidt Method
def GS_modifiednew(A):
    n = len(A[0])
    m = len(A[:, 0])
    R = np.zeros((n, n), dtype = A.dtype)
    V = np.zeros((m, n), dtype = A.dtype)
    Q = A
    for i in range(n):
        V[:, i] = A[:, i]

    for i in range(n):
        R[i, i] =np.sqrt(V[:, i].dot(V[:, i]))
        Q[:, i] = V[:, i] / R[i, i]
        for j in range(i+1, n):
            R[i, j] = np.dot(Q[:, i].T, V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]

    return Q, R
Q_2, R_2 = GS_modifiednew(A_2)
"""
Once Q and R are worked out, let's say that the coeffients I want is x, 
then QRx=f, Rx=Q*f and a solve_triangular can do the work.
"""
Q_2_star = np.conjugate(Q_2).T
x_2 = solve_triangular(R_2, Q_2_star.dot(f))
print(x_2)
"""
R_2 = GS_modified(A)
m, n = A.shape
Q_2 = A
R = np.zeros((n, n), dtype = A.dtype)
V = A.copy()
for i in range(n):
    R[i, i] =np.linalg.norm(V[:, i])
    Q_2[:, i] = V[:, i] / R[i, i]
    R[i, (i + 1):] = np.dot(np.conjugate(Q_2[:, i]).T, V[:, (i + 1):])
    V[:, (i + 1):] = V[:, (i + 1):] - np.outer(Q_2[:, i], R[i, (i + 1):])
Q_2_star = np.conjugate(Q_2).T
x_2 = solve_triangular(R_2, np.inner(Q_2_star, f.T))
print(x_2)
"""

#  Householder method: use householder_ls(A) directly
x_3 = householder_ls(A_3, f)
print(x_3)

#4(c)
def Rv_array(A):
    m, n = A.shape
    R = A.copy()
    V_new = np.zeros((m + 1, n), dtype = A.dtype)
    V_new[:m, :] = R
    for k in range(n):
        #x = 1.0 * R[k : m, k]
        x = V_new[k : m, k]
        if x[0] != 0:
            f = np.sign(x[0])
        else:
            f = 1
        I = np.eye(1, (m-k), 0)
        v_k = x + f * np.sqrt(np.conjugate(x).dot(x)) * I
        v_k = v_k / np.linalg.norm(v_k)
        #R[k : m, k : n] = R[k : m, k : n] - 2 * np.outer(v_k, np.matmul(v_k, A[k : m, k : n]))
        #R[(k+1):, k] = v_k
        V_new[k : m, k : n] = V_new[k : m, k : n] - 2 * np.outer(v_k, v_k.dot(V_new[k : m, k : n]))
        V_new[(k+1):, k] = v_k
        
    R_v = V_new
    return R_v

# 4(d)

def Q_starb(R_v, b):
    m, n = R_v.shape
    Q_starb_value = b.copy()
    for k in range(n):
        v_k = R_v[(k + 1):, k].copy()
        Q_starb_value[k:] = Q_starb_value[k:] - 2*v_k.dot(v_k.dot((Q_starb_value[k:])))
    return Q_starb_value

# 4(e)

def R_v_ls(R_v, b):
    R = np.triu(R_v)
    m, n = R_v.shape
    R_hat = R[:n, :]
    Q_starb_value = Q_starb(R_v, b)
    Q_hat_starb_value = Q_starb_value[:n]
    x = solve_triangular(R_hat, Q_hat_starb_value)
    return x


# 5(c)
def solve_x(A, b, lam):
    #lam is lambda
    n, m = A.shape
    I_n = np.identity(n)
    Q, R = householder_qr(A)
    LRRT_inv = np.linalg.inv(lam * I_n + R.dot(R.T))
    x = np.inner(np.inner(R.T, LRRT_inv), Q.T).dot(b)
    return x

# 5(d)
def com_lam(A, b, lam, er):
    x = solve_x(A, b, lam)
    x_norm = np.linalg.norm(x)
    error = abs(x_norm - 1)
    while error > er:
        lam = lam * x_norm
        x = solve_x(A, b, lam)
        x_norm = np.linalg.norm(x)
        error = abs(x_norm - 1)
    return lam

# 5(e)

