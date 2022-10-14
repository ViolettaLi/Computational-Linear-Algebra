'''Tests for the cw1.'''
import pytest
import cw1
from numpy import random
import numpy as np
from cla_utils.exercises3 import householder, householder_qr, householder_ls
from scipy.linalg import solve_triangular



# :test compress(C)
def test_compress():
    C = np.loadtxt('C.dat', delimiter=',')
    Q_basis, R_coefficient, C_new = cw1.compress(C)
    assert (np.linalg.norm(C_new-C) < 1.0e-8)

# 4(c): test Rv_array(A)
@pytest.mark.parametrize('m, n', [(20, 17), (40, 3), (20, 12)])
def test_Rv_array(m, n):
    random.seed(1878*m + 1950*n)
    A = random.randn(m, n)
    m, n = A.shape
    R_v = cw1.Rv_array(A)
    R = np.triu(R_v)
    V = np.tril(R_v)
    assert(np.linalg.norm(R[:m, :] - householder(A) < 1.0e-6))
    A_1 = 1.0 * A
    A_2 = 1.0 * A
    b = random.randn(m)
    Q_1, R_1 = householder_qr(A_1)
    b_1 = np.linalg.inv(Q_1).dot(b)
    R_2 = cw1.Rv_array(A_2)
    b_2 = cw1.Q_starb(R_2, b.copy())
    assert(b_1.all() == b_2.all())

# 4(d): test Q_starb(R_v, b)
@pytest.mark.parametrize('m, n', [(20, 17), (40, 3), (20, 12)])
def test_Q_starb(m, n):
    random.seed(1878*m + 1950*n)
    A = random.randn(m, n)
    b = random.randn(m)
    A_hat = np.column_stack((A,b))
    R = householder(A_hat, n)
    Q_starb_1 = R[:, n]
    R_v = cw1.Rv_array(A)
    Q_starb_2 = cw1.Q_starb(R_v, b.copy())   
    assert(np.linalg.norm(Q_starb_1 - Q_starb_2) < 1.0e-6)

# 4(e): test R_v_ls(R_v, b)
@pytest.mark.parametrize('m, n', [(20, 17), (40, 3), (20, 12)])
def test_R_v_ls(m, n):
    random.seed(1878*m + 1950*n)
    A = random.randn(m, n)
    b = random.randn(m)
    R_v = cw1.Rv_array(A)
    x_1 = cw1.R_v_ls(R_v, b)
    x_2 = householder_ls(A, b)
    assert(np.linalg.norm(x_1 - x_2) < 1.0e-6)

# 5(c): test solve_x(A, b, lam)

@pytest.mark.parametrize('m, n, lam', [(20, 17, 2), (40, 3, 4), (20, 12, 12)])
def test_R_v_ls(m, n, lam):
    random.seed(1878*m + 1950*n)
    A = random.randn(m, n)
    b = random.randn(m)
    x_c = cw1.solve_x(A, b, lam)
    m, n = A.shape
    Q, R = householder_qr(A)
    LHS = np.matmul(np.matmul(np.matmul(R.T, Q.T), Q), R) + lam * np.identity(n)
    RHS = np.dot(A.T, b) 
    #LHS = np.matmul(A.T, A) + lam * np.identity(n)
    #RHS = np.dot(A.T, b)
    x_orig = solve_triangular(LHS, RHS)
    #x_orig = np.inner(np.linalg.inv(LHS),RHS)
    assert(np.linalg.norm(x_c - x_orig) < 1.0e-6)


# 5(d): test com_lam(A, b, lam, e)
@pytest.mark.parametrize('m, n, lam, er', [(20, 17, 2, 0.003), (40, 3, 4, 0.006), (20, 12, 12, 0.0004)])
def test_com_lam(m, n, lam, er):
    random.seed(1878*m + 1950*n)
    A = random.randn(m, n)
    b = random.randn(m)
    lam_d = cw1.com_lam(A, b, lam, er)
    x = cw1.solve_x(A, b, lam_d)
    assert(abs(np.linalg.norm(x) - 1) < er)

# 5（e）


