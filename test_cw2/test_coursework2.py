import pytest
import cla_utils
from numpy import random
import numpy as np
from cla_utils.exercises6 import LU_inplace
from cla_utils.exercises7 import solve_LUP
from cw2.CW2 import CWLU_inplace, creatA, solve_bandtri, CWLU_inplace5

#Test for 4(c) by checking A = LU
@pytest.mark.parametrize('n, epsilon', [(20, 0.004), (204, 0.00001), (18, 0.07)])
def testA_CWLU_inplace(n, epsilon):
    random.seed(8564*n)
    A = creatA(n, epsilon)
    A0 = 1.0*A
    CWLU_inplace(A)
    L = np.eye(4 * n + 1)
    i1 = np.tril_indices(4 * n + 1, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
    A1 = np.dot(L, U)
    err = A1 - A0
    assert(np.linalg.norm(err) < 1.0e-6)

#Test for 4(c) by comparing the 1 loop LU_inplace in exercise6 and this new LU_inplace
@pytest.mark.parametrize('n, epsilon', [(20, 0.004), (204, 0.00001), (18, 0.07)])
def testLU_CWLU_inplace(n, epsilon):
    random.seed(8564*n)
    A = creatA(n, epsilon)
    A0 = 1.0*A
    CWLU_inplace(A)
    L1 = np.eye(4 * n + 1)
    i1 = np.tril_indices(4 * n + 1, k=-1)
    L1[i1] = A[i1]
    U1 = np.triu(A)
    LU_inplace(A0)
    L0 = np.eye(4 * n + 1)
    i1 = np.tril_indices(4 * n + 1, k=-1)
    L0[i1] = A0[i1]
    U0 = np.triu(A0)
    errL = L1 - L0
    errU = U1 - U0
    assert(np.linalg.norm(errL) < 1.0e-6)
    assert(np.linalg.norm(errU) < 1.0e-6)


# 4e autotest
@pytest.mark.parametrize('n, epsilon', [(20, 0.004), (204, 0.00001), (18, 0.07)])
def testsolve_bandtri(n, epsilon):
    random.seed(8564*n)
    m = 4*n + 1
    A = creatA(n, epsilon)
    b = random.randn(m)
    x0 = solve_bandtri(A, b)
    x1 = solve_LUP(A, b)
    err = x0 - x1
    assert(np.linalg.norm(err) < 1.0e-6)


# 5b autotest
@pytest.mark.parametrize('n', [20,30,40])
def test_bandedLU(n):
    random.seed(1713*n)
    m = (n-1) ** 2
    A = np.zeros((m,m))
    l = np.array([1-n,-1,0,1,n-1])
    for i in l:
        A += np.diag(np.random.randn(m-abs(i)),i)
    A0 = 1.0*A
    CWLU_inplace5(A)
    L = np.eye(m)
    i1 = np.tril_indices(m, k=-1)
    L[i1] = A[i1]
    U = np.triu(A)
    A1 = np.dot(L, U)
    err = A1 - A0
    assert(np.linalg.norm(err) < 1.0e-6)