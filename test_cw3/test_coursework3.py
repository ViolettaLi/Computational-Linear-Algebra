import pytest
from numpy import random
import sys
sys.path.append("/Users/appler/Desktop/comp-lin-alg-course/cw3")
print(sys)
import numpy as np
from cw3.CW3 import pure_QR_A_ev, create_A, pure_QR_A_ev_new, create_A_d, u_to_v_wb, dx_to_v, dy_to_v, v_to_u_wb, v_to_d, H_apply
#from cw3.CW3 import u_to_v_nb, v_to_u_nb
from cla_utils.exercises9 import pure_QR
from cla_utils.exercises10 import GMRES


#test 2c
@pytest.mark.parametrize('n', [20, 30, 14])
def testpure_QR_A_ev(n):
    A = create_A(n)
    lamb1 = pure_QR_A_ev(n, maxit=1000, tol=1.0e-5)
    lamb1 = np.sort_complex(lamb1) 
    lamb2 = np.linalg.eig(A)[0]
    lamb2 = np.imag(lamb2) * 1j
    lamb2 = np.sort_complex(lamb2) 
    err = lamb1 - lamb2
    assert(np.linalg.norm(err)) < 1.0e-4


"""
def create_A_d(n):
    A = np.zeros((2 * n, 2 * n), dtype = complex)
    for i in range(2 * n - 1):
        A[i, i + 1] = 2
        A[i + 1, i] = -1
    return A"""


#test 2d
@pytest.mark.parametrize('n', [20, 3, 14])
def testpure_QR_A_ev_new(n):
    A = create_A_d(n)
    lamb1 = pure_QR_A_ev_new(n, maxit=3000, tol=1.0e-5)
    lamb1 = np.sort_complex(lamb1) 
    lamb2 = np.linalg.eig(A)[0]
    lamb2 = np.imag(lamb2) * 1j
    lamb2 = np.sort_complex(lamb2) 
    err = lamb1 - lamb2
    assert(np.linalg.norm(err)) < 1.0e-4


# Create symmetric matrix
def create_symA(n):
    A = random.randn(n, n)
    A = np.triu(A)
    A = A + A.T - np.diag(A.diagonal())
    return A


# test 3b
@pytest.mark.parametrize('m', [20, 30, 18])
def test_pure_QR_cw3b(m):
    random.seed(1302*m)
    A = create_symA(m) + 1j*create_symA(m)
    A = 0.5*(A + A.conj().T)
    A0 = 1.0*A
    A2 = pure_QR(A0, maxit=10000, tol=1.0e-5, sym = True)
    #check it is still Hermitian
    assert(np.linalg.norm(A2 - np.conj(A2).T) < 1.0e-4)
    #check for conservation of trace
    assert(np.abs(np.trace(A0) - np.trace(A2)) < 1.0e-6)


# test 3c
@pytest.mark.parametrize('m', [20, 30, 18])
def test_pure_QR_cw3c(m):
    random.seed(1302*m)
    A = create_symA(m) + 1j*create_symA(m)
    A = 0.5*(A + A.conj().T)
    A0 = 1.0*A
    A2 = pure_QR(A0, maxit=10000, tol=1.0e-5, sym = True)
    #check it is still Hermitian
    assert(np.linalg.norm(A2 - np.conj(A2).T) < 1.0e-4)
    # check eigenvalues
    lamb1 = np.linalg.eig(A)[0]
    lamb1 = np.sort_complex(lamb1) 
    lamb2 = np.linalg.eig(A2)[0]
    lamb2 = np.sort_complex(lamb2) 
    err = lamb1 - lamb2
    assert(np.linalg.norm(err)) < 1.0e-4
    #check for conservation of trace
    assert(np.abs(np.trace(A0) - np.trace(A2)) < 1.0e-6)

# test 3d
@pytest.mark.parametrize('m', [20, 30, 18])
def test_pure_QR_cw3d(m):
    random.seed(1302*m)
    A = create_symA(m) + 1j*create_symA(m)
    A = 0.5*(A + A.conj().T)
    A0 = 1.0*A
    A2 = pure_QR(A0, maxit=10000, tol=1.0e-5, sym = True, shift = True)
    #check it is still Hermitian
    assert(np.linalg.norm(A2 - np.conj(A2).T) < 1.0e-4)
    # check eigenvalues
    lamb1 = np.linalg.eig(A)[0]
    lamb1 = np.sort_complex(lamb1) 
    lamb2 = np.linalg.eig(A2)[0]
    lamb2 = np.sort_complex(lamb2) 
    err = lamb1 - lamb2
    assert(np.linalg.norm(err)) < 1.0e-4
    #check for conservation of trace
    assert(np.abs(np.trace(A0) - np.trace(A2)) < 1.0e-6)


# Test 4(a)
# Test function u_to_v which can serialise normal two-dimensional matrix u row by row
"""
@pytest.mark.parametrize('m', [20, 30, 18])
def test_u_to_v_nb(m):
    random.seed(1878*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    v = u_to_v_nb(A)
    l1 = len(v)
    assert(l1 - (m - 2) * (m - 2)) == 0
"""


@pytest.mark.parametrize('m', [20, 30, 18])
def test_u_to_v_wb(m):
    random.seed(1878*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    v = u_to_v_wb(A)
    l1 = len(v)
    assert(l1 - m * m) == 0


# Test function dx_to_v which can serialise 2d matrix dx with different size (different dimension) into 1d array row by row
@pytest.mark.parametrize('m', [20, 30, 18])
def test_dx_to_v(m):
    random.seed(1878*m)
    A = random.randn(m + 1, m) + 1j*random.randn(m + 1, m)
    v = dx_to_v(A)
    l1 = len(v)
    assert(l1 - (m + 1) * m) == 0


# Test function dy_to_v which can serialise 2d matrix dy with different size (different dimension) into 1d array row by row
@pytest.mark.parametrize('m', [16, 25, 36])
def test_dy_to_v(m):
    random.seed(1878*m)
    A = random.randn(m, m + 1) + 1j*random.randn(m, m + 1)
    v = dy_to_v(A)
    l1 = len(v)
    assert(l1 - (m + 1) * m) == 0


"""
# Test the inverse function from 1d array into the normal two-dimensional array of u
@pytest.mark.parametrize('m', [20, 30, 18])
def test_v_to_u(m):
    random.seed(1878*m)
    A = random.randn(m) + 1j*random.randn(m)
    u = v_to_u(A)
    l1 = len(u[0])
    assert(l1 * 2 - m) == 0
"""


# Test the inverse function from 1d array into the normal two-dimensional array of u
@pytest.mark.parametrize('m', [25, 36, 9])
def test_v_to_u_wb(m):
    random.seed(1302*m)
    v= random.randn(m) 
    v0 = v.copy()
    u=v_to_u_wb(v0)
    n,n=u.shape
    norm1=np.linalg.norm(u)
    norm2=np.linalg.norm(v)
    #check shape
    assert(np.abs(n*n-m)<1e-4)
    #check for values
    assert(np.abs(norm1-norm2) < 1.0e-6)


"""
@pytest.mark.parametrize('m', [25, 36, 9])
def test_v_to_u_nb(m):
    random.seed(1302*m)
    v = random.randn(m) 
    v0 = v.copy()
    u = v_to_u_nb(v0)
    n, n = u.shape
    norm1 = np.linalg.norm(u)
    norm2 = np.linalg.norm(v)
    #check shape
    assert(np.abs(n - int(np.sqrt(m) + 2)) < 1e-4)
    #check for values
    assert(np.abs(norm1 - norm2) < 1.0e-6)
"""


# Test the inverse function from 1d array into the 2d matrix with different sizes (different dimensions)
@pytest.mark.parametrize('m, x', [(12, True), (20, True), (30, False)])
def test_v_to_d(m, x):
    random.seed(1878*m)
    A = random.randn(m) + 1j*random.randn(m)
    u = v_to_d(A, x)
    l1, l2 = u.shape
    assert(l1 * l2 - m) == 0

# test 4(b)
@pytest.mark.parametrize('n, l, m', [(16, 0.3, 2), (25, 12, 0.4), (36, 1.2, 3.7)])
def test_H_apply(n, l, m):
    random.seed(1878*n)
    # Create an array randomly
    v1= random.randn(n) + 1j*random.randn(n)
    sumv0 = sum(v1)
    div = sumv0/n
    v = v1 - div
    L = H_apply(v, l, m)
    suml = sum(L)
    assert(suml) < 1.0e-6


# test 4(c)
@pytest.mark.parametrize('m', [20, 204, 18])
def test_GMRES_Afunction(m):
    A = random.randn(m,m) 
    x = random.randn(m)
    def function(x):
        return np.dot(A, x)
    # For A is an matrix
    x0, _ = GMRES(A, x, maxit = 1000, tol = 1.0e-3)
    # When A is a function
    x1, _ = GMRES(function, x, maxit = 1000, tol = 1.0e-3, function_A = True)
    assert(np.linalg.norm(np.dot(A, x0) - x) < 1.0e-3)
    assert(np.linalg.norm(function(x1) - x)<1.0e-3)


# test 4(e)