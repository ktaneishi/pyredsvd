import numpy as np
from scipy import linalg, sparse
from sklearn import datasets
import timeit
import gzip

SVD_EPS = 1e-4

def svd(mat, k=10):
    from scipy.sparse.linalg import svds
    return svds(mat,k=k)

def GramSchmidt(mat):
    for i in range(mat.shape[1]):
        for j in range(i):
            r = np.dot(mat[:,i], mat[:,j]);
            mat[:,i] -= r * mat[:,j];
        norm = linalg.norm(mat[:,i])
        if norm < SVD_EPS:
            for k in range(i, mat.shape[1]):
                mat[:,k] = 0.0
            return
        mat[:,i] *= 1.0 / norm

def redsvd(A, k=10):
    O = np.random.randn(A.shape[0]*k).reshape(A.shape[0],k)
    Y = A.T.dot(O)
    GramSchmidt(Y)
    B = A.dot(Y)

    P = np.random.randn(k*k).reshape(k,k)
    Z = np.dot(B, P)
    GramSchmidt(Z)
    C = np.dot(Z.T, B)

    U, S, V = linalg.svd(C)

    return S

if __name__ == '__main__':
    filename = 'news20.binary.gz'
    A, target = datasets.load_svmlight_file(gzip.open(filename))

    start_time = timeit.default_timer()
    print(redsvd(A))
    print(timeit.default_timer() - start_time)
