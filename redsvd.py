import os
import gzip
import numpy as np
from scipy.sparse import coo_matrix
from scipy import linalg
from scipy.io import savemat, loadmat

SVD_EPS = 1e-4

def build_matrix(filename):
    row = []
    col = []
    data = []
    nrow = 0
    for l in gzip.open(filename):
        seq = l.strip().split()
        cls = seq.pop(0)
        if len(seq) == 0:
            continue
        for ncol,val in map(lambda x:x.split(':'), seq):
            row.append(int(nrow))
            col.append(int(ncol)-1)
            data.append(float(val))
        nrow += 1

    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)

    mat = coo_matrix((data, (row,col)), shape=(row.max()+1,col.max()+1))
    savemat(filename.replace('gz','mat'), {'default':mat})

def svd(mat,k=10):
    from scipy.sparse.linalg import svds
    return svds(mat,k=k)

def GramSchmidt(mat):
    for i in xrange(mat.shape[1]):
        for j in xrange(i):
            r = np.dot(mat[:,i], mat[:,j]);
            mat[:,i] -= r * mat[:,j];
        norm = linalg.norm(mat[:,i])
        if norm < SVD_EPS:
            for k in xrange(i, mat.shape[1]):
                mat[:,k] = 0.0
            return
        mat[:,i] *= 1.0 / norm

def redsvd():
    filename = 'news20.binary.gz'
    if not os.path.isfile(filename.replace('gz','mat')):
        build_matrix(filename)

    k = 10
    A = loadmat(filename.replace('gz','mat'))['default']

    O = np.random.randn(A.shape[0]*k).reshape(A.shape[0],k)
    Y = A.T.dot(O)
    GramSchmidt(Y)
    B = A.dot(Y)

    P = np.random.randn(k*k).reshape(k,k)
    Z = np.dot(B, P)
    GramSchmidt(Z)
    C = np.dot(Z.T, B)

    U,S,V = linalg.svd(C)
    return S

if __name__ == '__main__':
    print redsvd()
