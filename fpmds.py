"""
Reference: Peltason L et al. Rationalizing three-dimensional activity landscapes 
and the influence of molecular representations on landscape topology and formation of activity cliffs. 
J Chem Inf Model, J Chem Inf Model 50, 1021-1033, 2010.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pybel
import gzip

def scale(X, center=True, scale=True):
    """ compatible with GNU R scale() """
    if center:
        X -= X.mean(axis=0)

    if scale:
        std = X.std(axis=0)
        # assign 1. if std is False
        std = np.where(std, std, 1.)
        X /= std
    return X

def pca(X,npc=2):
    # calculate eigenvalues(l) and eigenvectors(w) of the covariance matrix
    C = np.cov(X.T)
    l,w = np.linalg.eig(C)

    # sort eigenvectors by eigenvalue in descending order
    w = w[:,np.argsort(l)[::-1]]
    """ T = X*W """
    pc = np.dot(X, w[:,:npc])
    return pc

def fp_mds(fptype):
    fpss = []
    for mol in pybel.readfile('sdf', 'solubility.test.sdf'):
        fps = mol.calcfp(fptype=fptype).bits
        if len(fps) > 0: 
            fpss.append(fps)

    cols = max(max(fps) for fps in fpss)
    mat = np.zeros((len(fpss), cols+1), dtype=np.float32)

    for i,fps in enumerate(fpss):
        mat[i,fps] = 1.0

    mat = scale(mat)
    pcs = np.real(pca(mat,npc=2))

    activities = np.random.randn(mat.shape[0]) # dummy activity

    # Set up a regular grid of interpolation points
    xi, yi = np.linspace(pcs[:,0].min(), pcs[:,0].max()), np.linspace(pcs[:,1].min(), pcs[:,1].max())
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate
    rbf = scipy.interpolate.Rbf(pcs[:,0], pcs[:,1], activities, function='linear', smooth=0.1)
    zi = rbf(xi, yi)

    plt.subplot(2,2,pybel.fps[-4:].index(fptype)+1)
    plt.title('%s' % fptype)
    plt.imshow(zi, vmin=zi.min(), vmax=zi.max(), origin='lower', cmap='RdYlGn_r', aspect='auto',
            extent=[pcs[:,0].min(), pcs[:,0].max(), pcs[:,1].min(), pcs[:,1].max()])
    plt.scatter(pcs[:,0], pcs[:,1], c=activities, cmap='RdYlGn_r')
    
def main():
    plt.figure(figsize=(8,8))
    for fptype in pybel.fps[-4:]:
        fp_mds(fptype)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
