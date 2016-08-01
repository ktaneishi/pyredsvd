#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.decomposition
import sklearn.datasets
import os

def scale(df, center=True, scale=True):
    """ compatible with R scale() """
    if center:
        df = df - df.mean()
    if scale:
        df =  df / df.std()
    return df

class PCA:
    def __init__(self, df, npc=2, method="svd"):
        self.df = df
        self.npc = npc
        self.method = method

    def svd(self):
        """ SVD decomposite X to U*diag(S)*Wt """
        self.U, self.diagS, self.Wt = np.linalg.svd(self.df, full_matrices=False)

        # singular values are already sorted
        assert np.all(self.diagS[:-1] >= self.diagS[1:]) 

        """ T = X*W = U*S """
        pcXW = np.dot(self.df, self.Wt.T[:,:self.npc])
        pcUdS = self.U[:,:self.npc] * self.diagS[:self.npc]
        # these two ways return same results
        assert np.allclose(pcXW, pcUdS)
        self.score = pcUdS

    def eig(self):
        # calculate eigenvalues(l) and eigenvectors(w) of the covariance matrix
        C = np.cov(self.df.T)
        self.l,self.w = np.linalg.eig(C)

        # sort eigenvectors by eigenvalue in descending order
        self.w = self.w[:,np.argsort(self.l)[::-1]]

        """ T = X*W """
        self.score = np.dot(self.df, self.w[:,:self.npc])

    def pc(self):
        if self.method == "svd":
            self.svd()
        else:
            self.eig()
        return self.score

def plot(df):
    for name in df['target'].unique():
        cond = df['target'] == name
        plt.plot(df.ix[cond,'PC1'], df.ix[cond,'PC2'], 'o', label=name)
    plt.legend(framealpha=0.5)

if __name__ == "__main__":
    np.set_printoptions(precision=4, threshold=30)

    # famous Fisher iris data
    dataset = sklearn.datasets.load_iris()
    df = pd.DataFrame(dataset['data'])
    df.columns = dataset['feature_names']
    df['target'] = dataset['target']

    # use only numerical columns
    values = scale(df.ix[:,:4], center=True, scale=True)
    print 'DataFrame is a (%d x %d) matrix of "iris.csv":\n' % (values.shape[0], values.shape[1]) 

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title('Singular value decomposition')
    pca_svd = PCA(values,method='svd')
    score = pd.DataFrame(pca_svd.pc(), columns=['PC1','PC2'])
    score = pd.concat([df, score], axis=1)
    plot(score)

    plt.subplot(1,2,2)
    plt.title('Eigenvalue decomposition')
    pca_eig = PCA(values,method='eig')
    score = pd.DataFrame(pca_eig.pc(), columns=['PC1','PC2'])
    score = pd.concat([df, score], axis=1)
    plot(score)

    pca_sklearn = sklearn.decomposition.PCA(n_components=2)
    score = pca_sklearn.fit(values).transform(values)

    assert np.allclose(score, pca_svd.pc())

    plt.tight_layout()
    plt.savefig('pca.png')

    # assert two princople components based on SVD and EigenDecomposition
    print 'PCA(svd) == PCA(eig) is %s' % np.allclose(pca_svd.pc(),pca_eig.pc())
