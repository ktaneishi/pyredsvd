# Randomized Singular Value Decomposition in Python

pyredsvd is a python routine for solving several matrix decompositions including singular value decomposition (SVD), principal component analysis (PCA), and eigen value decomposition. 
pyredsvd can handle very large matrix efficiently, and optimized for a truncated SVD of sparse matrices.
For example, pyredsvd can compute a truncated SVD with top 20 singular values for a 100K x 100K matrix with 1M nonzero entries in less than five second.

The algorithm is based on _a randomized algorithm_ for computing large-scale SVD. Although it uses randomized matrices, _the results is very accurate with very high probability_.

## Requirements

- numpy
- scipy

# Examples

```
$ python redsvd.py 
read sparse matrix from news20.binary ... 33.261 sec.
  rows: 19996
  cols: 1355191
  nonzero: 9097916
  rank: 10
[18.78031779  3.05987719  2.37272713  2.0208399   1.95307871  1.94165703
  1.88591713  1.82173629  1.80488558  1.73713894]
compute SVD... 5.104 sec.

> cat news20.S
+17.973207
+2.556800
+2.460566
+2.135978
+2.022737
+1.931362
+1.927033
+1.853175
+1.770231
+1.764138
```

# Environment

Intel(R) Xeon Phi(TM) CPU 7210 @ 1.30GHz

```
For dense matrices
n	m	rank	time (msec)
500	100	10	0.76
500	1000	10	3.24
500	10000	10	32.3
500	100000	10	306.3
n	m	rank	time (msec)
500	100	100	12.3
500	1000	1000	987.5
500	10000	1000	3850.0
500	100000	1000	32824.3
n	m	rank	time (msec)
100	100	10	0.20
1000	1000	10	6.34
10000	10000	10	578
n	m	rank	time (msec)
100	100	500	8.67
1000	1000	500	8654
10000	10000	500	45001
```

```
For sparse matrices
n	m	rank	nonzero ratio (%)	time (msec)
100	100	10	0.1	0.31
1000	1000	10	0.1	1.17
10000	10000	10	0.1	22.5
100000	100000	10	0.1	1679.9
n	m	rank	nonzero ratio (%)	time (msec)
100	100	10	1.0	0.16
1000	1000	10	1.0	2.0
10000	10000	10	1.0	124.1
100000	100000	10	1.0	12603.4
Latent Semantic Analyasis (SVD for doc-term matrix) of English Wikipedia
The target rank is 10
```

```
# doc	# word	# total words	time (msec)
3560	27106	172823	27
46857	147144	2418406	390
118110	261495	6142438	1073
233717	402239	12026852	1993
```

# Inside of redsvd

Let A be a matrix to be analyzed with n rows and m columns, and r be the ranks of a truncated SVD (if you choose r = min(n, m), then this is the original SVD).

First a random Gaussian matrix O with m rows and r columns is sampled and computes Y = At O. Then apply the Gram-Schmidt process to Y so that each column of Y is ortho-normalized. Then we compute B = AY with n rows and r columns. Although the size of Y is much smaller than that of A, Y holds the informatin of A; that is AYYt = A. Intuitively, the row informatin is compresed by Y and can be decompressed by Yt

Similarly, we take another random Gaussian matrix P with r rows and r columns, and compute Z = BP. As in the previous case, the columns of Z are ortho-normalized by the Gram-Schmidt process. ZZtt B = B. Then compute C = Zt B.

Finally we compute SVD of C using the traditional SVD solver, and obtain C = USVt where U and V are orthonormal matrices, and S is the diagonal matrix whose entriesa are singular values. Since a matrix C is very small, this time is negligible.

Now A is decomposed as A = AYYt = BYt = ZZtBYt = ZCYt = ZUSVtYt. Both ZU and YV are othornormal, and ZU is the left singular vectors and YV is the right singular vector. S is the diagonal matrix with singular values.

# Thanks

redsvd uses the algorithm based on the randomized algorithm described in the following paper.
However, although the algorithm in redsvd samples both rows and columns, the original algorithm samples in one way (this would be because the performance analysis becomes complex).

"Finding structure with randomness: Stochastic algorithms for constructing approximate matrix decompositions", N. Halko, P.G. Martinsson, J. Tropp, arXiv 0909.4061

# TODO

- [ ] PCA
- [ ] SymEigen

# Original author

Original author is [Daisuke Okanohara](https://code.google.com/p/redsvd/).
