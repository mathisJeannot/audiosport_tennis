from sklearn import decomposition
import convolutive_MM
import nimfa
import numpy as np
import nmf_from_sklearn as sknmf

def NMF_old(V: np.array, rank: int, beta=2, max_iter=1000000, init='random', random_state=0, solver='mu', tol=1e-4):
    """
    Algorithm for NMF
    ================================
    :param V: matrix to be approached
    :param rank: rank of the NMF
    :param beta: beta in a beta divergence
    :param init: type of initialisation for W and H
    :param random_state: used for initialisation
    :param solver: numerical solver to use ('cd' or 'mu')
    :param tol: tolerence of the stopping condition
    :param W_fixed: fixed components for W
    --------------------------------
    :return: matrix W, matrix H
    """ 
    model = decomposition.NMF(n_components=rank, max_iter=max_iter, init=init, random_state=random_state, solver=solver, beta_loss=beta, tol=tol)
    W = model.fit_transform(V)
    H = model.components_
    return W,H

def NMF(V: np.array, rank: int, beta:int, max_iter=1000000, tol=1e-4, verbose=0, W0=None, H0=None, components_fixed=None):
    if W0 is None:
        W0 = np.random.rand(V.shape[0], rank)
    if H0 is None:
        H0 = np.random.rand(rank, V.shape[1])
    return sknmf._fit_multiplicative_update(X=V, W=W0, H=H0, beta_loss=beta, max_iter=max_iter, tol=tol, verbose=verbose, components_W=components_fixed)[:2]

def SNMF(V: np.array, rank: int, max_iter=1000000, min_residual=1e-4, beta=1e-4, seed="random_vcol", W0=None, H0=None, W_fixed=None):
    """
    Algorithm for NMF with sparce H
    ================================
    :param V: the matrix to be approached
    :param beta: control the sparceness. Beta high => high parcimonie
    :param seed: Method used if H0=None and W0=None 
    :param W0: initialisation of W
    :param H0: initialisation of H
    :param rank: rank of the NMF
    - Stopping criterion -
    :param max_iter: max number of iteration
    :param min_residuals: minimum error for the divergence
    :param W_fixed: fixed components for W
    --------------------------------
    :return: matrix W, matrix H
    """
    snmf = nimfa.Snmf(V, seed=seed, rank=rank, max_iter=max_iter, min_residual=min_residual,
                      version='r', eta=1., beta=beta, i_conv=10, w_min_change=0, W=W0, H=H0)
    snmf_fit = snmf()
    return np.array(snmf.basis()), np.array(snmf.coef())



def CNMF(V: np.array, rank: int, tau: int, beta=2, max_iter=1000000, e=1e-4, W0=None, H0=None, components_fixed=[]):
    """
    Algorithm MM for cnmf
    ================================
    :param V: the matrix to be approached
    :param rank: factorization rank
    :param itmax: limit number of iteration
    :param beta: beta in beta divergence
    :param tau: dictionary number
    :param e: relative err tolerance
    :param W0: initialization of W
    :param H0: initialization of H
    :param W_fixed: fixed components for W
    --------------------------------
    :return: matrix W, matrix H
    """
    return convolutive_MM.convolutive_MM(V, rank, max_iter, beta, tau, e, W0, H0, components_fixed)[:2]


def C_product(W,H):
    return sum(np.dot(W[t], convolutive_MM.shift(H,t)) for t in range(W.shape[0]))


def thresholding(H: np.array, r_type='avg', rate=1):
    """
    Algorithm for H thresholding
    ================================
    :param H: activation matrix
    :param type: type of the reference for the threshold ('avg' or 'max')
    :param rate: threshold is rate times the average (or maximum) of the values of H
    --------------------------------
    :return: H_thresholded
    """
    H_threshold = np.zeros(H.shape)
    if r_type == 'avg':
        avg = np.mean(H)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if H[i][j] >= rate*avg:
                    H_thresholded[i][j] = H[i][j]
    elif r_type == 'max':
        maxi = np.max(H)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if H[i][j] >= rate*maxi:
                    H_thresholded[i][j] = H[i][j]
    return H_thresholded

def arrondi(x:float, n:int):
    return int(x*(10**n))/(10**n)
    