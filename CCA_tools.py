### code from Xinshuai

import numpy as np
from statsmodels.multivariate.cancorr import CanCorr
from math import log, pow
from scipy.stats import chi2
from scipy.linalg import eigh


class EigRankTest:
    """Statistical rank test for a covariance or second-moment matrix.

    Given samples h_1, ..., h_n (rows of ``data``), estimates the rank of
    M = (1/n) sum h_i h_i^T  (center=False)  or  Cov(h) (center=True)
    using Bartlett's sequential sphericity test on the tail eigenvalues.

    Parameters
    ----------
    data : ndarray of shape (n_samples, d)
    center : bool
        If True, test rank of Cov(data).  If False, test rank of E[hh^T].
    """

    def __init__(self, data, center=True):
        self.n, self.d = data.shape
        if center:
            data = data - data.mean(axis=0)
        self.M = data.T @ data / self.n
        self.eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(self.M), 0))[::-1]

    def test(self, r, alpha=0.05):
        """Test H0: rank(M) <= r.

        Uses Bartlett's test for equality of the last (d - r) eigenvalues.

        Returns
        -------
        fail_to_reject : bool
            True means rank <= r is plausible at level alpha.
        p_value : float
        """
        m = self.d - r  # number of tail eigenvalues
        if m <= 1:
            return True, 1.0

        tail = self.eigenvalues[r:]
        arith_mean = tail.mean()
        if arith_mean < 1e-15:
            return True, 1.0

        log_arith = np.log(arith_mean)
        log_geom = np.mean(np.log(np.maximum(tail, 1e-300)))

        # Bartlett correction factor
        nu = max(self.n - 1 - (2 * m + 1) / 3, 1)
        stat = nu * m * (log_arith - log_geom)  # always >= 0 (AM >= GM)
        df = (m - 1) * (m + 2) / 2

        if df <= 0:
            return True, 1.0

        p_value = 1 - chi2.cdf(stat, df)
        return p_value >= alpha, p_value

    def estimate_rank(self, alpha=0.05):
        """Estimate rank: smallest r such that H0: rank <= r is not rejected."""
        for r in range(self.d):
            fail_to_reject, p = self.test(r, alpha)
            if fail_to_reject:
                return r
        return self.d


class Chi2RankTest(object):

    def __init__(self, data, N_scaling=1):

        self.data = data
        self.data = self.data - self.data.mean(axis=0)
        self.data = self.data / self.data.std(axis=0)

        self.N = data.shape[0]
        self.N_scaling = N_scaling
        self.unnormalized_crosscovs = self.data.T @ self.data  # data are zero mean
        self.cca_cache_dict = {}

    def get_cachekey(self, pcols_, qcols_):
        pcols = tuple(sorted((pcols_)))
        qcols = tuple(sorted((qcols_)))
        pcols, qcols = sorted([pcols, qcols])
        cachekey = (pcols, qcols)
        return cachekey

    def test(self, pcols, qcols, r, alpha):
        '''
        Parameters
        ----------
        pcols, qcols : column indices of data
        r: null hypo that rank <= r
        alpha: significance level

        Returns
        -------
        if_fail_to_reject: 0 means reject and 1 means fail to reject
        p : the p-value of the test
        '''
        cachekey = self.get_cachekey(pcols, qcols)

        if cachekey in self.cca_cache_dict:
            cancorr = self.cca_cache_dict[cachekey]
        else:
            X = self.data[:, pcols]
            Y = self.data[:, qcols]

            unnormalized_crosscovs = [self.unnormalized_crosscovs[pcols, :][:, pcols], self.unnormalized_crosscovs[pcols, :][:, qcols], \
                                      self.unnormalized_crosscovs[qcols, :][:, pcols], self.unnormalized_crosscovs[qcols, :][:, qcols]]

            try:
                comps = kcca_modified([X, Y], reg=0.,
                                      numCC=None, kernelcca=False, ktype='linear',
                                      gausigma=1.0, degree=2, crosscovs=unnormalized_crosscovs)

                cancorr, _, _ = recon([X, Y], comps, kernelcca=False)
                cancorr = cancorr[:, 0, 1]
            except:
                print(f"calculating cancorr error {pcols} {qcols}, using slower implementation instead")
                cancorr = CanCorr(X, Y, tolerance=1e-8).cancorr

            self.cca_cache_dict[cachekey] = cancorr

        testStat = 0
        p = len(pcols)
        q = len(qcols)

        l = cancorr[r:]
        for li in l:
            li = min(li, 1 - 1e-15)
            testStat += log(1) - log(1 - li * li)
        ratio = 0
        for i in range(r):
            li = cancorr[i]
            ratio += 1 / (li * li) - 1

        ratio += self.N * self.N_scaling - r - 0.5 * (p + q + 1)
        testStat = testStat * ratio

        dfreedom = (p - r) * (q - r)
        criticalValue = chi2.ppf(1 - alpha, dfreedom)
        p = 1 - chi2.cdf(testStat, dfreedom)
        if_fail_to_reject = testStat <= criticalValue

        # due to numerical errors comparing criticalValue with testStat is more accurate

        return if_fail_to_reject, p, testStat, criticalValue


def kcca_modified(
        data, reg=0.0, numCC=None, kernelcca=False, ktype="linear", gausigma=1.0, degree=2, crosscovs=None
):
    """Set up and solve the kernel CCA eigenproblem"""
    if kernelcca:
        raise NotImplementedError
        # kernel = [
        #    _make_kernel(d, ktype=ktype, gausigma=gausigma, degree=degree) for d in data
        # ]
    else:
        kernel = [d.T for d in data]

    nDs = len(kernel)
    nFs = [k.shape[0] for k in kernel]
    numCC = min([k.shape[0] for k in kernel]) if numCC is None else numCC

    # Get the auto- and cross-covariance matrices
    if crosscovs is None:
        crosscovs = [np.dot(ki, kj.T) for ki in kernel for kj in kernel]

    # Allocate left-hand side (LH) and right-hand side (RH):
    n = sum(nFs)
    LH = np.zeros((n, n))
    RH = np.zeros((n, n))

    # Fill the left and right sides of the eigenvalue problem
    for i in range(nDs):
        RH[
        sum(nFs[:i]): sum(nFs[: i + 1]), sum(nFs[:i]): sum(nFs[: i + 1])
        ] = crosscovs[i * (nDs + 1)] + reg * np.eye(nFs[i])

        for j in range(nDs):
            if i != j:
                LH[
                sum(nFs[:j]): sum(nFs[: j + 1]), sum(nFs[:i]): sum(nFs[: i + 1])
                ] = crosscovs[nDs * j + i]

    LH = (LH + LH.T) / 2.0
    RH = (RH + RH.T) / 2.0

    maxCC = LH.shape[0]
    r, Vs = eigh(LH, RH, subset_by_index=(maxCC - numCC, maxCC - 1))
    r[np.isnan(r)] = 0
    rindex = np.argsort(r)[::-1]
    comp = []
    Vs = Vs[:, rindex]
    for i in range(nDs):
        comp.append(Vs[sum(nFs[:i]): sum(nFs[: i + 1]), :numCC])
    return comp


def _listdot(d1, d2):
    return [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]


def _listcorr(a):
    """Returns pairwise row correlations for all items in array as a list of matrices"""
    corrs = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j > i:
                corrs[:, i, j] = [
                    np.nan_to_num(np.corrcoef(ai, aj)[0, 1])
                    for (ai, aj) in zip(a[i].T, a[j].T)
                ]
    return corrs


def recon(data, comp, corronly=False, kernelcca=False):
    # Get canonical variates and CCs
    if kernelcca:
        ws = _listdot(data, comp)
    else:
        ws = comp
    ccomp = _listdot([d.T for d in data], ws)
    corrs = _listcorr(ccomp)
    if corronly:
        return corrs
    else:
        return corrs, ws, ccomp

