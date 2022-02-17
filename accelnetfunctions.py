import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
import matplotlib as mpl
mpl.rc('image', cmap='Paired')
import scipy.stats as stats
from scipy.special import zeta as zeta
# import tensorflow as tf
from sklearn.decomposition import PCA
import random
import matplotlib.colors as colors
import matplotlib.cm as cmx
jet = cm = plt.get_cmap('Paired')
cNorm  = colors.Normalize(vmin=0, vmax=12)
cMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

from matplotlib.colors import ListedColormap

import numba
@numba.njit
def hist1d(v,b):
    return np.histogram(v, b)

def Cov(x):
    N = x.shape[0]
    x_centered = x - np.mean(x, axis = 0)
    return  (x_centered.T @ x_centered) / N

def DimPR(X):
    dimpr = np.trace(X) ** 2 / np.trace(X @ X)
    return dimpr

def ScalePR(X):
    scalepr = np.trace(X @ X) / np.trace(X)
    return scalepr

def multidimensional_shifting(num_samples, sample_size, elements, probabilities=None):
    if probabilities == None:
        probabilities = np.ones_like(elements) / np.prod(elements.shape)
    replicated_probabilities = np.tile(probabilities, (num_samples, 1))
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    shifted_probabilities = random_shifts - replicated_probabilities
    return elements[np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]].squeeze()

def Dim_Corr_Takens(x, dists=None):
    n_components = np.min([x.shape[1], 30])
    pca = PCA(n_components=n_components, svd_solver='full')
    x = pca.fit_transform(x)
    if dists is None:
        dists = sklm.pairwise_distances(x)
        dists = dists[np.triu_indices_from(dists, k=1)]
        dists = dists[dists > 0]
    bins_min, bins_max = dists.min(), dists.max()
    bins = np.logspace(np.log(bins_min), np.log(bins_max), 100, base = np.e)
    # bins_pcg = np.percentile(dists, np.arange(0, 100, 1))
    # bins = np.logspace(np.log(bins_pcg[0]), np.log(bins_pcg[99]), 100, base=np.e)
    Npairs = x.shape[0] * (x.shape[0] - 1)
    Dcorr = np.zeros(len(bins)-1)
    for i_bin, bin in enumerate(bins[1:]):
        dists_bin = dists[dists < bin]
        Dcorr[i_bin] = - 1 / (np.nanmean(np.log(dists_bin / bin)))
    logmidpoints = np.log(bins[1:])
    return logmidpoints, Dcorr

def scalelog(x, dists = False, Npoints=100):
    if not dists:
        allx = np.concatenate([np.abs(np.diff(np.random.permutation(x[:,i_dim]))) for i_dim in np.arange(x.shape[1])])
    else:
        allx=x
    # N_points = x.shape[0]
    # min_pcg = min_points/N_points
    # pcgslog = np.arange(np.log(min_pcg),0, np.log(np.min([0.1, min_pcg])))
    # pcgslin = np.concatenate([np.exp(pcgslog), np.arange(1.1,100.,.1)])

    bins_pcg = np.percentile(allx, [0.0, 100.])
    scales = np.logspace(np.log(bins_pcg[0]), np.log(bins_pcg[-1]), Npoints, base=np.e)
    # pcgslin = np.arange(min_pcg,100.,np.min([.1,min_pcg]))
    # if includemax:
    #     pcgslin = np.concatenate([pcgslin, [100.]])
    # scales = np.percentile(allx, pcgslin)
    return scales

#%
def Dim_Renyi(x, q=2, bins=None):
    def boxcount(Z, scale):
        dims = Z.shape[1]
        ranges = [np.arange(Z[:,i_dim].min(), Z[:,i_dim].max(), scale) for i_dim in np.arange(dims)]
        H, edges = np.histogramdd(Z, bins=ranges, density=False)
        return H
    if bins is None:
        dists_matrix = sklm.pairwise_distances(x, n_jobs=-1)
        dists = dists_matrix[np.triu_indices_from(dists_matrix, k=1)]
        dists = dists[dists > 0]
        bins = scalelog(dists, True)
    Dim_q = np.zeros_like(scales)
    N_tot = x.shape[0]
    for i_scale, scale in enumerate(bins):
        H = boxcount(x, scale)
        if np.mean(H[H>0]) < 1.01:
            Dim_q[i_scale] = np.nan
            continue
        H = H / N_tot
        Dim_q[i_scale] = 1/(1-q) * np.log(np.sum(H[H>0]**q)) / np.log(1/scale)
    return bins, Dim_q

#%

def Dim_Corr(x, dists=None, bins = None):
    if x is not None:
        n_components = np.min([x.shape[1], 30])
        if n_components < x.shape[1]:
            pca = PCA(n_components=n_components, svd_solver='full')
            x = pca.fit_transform(x)
    if dists is None:
        dists = sklm.pairwise_distances(x)
        dists = dists[np.triu_indices_from(dists, k=1)]
        dists = dists[dists>0]
    if bins is None:
        bins = scalelog(dists, dists=True)
    Npairs = len(dists)#x.shape[0] * (x.shape[0] - 1)
    base = bins.copy()
    # if np.min(bins) > np.min(dists):
    #     base = np.concatenate([[np.min(dists)], base])
    values, base = np.histogram(dists[dists>0], bins=base)
    cdf = np.cumsum(values) / np.sum(values)
    idx_min = np.where(np.abs(np.diff(values[1:-10]))<Npairs*0.00001)[0][-1]+2
    cdf[:idx_min] = np.nan
    # base[:idx_min] = np.nan
    midpoints = (base[:-1] + base[1:]) / 2
    xd = np.log(midpoints)
    yd = np.log(cdf)
    Dcorr = np.diff(yd) / np.diff(xd)
    logmidpoints = (xd[:-1] + xd[1:]) / 2
    # p = np.polyfit(logmidpoints[:3], Dcorr[:3], 1)
    return logmidpoints, Dcorr

def boot_dim_corr(dists_matrix, scales, boot_pcg=0.8, N_bootstrap=10):
    N_tot = dists_matrix.shape[0]
    Dcorr = []
    for i_bootstrap in np.arange(N_bootstrap):
        print(i_bootstrap)
        np.random.seed(i_bootstrap)
        idxs = multidimensional_shifting(1, int(N_tot * boot_pcg), np.arange(N_tot))
        dists_matrix_boot = dists_matrix[np.ix_(idxs, idxs)]
        dists_boot = dists_matrix_boot[np.triu_indices_from(dists_matrix_boot, k=1)]
        dists_boot = dists_boot[dists_boot > 0]
        logmidpoints, Dcorr_boot = Dim_Corr(None, dists_boot, bins=scales)
        Dcorr.append(Dcorr_boot)
    Dcorr = np.vstack(Dcorr)
    return Dcorr, logmidpoints


def boot_dim_renyi(x, scales, q=0, boot_pcg=0.8, N_bootstrap=10):
    N_tot = x.shape[0]
    Drenyi = []
    for i_bootstrap in np.arange(N_bootstrap):
        print(i_bootstrap)
        np.random.seed(i_bootstrap)
        idxs = multidimensional_shifting(1, int(N_tot * boot_pcg), np.arange(N_tot))
        x_boot = x[idxs,:]
        logmidpoints, Dcorr_boot = Dim_Renyi(x_boot, q=q, bins=scales)
        Drenyi.append(Dcorr_boot)
    Drenyi = np.vstack(Drenyi)
    return scales, Drenyi


def Dim_PRs(x):
    pca = PCA(n_components=30, svd_solver='full')
    x = pca.fit_transform(x)
    dists = sklm.pairwise_distances(x)
    bins_pcg = np.percentile(dists[dists > 0], np.arange(0, 100, 1))
    sigma_space = np.logspace(np.log(bins_pcg[0]), np.log(bins_pcg[99]), 100)
    N_points = 50
    PRs = np.zeros((sigma_space.shape[0], N_points))
    Scales = np.zeros((sigma_space.shape[0], N_points))
    points = np.random.choice(np.arange(x.shape[0]), N_points)
    for i_point, point in enumerate(points):
        for i_sigma, sigma in enumerate(sigma_space):
            x_sigma = x[np.where(dists[point] < sigma)[0]]
            PRs[i_sigma, i_point] = DimPR(Cov(x_sigma))
            Scales[i_sigma, i_point] = ScalePR(Cov(x_sigma))
    return sigma_space, Scales, PRs


def boot_dim_pr(x, dists_matrix, scales, N_points=100, N_maxiters=1000, C_minsize=50):
    PRs = np.nan*np.zeros((scales.shape[0], N_points))
    Scales = np.nan*np.zeros((scales.shape[0], N_points))
    for i_sigma, sigma in enumerate(scales):
        i_sigma
        print(i_sigma)
        i_point = 0
        tot_iters = 0
        while ((i_point < N_points) and (tot_iters < N_maxiters)):
            tot_iters += 1
            point = np.random.choice(np.arange(dists_matrix.shape[0]), 1)
            x_sigma = x[np.where(dists_matrix[point] < sigma)[1]]
            if x_sigma.shape[0] < C_minsize:
                continue
            PR = DimPR(Cov(x_sigma))
            if np.isnan(PR):
                continue
            PRs[i_sigma, i_point] = PR
            Scales[i_sigma, i_point] = ScalePR(Cov(x_sigma))
            i_point = i_point + 1
    return PRs, Scales


def dataset_spiral(N_points, sigma_noise):
    # r = 10 * np.random.rand(N_points)
    r = 20 * np.random.triangular(0, 1, 1, N_points)
    r = np.sort(r)
    x_coor = r/20 * np.sin(r) + sigma_noise * np.random.rand(N_points)
    y_coor = r/20 * np.cos(r) + sigma_noise * np.random.rand(N_points)
    return np.array([x_coor, y_coor]).T
    # plt.scatter(x_coor, y_coor)
    # plt.show()


def dataset_swissroll(N_points, sigma_noise):
    r = 20 * np.random.rand(N_points)
    r = np.sort(r)
    x_coor = r/20 * np.sin(r) + sigma_noise * np.random.rand(N_points)
    y_coor = r/20 * np.cos(r) + sigma_noise * np.random.rand(N_points)
    z_coor = 2*np.random.rand(N_points)-1
    return np.array([x_coor, y_coor, z_coor]).T
    # plt.scatter(x_coor, y_coor)
    # plt.show()

def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def dataset_lorenz(N_points):
    dt = 0.01
    num_steps = N_points
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    return np.array([xs, ys, zs]).T[1:]