import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import os
import random
import torch
from sklearn.neighbors import NearestNeighbors

def set_seed(seed=9):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_spatial(mdata, spatial_key='spatial', modality_key='physical'):
    try:
        x, y = mdata[modality_key].obsm[spatial_key].T
    except KeyError:
        x, y = mdata.obsm[spatial_key].T
    
    return x, y
    
def get_features(mdata, feature_key='protein'):
    try:
        features = mdata[feature_key].X
    except:
        features = mdata.X

    return features
    
def get_labels(mdata, label_key='celltype', modality_key='protein'):
    try:
        labels = mdata[modality_key].obs[label_key]
    except:
        labels = mdata.obs[label_key]

    return labels

def remove_lonely(data, labels, n_neighbors=12, threshold=225.):
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(data[:, :2])
    max_dist = knn.kneighbors()[0].max(-1)
    mask_idx, = np.where(max_dist > threshold)
    data = np.delete(data, mask_idx, axis=0)
    labels = np.delete(labels, mask_idx, axis=0)

    return data, labels

def read_anndata(filename, spatial_key='spatial', feature_key='protein', label_key='celltype', n_neighbors=12, threshold=225., tensor=False):
    mdata = mu.read(filename)
    x, y = get_spatial(mdata, spatial_key)
    features = get_features(mdata, feature_key)
    data = np.hstack([x[None].T, y[None].T, features])
    labels = get_labels(mdata, label_key)
    _, labels = np.unique(labels, return_inverse=True)

    if threshold is not None:
        data, labels = remove_lonely(data, labels, threshold, n_neighbors)

    if tensor:
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.int32)
    return data, labels

def itemize(n, *items):
    for i in items:
        yield i if isinstance(i, (list, tuple)) else (i,)*n

def format(ax, aspect='equal', show_ax=True):
    ax.set_aspect(aspect)
    
    if not show_ax:
        ax.axis('off')

    return ax

def show_dataset(data, labels, size=15, show_ax=True, figsize=10, colormap='tab10', filename=None):
    figsize, = itemize(2, figsize)
    fig, ax = plt.subplots()
    locs = data[:, :2].T
    color = plt.colormaps[colormap](labels)
    ax.scatter(*locs, s=size, c=color)
    ax = format(ax, aspect='equal', show_ax=show_ax)

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

def show_datasets(data, labels, size=15, show_ax=True, figsize=10, colormap='tab10', filename=None):
    n_datasets = np.unique(data[:, 0]).shape[0]
    size, = itemize(n_datasets, size)
    figsize, = itemize(2, figsize)
    fig, ax = plt.subplots(1, n_datasets, figsize=figsize)

    for i in range(n_datasets):
        idx = data[:, 0] == i
        locs = data[idx, 1:3].T
        color = plt.colormaps[colormap](labels[idx])
        ax[i].scatter(*locs, s=size[i], c=color)
        ax[i] = format(ax[i], aspect='equal', show_ax=show_ax)

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
