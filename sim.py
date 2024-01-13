import numpy as np
import torch
import util
from sklearn.datasets import make_classification

HSBLOCKS = np.array([[0, 0],
                     [1, 1]], dtype=np.int32)

VSBLOCKS = np.array([[0, 1],
                     [0, 1]], dtype=np.int32)

CHBLOCKS = np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]], dtype=np.int32)

GGBLOCKS = np.array([[0, 1, 0, 2, 2, 2],
                     [1, 1, 1, 2, 0, 2],
                     [0, 1, 0, 2, 2, 2],
                     [3, 0, 0, 4, 4, 4],
                     [3, 3, 0, 0, 4, 0],
                     [3, 3, 3, 0, 4, 0]], dtype=np.int32)

def partition(n_samples, n_partitions):
    parts = [n_samples // n_partitions]

    for i in range(1, n_partitions):
        parts.append((n_samples - sum(parts))//(n_partitions - i))
        
    return parts

def parameterize(n_components, n_features, n_equivocal=0, n_redundant=0, mean_scale=1., variance_scale=1.):
    n_informative = n_features - n_equivocal - n_redundant
    means, _ = make_classification(n_samples=n_components, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_classes=n_components, n_clusters_per_class=1, scale=mean_scale)
    variances = variance_scale*np.random.rand(n_features)

    return means, variances

def designate(n_samples, n_components, n_features, n_equivocal=0, n_redundant=0, sample_counts=None, means=None, variances=None, mean_scale=1., variance_scale=1.):
    if sample_counts is None:
        sample_counts = partition(n_samples, n_components)
    if means is None:
        means, _ = parameterize(n_components, n_features, n_equivocal, n_redundant, mean_scale, variance_scale)
    if variances is None:
        _, variances = parameterize(n_components, n_features, n_equivocal, n_redundant, mean_scale, variance_scale)

    return sample_counts, means, variances

def sample(n_samples, n_components, n_features, means, variances, component, n_mix=0):
    labels = component*np.ones(n_samples, dtype=np.int32)
    data = np.random.normal(means[component], variances, (n_samples, n_features))
    
    mix_idx = np.random.randint(0, n_samples, n_mix)
    labels[mix_idx] = np.random.randint(0, n_components, n_mix)
    data[mix_idx] = np.random.normal(means[labels[mix_idx]], variances, (n_mix, n_features))
    
    return data, labels

def transfigure(blocks):
    if isinstance(blocks, str):
        if blocks == 'hsblocks':
            return HSBLOCKS
        if blocks == 'vsblocks':
            return VSBLOCKS
        if blocks == 'chblocks':
            return CHBLOCKS
        if blocks == 'ggblocks':
            return GGBLOCKS
    
        raise NotImplementedError(f'Block palette "{blocks}" not supported.')
    
    return blocks

def interpret(blocks, share_components=False):  
    n_samples, n_components = [], []

    for idx, b in enumerate(blocks):
        blocks[idx] = b = transfigure(b)
        n_samples.append(b.shape[0])
        n_components.append(np.unique(b).shape[0])

    n_samples = sum(n_samples)
    n_components = max(n_components) if share_components else sum(n_components)

    return blocks, n_samples, n_components

def make_data(n_samples, n_components, n_features, n_equivocal=0, n_redundant=0, sample_counts=None, means=None, variances=None, mean_scale=1., variance_scale=1., mix=0.):
    n_mix = partition(int(mix*n_samples), n_components)
    sample_counts, means, variances = designate(n_samples, n_components, n_features, n_equivocal, n_redundant, sample_counts, means, variances, mean_scale, variance_scale)
    data = np.zeros((n_samples, n_features), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int32)
    j = 0

    for i in range(n_components):
        k = j + sample_counts[i]
        data[j:k], labels[j:k] = sample(sample_counts[i], n_components, n_features, means, variances, i, n_mix[i])
        j += sample_counts[i]

    return data, labels

def make_blocks(blocks, n_features, n_equivocal=0, n_redundant=0, means=None, variances=None, mean_scale=1., variance_scale=1., block_size=5, wiggle=0., mix=0.):
    grid = np.repeat(np.repeat(blocks, block_size, 0), block_size, 1)
    n_samples = grid.shape[0]*grid.shape[1]
    _, sample_counts = np.unique(grid, return_counts=True)
    n_components = sample_counts.shape[0]
    data, labels = make_data(n_samples, n_components, n_features, n_equivocal, n_redundant, sample_counts, means, variances, mean_scale, variance_scale, mix)
    data = np.hstack([np.zeros((data.shape[0], 2)), data])
    j = 0

    for i in range(n_components):
        locs = np.stack(np.where(grid.T == i)).T
        locs[:, -1] = -locs[:, -1] + grid.shape[0]
        data[j:j + locs.shape[0], :2] = locs
        j += locs.shape[0]

    data[:, :2] += 2*wiggle*np.random.rand(*data[:, :2].shape) - wiggle
    
    return data, labels

def make_dataset(blocks, n_features, n_equivocal=0, n_redundant=0, means=None, variances=None, mean_scale=1., variance_scale=1., block_size=5, wiggle=0., mix=0., img=None, offset=0, tensor=False):
    if means is not None:
        means = means[offset:]
    
    blocks = transfigure(blocks)
    data, labels = make_blocks(blocks, n_features, n_equivocal, n_redundant, means, variances, mean_scale, variance_scale, block_size, wiggle, mix)
    labels += offset

    if img is not None:
        data = np.hstack([img*np.ones((data.shape[0], 1)), data])

    if tensor:
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.int32)
    return data, labels

def make_datasets(blocks, n_features, n_equivocal=0, n_redundant=0, means=None, variances=None, mean_scale=1., variance_scale=1., block_size=5, wiggle=0., mix=0., share_components=True, tensor=False):
    n_datasets = len(blocks) if isinstance(blocks, list) else 1
    blocks, block_size, wiggle, mix = util.itemize(n_datasets, blocks, block_size, wiggle, mix)
    blocks, _, n_components = interpret(blocks, share_components)
    _, means, variances = designate(_, n_components, n_features, n_equivocal, n_redundant, _, means, variances, mean_scale, variance_scale)
    datasets = []

    for idx, (b, s, w, m) in enumerate(zip(blocks, block_size, wiggle, mix)):
        offset = 0 if share_components or len(datasets) == 0 else max(datasets[-1][1]) + 1
        datasets.append(make_dataset(b, n_features, n_equivocal, n_redundant, means, variances, mean_scale, variance_scale, s, w, m, idx, offset))

    data, labels = zip(*datasets)
    data = np.concatenate(data)
    labels = np.concatenate(labels)

    if tensor:
        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.int32)
    return data, labels
