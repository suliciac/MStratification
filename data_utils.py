from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
from abc import ABC
import copy
import torch


def normalize(im, quantile=(1, 99)):
    qmax = np.percentile(im, quantile[1])
    qmin = np.percentile(im, quantile[0])
    im_norm = 1 * (im - qmin) / (qmax - qmin)

    return im_norm


class ZipSet(Dataset):
    """Zips a list of sets and returns a tuple with the ith element of each given set.
    :param sets: list or tuple of sets.
    :Example:
    #>>> zipped_set = ZipSet([ListSet([1,2,3]), ListSet([1,2,3])])
    #>>> print([x for x in zipped_set])
    [(1, 1), (2, 2), (3, 3)]
    """

    def __init__(self, sets):
        assert all([isinstance(set, Dataset) for set in sets])
        self.sets = sets

    def __len__(self):
        return min(len(s) for s in self.sets)

    def __getitem__(self, index):
        return tuple([dataset[index] for dataset in self.sets])


class ListSet(Dataset):
    """Creates a data set from a given list.
    :param items: list of elements to return.
    :Example:
    #>>> myset = ListSet([1, 2, 3])
    #>>> print(myset[0])
    1
    """

    def __init__(self, items):
        assert isinstance(items, list)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def make_generator(sett, batch_size, shuffle, num_workers=4):
    """Makes a generator from a given set that returns elements with the specified batch size.
    :param torch.utils.data.Dataset sett: the data _set_.
    :param batch_size: batch size of generated items.
    :param shuffle: whether to shuffle the generator elements each epoch (SHOULD be True for training generators).
    :param num_workers: (default: 4) number of parallel workers getting items.
    :Example:
    #>>> myset = ListSet(list(range(10)))
    #>>> mygen = make_generator(myset, batch_size=2, shuffle=False)
    #>>> print([x for x in mygen])
    [tensor([0, 1]), tensor([2, 3]), tensor([4, 5]), tensor([6, 7]), tensor([8, 9])]
    """

    return DataLoader(sett, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class PatchSetPad(Dataset):
    """
    Creates a torch dataset that returns patches extracted from images either from predefined extraction centers or
    using a predefined patch sampling strategy.
    :param List[np.ndarray] images: list of images with shape (CH, X, Y, Z)
    :param PatchSampling sampling: An object of type PatchSampling defining the patch sampling strategy.
    :param normalize: one of ``'none'``, ``'patch'``, ``'image'``.
    :param dtype: the desired output data type (default: torch.float)
    :param List[List[tuple]] centers: (optional) a list containing a list of centers for each provided image.
        If provided it ignores the given sampling and directly uses the centers to extract the patches.
    """

    def __init__(self, images, patch_shape, sampling, normalize, dtype=torch.float, centers=None, transf_func=None):
        assert all([img.ndim == 4 for img in images]), 'Images must be numpy ndarrays with dimensions (C, X, Y, Z)'
        assert len(patch_shape) == 3, 'len({}) != 3'.format(patch_shape)
        assert normalize in ['none', 'patch', 'image']
        if centers is None: assert isinstance(sampling, PatchSampling)
        if centers is not None: assert len(centers) == len(images)

        self.images, self.dtype = images, dtype

        # Build all instructions according to centers and normalize
        self.instructions = []
        images_centers = sampling.sample_centers(images, patch_shape) if centers is None else centers

        for image_idx, image_centers in enumerate(images_centers):
            # Compute normalize function for this image's patches

            if normalize == 'image': # Update norm_func with the statistics of the image
                means = np.mean(self.images[image_idx], axis=(1, 2, 3), keepdims=True, dtype=np.float64)
                stds = np.std(self.images[image_idx], axis=(1, 2, 3), keepdims=True, dtype=np.float64)

                # BY PROVIDING MEANS AND STDS AS ARGUMENTS WITH DEFAULT VALUES, WE MAKE A COPY of their values inside
                # norm_func. If not, the means and stds would be of the last stored value (last image's statistics)
                # leading to incorrect results
                norm_func = lambda x, m=means, s=stds: (x - m) / s
            else:
                # Identity function (normalize == 'none')
                norm_func = lambda x: x

            # Generate instructions
            self.instructions += [PatchInstruction(
                image_idx, center=center, shape=patch_shape, normalize_function=norm_func, augment_function=transf_func) for center in image_centers]

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, index):
        instr = self.instructions[index]
        x_patch = copy.deepcopy(self.images[instr.idx][_get_patch_slice(instr.center, instr.shape)])

        if x_patch[0].shape != instr.shape:
            des_shape = (x_patch.shape[0],) + instr.shape
            x_patch_z = np.zeros(des_shape)
            cent_sh = tuple(ti/2 for ti in instr.shape)
            pos_sls = _get_patch_slice(cent_sh, x_patch[0].shape)
            x_patch_z[pos_sls] = x_patch
            x_patch = x_patch_z

        if instr.normalize_function is not None:
            x_patch = instr.normalize_function(x_patch)

        if instr.augment_function is not None:
            x_patch = instr.augment_function[1](x_patch)

        return torch.tensor(np.ascontiguousarray(x_patch), dtype=self.dtype)


class PatchSampling(ABC):
    """Abstract Base Class that defines a patch sampling strategy.

    A new sampling can be made by inheriting from this class and overriding the abstract method ``sample_centers``.

    .. py:function:: sample_centers(self, images, patch_shape)

        :param List[np.ndarray] images: list of images with shape (CH, X, Y, Z)
        :param tuple patch_shape: patch shape as (X, Y, Z)
        :return: (List[List[tuple]]) A list containing a list of centers (tuples as (x, y, z)) for each image in images.
    """

    def sample_centers(self, images, patch_shape):
        pass


class PatchInstruction:
    def __init__(self, idx, center, shape, normalize_function=None, augment_function=None):
        self.idx = idx
        self.center = center
        self.shape = shape
        self.normalize_function = normalize_function
        self.augment_function = augment_function


def _get_patch_slice(center, patch_shape):
    """
    :param center: a tuple or list of (x,y,z) tuples
    :param tuple patch_shape: (x,y,z) tuple with arr dimensions
    :return: a tuple (channel_slice, x_slice, y_slice, z_slice) or a list of them
    """
    if not isinstance(center, list): center = [center]
    span = [[int(np.ceil(dim / 2.0)), int(np.floor(dim / 2.0))] for dim in patch_shape]
    patch_slices = \
        [(slice(None),) + tuple(slice(cdim - s[0], cdim + s[1]) for cdim, s in zip(c, span)) for c in center]
    return patch_slices if len(patch_slices) > 1 else patch_slices[0]
