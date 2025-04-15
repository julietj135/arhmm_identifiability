import yaml
import os
import pandas as pd
from textwrap import fill
import numpy as np
import h5py
import tqdm
import jax.numpy as jnp
import jax
import warnings

def _set_parallel_flag(parallel_message_passing):
    if parallel_message_passing == "force":
        parallel_message_passing = True
    elif parallel_message_passing is None:
        parallel_message_passing = jax.default_backend() != "cpu"
    elif parallel_message_passing and jax.default_backend() == "cpu":
        warnings.warn(
            fill(
                "Setting parallel_message_passing to True when JAX is CPU-bound can "
                "result in long jit times without speed increase for calculations. "
                '(To suppress this message, set parallel_message_passing="force")'
            )
        )
    return parallel_message_passing

def load_checkpoint(project_dir=None, model_name=None, path=None, iteration=None):
    """Load data and model snapshot from a saved checkpoint.

    The checkpoint path can be specified directly via `path` or else it is
    assumed to be `{project_dir}/{model_name}/checkpoint.h5`.

    Parameters
    ----------
    project_dir: str, default=None
        Project directory; used in conjunction with `model_name` to determine the
        checkpoint path if `path` is not specified.

    model_name: str, default=None
        Model name; used in conjunction with `project_dir` to determine the
        checkpoint path if `path` is not specified.

    path: str, default=None
        Checkpoint path; if not specified, the checkpoint path is set to
        `{project_dir}/{model_name}/checkpoint.h5`.

    iteration: int, default=None
        Determines which model snapshot to load. If None, the last snapshot is
        loaded.

    Returns
    -------
    model: dict
        Model dictionary containing states, parameters, hyperparameters,
        noise prior, and random seed.

    data: dict
        Data dictionary containing observations, confidences, mask and
        associated metadata (see :py:func:`keypoint_moseq.util.format_data`).

    metadata: tuple (keys, bounds)
        Recordings and start/end frames for the data (see
        :py:func:`keypoint_moseq.util.format_data`).

    iteration: int
        Iteration of model fitting corresponding to the loaded snapshot.
    """
    path = _get_path(project_dir, model_name, path, "checkpoint.h5")

    with h5py.File(path, "r") as f:
        saved_iterations = np.sort([int(i) for i in f["model_snapshots"]])

    if iteration is None:
        iteration = saved_iterations[-1]
    else:
        assert iteration in saved_iterations, fill(
            f"No snapshot found for iteration {iteration}. "
            f"Available iterations are {saved_iterations}"
        )

    model = load_hdf5(path, f"model_snapshots/{iteration}")
    # metadata = load_hdf5(path, "metadata")
    data = load_hdf5(path, "data")
    # return model, data, metadata, iteration
    return model, data, iteration

def _loadtree_hdf5(leaf):
    """Recursively load a pytree from an h5 file group."""
    if isinstance(leaf, h5py.Dataset):
        data = np.array(leaf[()])
        if h5py.check_dtype(vlen=data.dtype) == str:
            data = np.array([item.decode("utf-8") for item in data])
        elif data.dtype.kind == "S":
            data = data.item().decode("utf-8")
        elif data.shape == ():
            data = data.item()
        return data
    else:
        leaf_type = leaf.attrs["type"]
        values = map(_loadtree_hdf5, leaf.values())
        if leaf_type == "dict":
            return dict(zip(leaf.keys(), values))
        elif leaf_type == "list":
            return list(values)
        elif leaf_type == "tuple":
            return tuple(values)
        else:
            raise ValueError(f"Unrecognized type {leaf_type}")

def load_hdf5(filepath, datapath=None):
    """Load a dict of pytrees from an hdf5 file.

    Parameters
    ----------
    filepath: str
        Path of the hdf5 file to load.

    datapath: str, default=None
        Path within the hdf5 file to load the data from. If None, the data is
        loaded from the root of the hdf5 file.

    Returns
    -------
    save_dict: dict
        Dictionary where the values are pytrees, i.e. recursive collections of
        tuples, lists, dicts, and numpy arrays.
    """
    with h5py.File(filepath, "r") as f:
        if datapath is None:
            return {k: _loadtree_hdf5(f[k]) for k in f}
        else:
            return _loadtree_hdf5(f[datapath])

def _get_path(project_dir, model_name, path, filename, pathname_for_error_msg="path"):
    if path is None:
        assert project_dir is not None and model_name is not None, fill(
            f"`model_name` and `project_dir` are required if `{pathname_for_error_msg}` is None."
        )
        path = os.path.join(project_dir, model_name, filename)
    return path

def _savetree_hdf5(tree, group, name):
    """Recursively save a pytree to an h5 file group."""
    if name in group:
        del group[name]
    if isinstance(tree, np.ndarray):
        if tree.dtype.kind == "U":
            dt = h5py.special_dtype(vlen=str)
            group.create_dataset(name, data=tree.astype(object), dtype=dt)
        else:
            group.create_dataset(name, data=tree)
    elif isinstance(tree, (float, int, str)):
        group.create_dataset(name, data=tree)
    else:
        subgroup = group.create_group(name)
        subgroup.attrs["type"] = type(tree).__name__

        if isinstance(tree, (tuple, list)):
            for k, subtree in enumerate(tree):
                _savetree_hdf5(subtree, subgroup, f"arr{k}")
        elif isinstance(tree, dict):
            for k, subtree in tree.items():
                _savetree_hdf5(subtree, subgroup, k)
        else:
            raise ValueError(f"Unrecognized type {type(tree)}")
        
def save_hdf5(filepath, save_dict, datapath=None):
    """Save a dict of pytrees to an hdf5 file. The leaves of the pytrees must
    be numpy arrays, scalars, or strings.

    Parameters
    ----------
    filepath: str
        Path of the hdf5 file to create.

    save_dict: dict
        Dictionary where the values are pytrees, i.e. recursive collections of
        tuples, lists, dicts, and numpy arrays.

    datapath: str, default=None
        Path within the hdf5 file to save the data. If None, the data is saved
        at the root of the hdf5 file.
    """
    with h5py.File(filepath, "a") as f:
        if datapath is not None:
            _savetree_hdf5(jax.device_get(save_dict), f, datapath)
        else:
            for k, tree in save_dict.items():
                _savetree_hdf5(jax.device_get(tree), f, k)

def update_hypparams(model_dict, **kwargs):
    """Edit the hyperparameters of a model.

    Hyperparameters are stored as a nested dictionary in the `hypparams` key of
    the model dictionary. This function allows the user to update the
    hyperparameters of a model by passing in keyword arguments with the same
    name as the hyperparameter. The hyperparameter will be updated if it is a
    scalar value.

    Parameters
    ----------
    model_dict : dict
        Model dictionary.

    kwargs : dict
        Keyword arguments mapping hyperparameter names to new values.

    Returns
    -------
    model_dict : dict
        Model dictionary with updated hyperparameters.
    """
    assert "hypparams" in model_dict, fill(
        "The inputted model/checkpoint does not contain any hyperparams"
    )

    not_updated = list(kwargs.keys())

    for hypparms_group in model_dict["hypparams"]:
        for k, v in kwargs.items():
            if k in model_dict["hypparams"][hypparms_group]:
                old_value = model_dict["hypparams"][hypparms_group][k]
                if not np.isscalar(old_value):
                    print(
                        fill(
                            f"{k} cannot be updated since it is not a scalar hyperparam"
                        )
                    )
                else:
                    if not isinstance(v, type(old_value)):
                        warnings.warn(
                            f"'{k}' with {type(v)} will be cast to {type(old_value)}"
                        )

                    model_dict["hypparams"][hypparms_group][k] = type(old_value)(v)
                    not_updated.remove(k)

    if len(not_updated) > 0:
        warnings.warn(fill(f"The following hypparams were not found {not_updated}"))

    return model_dict