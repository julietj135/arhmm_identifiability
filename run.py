import yaml
from data.preprocess import *
from utils import _set_parallel_flag, save_hdf5, update_hypparams
from viz import *
import jax

import jax.numpy as jnp
from jaxmoseq.models import arhmm, slds
import jax.random as jr
from functools import partial

from jaxmoseq.utils.kalman import kalman_sample
from jaxmoseq.utils.distributions import sample_vonmises_fisher
from jaxmoseq.utils.utils import batch, jax_io, device_put_as_scalar
import tensorflow_probability.substrates.jax.distributions as tfd
from sklearn.decomposition import PCA

import tqdm
import h5py
import os
from textwrap import fill
import warnings

na = jnp.newaxis


def get_pca(Y_flat,
            mask,
            **kwargs):

    print("completing pca...")
    
    Y_flatter = Y_flat[mask > 0]
    print(Y_flatter.shape)

    N = Y_flatter.shape[0]
    N_sample = min(1000000, N)
    sample = np.random.choice(N, N_sample, replace=False)
    Y_sample = np.array(Y_flatter)[sample]
    
    # in actual keypoint moseq, they fit a sample of the data based on confidence and minimum number of PCA frames
    pca = PCA().fit(Y_flatter)

    cs = np.cumsum(pca.explained_variance_ratio_)
    latentdim = 0
    if cs[-1] < 0.9:
        latentdim = len(cs)
        print(
            f"All components together only explain {cs[-1]*100}% of variance."
        )
    else:
        latentdim = (cs>0.9).nonzero()[0].min()+1
        print(
            f">={0.9*100}% of variance exlained by {latentdim} components."
        )
    if latentdim == 1:
        latentdim = 2

    return Y_flatter, pca, latentdim

def init_model(
    coordinates,
    trans_hypparams,
    ar_hypparams,
    obs_hypparams,
    cen_hypparams,
    seed=jr.PRNGKey(0),
    whiten=True,
    **kwargs,
):
    """
    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    """
    # get mask for data
    Y, mask, metadata = batch(coordinates)
    
    print(Y.shape)

    # transform/flatten data
    dims = Y.shape[:-2]
    k, d = Y.shape[-2:]
    # DID NOT "EMBED" Y, so dimension is k*d instead of (k-1)*d
    Y_flat = Y.reshape(*dims, k * d)
    
    # pca
    Y_flatter, pca, latent_dim = get_pca(Y_flat, mask)
    
    model = {}
    
    # initialize seed
    if isinstance(seed, int):
        seed = jr.PRNGKey(seed)
    model["seed"] = seed
    
    # initialize noise prior
    noise_prior = 1.0 # may need to change
    model["noise_prior"] = noise_prior
    
    # initialize hyperparameters
    ar_hypparams['latent_dim'] = latent_dim
    hypparams = slds.init_hyperparams(trans_hypparams, ar_hypparams, obs_hypparams)
    hypparams["cen_hypparams"] = cen_hypparams.copy()
    model["hypparams"] = hypparams
    
    # initialize params
    params = arhmm.init_params(seed, **hypparams)
    C = jnp.array(pca.components_[:latent_dim])
    d = jnp.array(pca.mean_)
    Cd = jnp.hstack([C.T, d[:, na]])
    if whiten:
        latents_flat = jax_io(pca.transform)(Y_flatter)[:, :latent_dim]
        cov = jnp.cov(latents_flat.T)
        W = jnp.linalg.cholesky(cov)
        C = W.T @ C
    params["Cd"] = jnp.hstack([C.T, d[:, na]]) 
    params["sigmasq"] = jnp.ones(Y_flat.shape[-1])
    model["params"] = params
    
    # initialize states
    obs_hypparams = hypparams["obs_hypparams"]
    x = slds.init_continuous_stateseqs(Y_flat, params["Cd"])
    states = arhmm.init_states(seed, x, mask, params)
    sqerr = slds.compute_squared_error(Y_flat, x, Cd)
    
    states["x"] = x
    states["s"] = slds.resample_scales_from_sqerr(seed, sqerr, **params, s_0=noise_prior, **obs_hypparams) 
    model["states"] = states
    
    data = jax.device_put({"mask": mask, "Y": Y})
    
    return model, data

def fit_model(project_dir,
              model_name,
              model,
              seed, 
              data,
              kappa,
              num_iters,
              ar_only=True,
              save_every_n_iters=25,
              jitter=1e-3,
              parallel_message_passing=None,
              **kwargs): 
    
    savedir = os.path.join(project_dir, model_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    print(fill(f"Outputs will be saved to {savedir}"))

    start_iter = 0
    checkpoint_path = os.path.join(savedir, "checkpoint.h5")
    if not os.path.exists(checkpoint_path):
        save_hdf5(
            checkpoint_path,
            {
                "model_snapshots": {f"{start_iter}": model},
                "data": data,
            },
        )
    else:  # delete model snapshots later than start_iter
        with h5py.File(checkpoint_path, "a") as f:
            for k in list(f["model_snapshots"].keys()):
                if int(k) > start_iter:
                    del f["model_snapshots"][k]
    
    parallel_message_passing = _set_parallel_flag(parallel_message_passing)
    model = device_put_as_scalar(model)
    
    with tqdm.trange(start_iter, num_iters + 1, ncols=72) as pbar:
        seed = model["seed"]
        for iteration in pbar:
            
            model = slds.resample_model(
                        data,
                        **model,
                        ar_only=ar_only)
            # model = arhmm.resample_model(data, **model)
            
            # params = model["params"].copy()
            # states = model["states"].copy()
                        
            # if ar_only:
            #     continue
            
            # states["x"] = slds.resample_continuous_stateseqs(
            #                         seed,
            #                         **data,
            #                         **states,
            #                         **params,
            #                         jitter=jitter,
            #                         parallel_message_passing=parallel_message_passing
            #                     )
            
            # sqerr = compute_squared_error(**data,
            #                         **states,
            #                         **params)
            # states["s"] = slds.resample_scales_from_sqerr(seed,
            #                         **data,
            #                         **states,
            #                         **params,
            #                         s_0=noise_prior,
            #                         **hypparams["obs_hypparams"])
            
            if save_every_n_iters is not None and iteration > start_iter:
                if iteration == num_iters or (
                    save_every_n_iters > 0 and iteration % save_every_n_iters == 0
                ):
                    save_hdf5(checkpoint_path, model, f"model_snapshots/{iteration}")
                    plot_progress(
                        model,
                        data,
                        checkpoint_path,
                        iteration,
                        project_dir,
                        model_name=model_name,
                        )
    
    if ar_only:
        return model, model_name
    else:
        # model = {"seed": seed,
        #         "states": states,
        #         "params": params,
        #         "hypparams": hypparams,
        #         "noise_prior": noise_prior,}
        
        return model, model_name

# def resample_model(
#     data,
#     seed,
#     states,
#     params,
#     hypparams,
#     noise_prior,
#     ar_only=False,
#     states_only=False,
#     resample_global_noise_scale=False,
#     resample_local_noise_scale=True,
#     verbose=False,
#     jitter=1e-3,
#     parallel_message_passing=False,
#     **kwargs
# ):
#     model = arhmm.resample_model(
#         data, seed, states, params, hypparams, states_only, verbose=verbose
#     )
#     if ar_only:
#         model["noise_prior"] = noise_prior
#         return model

#     seed = model["seed"]
#     params = model["params"].copy()
#     states = model["states"].copy()
    
#     states["x"] = slds.resample_continuous_stateseqs(
#         seed,
#         Y,
#         mask,
#         z,
#         s,
#         Ab,
#         Q,
#         Cd,
#         sigmasq,
#         jitter=jitter,
#         parallel_message_passing=parallel_message_passing,
#     )
    
#     sqerr = compute_squared_error(Y, x, v, h, Cd, mask)
#     params["sigmasq"] = slds.resample_obs_variance_from_sqerr(
#         seed, sqerr, mask, s, nu_sigma, sigmasq_0
#     )
    
#     states["s"] = slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)
    
#     return {
#         "seed": seed,
#         "states": states,
#         "params": params,
#         "hypparams": hypparams,
#         "noise_prior": noise_prior,
#     }
    
    
@jax.jit
def compute_squared_error(Y, x, v, h, Cd, mask=None, **kwargs):
    """
    Computes the squared error between model predicted
    and true observations.

    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    x : jax array of shape (..., latent_dim)
        Latent trajectories.
    v : jax array of shape (..., d)
        Centroid positions.
    h : jax array
        Heading angles.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    mask : jax array, optional
        Binary indicator for valid frames.

    Returns
    ------
    sqerr : jax array of shape (..., k)
        Squared error between model predicted and
        true observations.
    """
    Y_est = estimate_coordinates(x, v, h, Cd)
    sqerr = ((Y - Y_est) ** 2).sum(-1)
    if mask is not None:
        sqerr = mask[..., na] * sqerr
    return sqerr


def obs_log_prob(Y, x, v, h, s, Cd, sigmasq, **kwargs):
    """
    Calculate the log probability of keypoint coordinates at each
    time-step, given continuous latent trajectories, centroids, heading
    angles, noise scales, and observation parameters.

    Parameters
    ----------
    Y : jax array of shape (..., k, d)
        Keypoint observations.
    x : jax array of shape (..., latent_dim)
        Latent trajectories.
    v : jax array of shape (..., d)
        Centroid positions.
    h : jax array
        Heading angles.
    s : jax array of shape (..., k)
        Noise scales.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    log_pY: jax array of shape (..., k)
        Log probability of `Y`.
    """
    Y_bar = estimate_coordinates(x, v, h, Cd)
    sigma = jnp.broadcast_to(jnp.sqrt(s * sigmasq)[..., na], Y.shape)
    return tfd.MultivariateNormalDiag(Y_bar, sigma).log_prob(Y)


@jax.jit
def log_joint_likelihood(
    Y, mask, x, v, h, s, z, pi, Ab, Q, Cd, sigmasq, sigmasq_loc, s_0, nu_s, **kwargs
):
    """
    Calculate the total log probability for each latent state.

    Parameters
    ----------
    Y : jax array of shape (..., T, k, d)
        Keypoint observations.
    mask : jax array of shape (..., T)
        Binary indicator for valid frames.
    x : jax array of shape (..., T, latent_dim)
        Latent trajectories.
    v : jax array of shape (..., T, d)
        Centroid positions.
    h : jax array of shape (..., T)
        Heading angles.
    s : jax array of shape (..., T, k)
        Noise scales.
    z : jax_array of shape (..., T - n_lags)
        Discrete state sequences.
    pi : jax_array of shape (num_states, num_states)
        Transition probabilities.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    Cd : jax array of shape ((k - 1) * d, latent_dim + 1)
        Observation transform.
    sigmasq : jax_array of shape k
        Unscaled noise.
    sigmasq_loc : float
        Assumed variance in centroid displacements.
    s_0 : scalar or jax array broadcastable to `Y`
        Prior on noise scale.
    nu_s : int
        Chi-squared degrees of freedom in noise prior.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    -------
    ll: dict
        Dictionary mapping the name of each state variable to
        its total log probability.
    """
    ll = arhmm.log_joint_likelihood(x, mask, z, pi, Ab, Q)

    log_pY = obs_log_prob(Y, x, v, h, s, Cd, sigmasq)
    log_ps = slds.scale_log_prob(s, s_0, nu_s)
    log_pv = location_log_prob(v, sigmasq_loc)

    ll["Y"] = (log_pY * mask[..., na]).sum()
    ll["s"] = (log_ps * mask[..., na]).sum()
    ll["v"] = (log_pv * mask[..., 1:]).sum()
    return ll


def model_likelihood(data, states, params, hypparams, noise_prior, **kwargs):
    """
    Convenience class that invokes `log_joint_likelihood`.

    Parameters
    ----------
    data : dict
        Data dictionary containing the observations and mask.
    states : dict
        State values for each latent variable.
    params : dict
        Values for each model parameter.
    hypparams : dict
        Values for each group of hyperparameters.
    noise_prior : scalar or jax array broadcastable to `s`
        Prior on noise scale.
    **kwargs : dict
        Overflow, for convenience.

    Returns
    ------
    ll : dict
        Dictionary mapping state variable name to its
        total log probability.
    """
    return log_joint_likelihood(
        **data,
        **states,
        **params,
        **hypparams["obs_hypparams"],
        **hypparams["cen_hypparams"],
        s_0=noise_prior
    )
