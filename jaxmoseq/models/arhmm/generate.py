import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import cho_factor
import numpy as np

na = jnp.newaxis

from jaxmoseq.utils import pad_affine


def generate_initial_state(seed, Ab, Q):
    """
    Generate initial states for the ARHMM.

    Sample the initial discrete state from a uniform distribution
    and sample the initial latent trajectory from a standard normal.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    pi : jax array of shape (num_states, num_states)
        Transition matrix.
    Ab : jax array of shape (num_states, latent_dim, latent_dim*nlags+1)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.

    Returns
    -------
    z : int
        Initial discrete state.
    xlags : jax array of shape (nlags,latent_dim)
        Initial ``nlags`` states of the continuous state trajectory.
    seed : jax.random.PRNGKey
        Updated random seed.
    """
    z = jr.choice(seed, jnp.arange(Ab.shape[0]))
    latent_dim = Ab.shape[1]
    nlags = (Ab.shape[2] - 1) // latent_dim
    xlags = jr.normal(seed, (nlags, latent_dim))
    seed = jr.split(seed)[1]
    return z, xlags, seed


def generate_next_state(seed, z, xlags, Ab, Q, pi):
    """
    Generate the next states of an ARHMM.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    z : int
        Current discrete state.
    xlags : jax array of shape (nlags,latent_dim)
        ``nlags`` of continuous state trajectory.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    pi : jax array of shape (num_states, num_states)
        Transition matrix.

    Returns
    -------
    z : int
        Next state.
    xlags : jax array of shape (nlags, latent_dim)
        ``nlags`` of the state trajectory after appending
        the next continuous state.
    """
    # sample the next state
    z = jr.choice(seed, jnp.arange(pi.shape[0]), p=pi[z])

    # sample the next latent trajectory
    mu = jnp.dot(Ab[z], pad_affine(xlags.flatten()))
    x = jr.multivariate_normal(seed, mu, Q[z])
    xlags = jnp.concatenate([xlags[1:], x[na]], axis=0)

    # update the seed
    seed = jr.split(seed)[1]
    return z, xlags, seed


def generate_next_state_fast(seed, z, xlags, Ab, L, pi, sigma):
    """
    Generate the next states of an ARHMM, using cholesky
    factors and precomputed gaussian random variables to
    speed up sampling.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    z : int
        Current discrete state.
    xlags : jax array of shape (nlags, latent_dim)
        ``nlags`` of continuous state trajectory.
    Ab : jax array of shape (num_states, latent_dim, ar_dim)
        Autoregressive transforms.
    L : jax array of shape (num_states, latent_dim, latent_dim)
        Cholesky factors for autoregressive noise covariances.
    pi : jax array of shape (num_states, num_states)
        Transition matrix.
    sigma: jax array of shape (latent_dim,)
        Sample from a standard multivariate normal.

    Returns
    -------
    z : int
        Next state.
    xlags : jax array of shape (nlags, latent_dim)
        ``nlags`` of the state trajectory after appending
        the next continuous state.
    """
    # sample the next state
    z = jr.choice(seed, jnp.arange(pi.shape[0]), p=pi[z])

    # sample the next latent trajectory
    mu = jnp.dot(Ab[z], pad_affine(xlags.flatten()))
    x = mu + jnp.dot(L[z], sigma)
    xlags = jnp.concatenate([xlags[1:], x[na]], axis=0)

    # update the seed
    seed = jr.split(seed)[1]
    return z, xlags, seed


def generate_states(seed, pi, Ab, Q, n_steps, init_state=None):
    """
    Generate a sequence of states from an ARHMM.

    Parameters
    ----------
    seed : jax.random.PRNGKey
        Random seed.
    pi : jax array of shape (num_states, num_states)
        Transition matrix.
    Ab : jax array of shape (num_states, latent_dim, latent_dim*nlags+1)
        Autoregressive transforms.
    Q : jax array of shape (num_states, latent_dim, latent_dim)
        Autoregressive noise covariances.
    n_steps : int
        Number of steps to generate.
    init_states : tuple of jax arrays with shapes ((,), (nlags,latent_dim)), optional
        Initial discrete state and ``nlags`` of continuous trajectory.

    Returns
    -------
    zs : jax array of shape (n_steps,)
        Discrete states.
    xs : jax array of shape (n_steps,latent_dim)
        Continuous states.
    """
    # initialize the states
    if init_state is None:
        z, xlags, seed = generate_initial_state(seed, Ab, Q)
    else:
        z, xlags = init_state

    # precompute cholesky factors and random samples
    L = cho_factor(Q, lower=True)[0]
    sigmas = jr.normal(seed, (n_steps, Q.shape[-1]))

    # generate the states using jax.lax.scan
    def _generate_next_state(carry, sigma):
        z, xlags, seed = carry
        z, xlags, seed = generate_next_state_fast(seed, z, xlags, Ab, L, pi, sigma)
        return (z, xlags, seed), (z, xlags)

    carry = (z, xlags, seed)
    _, (zs, xs) = jax.lax.scan(_generate_next_state, carry, sigmas)

    return zs, xs[:, -1]
