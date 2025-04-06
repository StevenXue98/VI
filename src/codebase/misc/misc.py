import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import jax.numpy.linalg as jnla


def sigmoid(a):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + jnp.exp(-a))


def logit(a):
    """Logit function (inverse of sigmoid)"""
    return jnp.log(a) - jnp.log(1 - a)


def mvn_diag_logpdf(x, mean, log_std):
    """
    Log probability density function of a multivariate normal distribution
    with diagonal covariance matrix.

    Args:
        x: Samples, shape (n_samples, D)
        mean: Mean vector, shape (D,)
        log_std: Log of standard deviation, shape (D,)

    Returns:
        Log probability density values, shape (n_samples,)
    """
    D = len(mean)
    qterm = -0.5 * jnp.sum((x - mean) ** 2 / jnp.exp(2.0 * log_std), axis=1)
    coef = -0.5 * D * jnp.log(2.0 * jnp.pi) - jnp.sum(log_std)
    return qterm + coef


def mvn_diag_logpdf_grad(x, mean, log_std):
    """
    Gradient of the log PDF of a diagonal MVN w.r.t. the input

    This can be implemented using JAX's auto-differentiation:
    """
    return jax.grad(lambda x_: jnp.sum(mvn_diag_logpdf(x_, mean, log_std)))(x)


def mvn_diag_entropy(log_std):
    """
    Entropy of a multivariate normal with diagonal covariance matrix

    Args:
        log_std: Log of standard deviation, shape (D,)

    Returns:
        Entropy value
    """
    D = len(log_std)
    return 0.5 * (D * jnp.log(2 * jnp.pi * jnp.e) + jnp.sum(2 * log_std))


def mvn_logpdf(x, mean, icholSigma):
    """
    Log probability density function of a multivariate normal distribution

    Args:
        x: Samples, shape (n_samples, D)
        mean: Mean vector, shape (D,)
        icholSigma: Inverse of Cholesky factor of covariance matrix

    Returns:
        Log probability density values
    """
    D = len(mean)
    coef = -0.5 * D * jnp.log(2.0 * jnp.pi)
    dterm = jnp.sum(jnp.log(jnp.diag(icholSigma)))
    white = jnp.dot(jnp.atleast_2d(x) - mean, icholSigma.T)
    qterm = -0.5 * jnp.sum(white ** 2, axis=1)
    ll = coef + dterm + qterm

    # Handle scalar case
    return jnp.where(ll.shape[0] == 1, ll[0], ll)


def mvn_fisher_info(params):
    """
    Returns the Fisher information matrix (diagonal) for a multivariate
    normal distribution with params = [mu, ln sigma]

    Args:
        params: Parameters [mean, log_std], shape (2*D,)

    Returns:
        Fisher information diagonal, shape (2*D,)
    """
    D = len(params) // 2
    mean, log_std = params[:D], params[D:]
    return jnp.concatenate([jnp.exp(-2.0 * log_std), 2.0 * jnp.ones(D)])


def kl_mvn(m0, S0, m1, S1):
    """
    KL divergence between two normal distributions

    Args:
        m0: Mean of first distribution
        S0: Covariance matrix of first distribution
        m1: Mean of second distribution
        S1: Covariance matrix of second distribution

    Returns:
        KL divergence KL(N(m0,S0) || N(m1,S1))
    """
    # .5 log det (Sig1 Sig0^-1)
    # + .5 tr( Sig1^-1 * ((mu_0 - mu_1)(mu_0 - mu_1)^T + Sig0 - Sig1) )
    det_term = 0.5 * jnp.log(jnla.det(jnla.solve(S0, S1).T))
    S1inv = jnla.inv(S1)
    diff = m0 - m1

    # Use batched operations for the trace term
    outers = jnp.einsum("id,ie->ide", diff, diff) + S0 - S1
    tr_term = 0.5 * jnp.einsum("de,ide->i", S1inv, outers)

    return det_term + tr_term


def kl_mvn_diag(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed. Divergence is expressed in nats.

    Args:
        m0: Mean of first distribution, shape (batch, D)
        S0: Variance of first distribution, shape (D,)
        m1: Mean of second distribution, shape (batch, D)
        S1: Variance of second distribution, shape (D,)

    Returns:
        KL divergence KL(N(m0,S0) || N(m1,S1)), shape (batch,)
    """
    # Store inv diag covariance of S1 and diff between means
    N = m0.shape[1]
    iS1 = 1.0 / S1
    diff = m1 - m0

    # KL is made of three terms
    tr_term = jnp.sum(iS1 * S0)
    det_term = jnp.sum(jnp.log(S1)) - jnp.sum(jnp.log(S0))
    quad_term = jnp.sum((diff * diff) * iS1, axis=1)

    return 0.5 * (tr_term + det_term + quad_term - N)


def gamma_lnpdf(x, shape, rate):
    """
    Log probability density function of a Gamma distribution

    Args:
        x: Input value(s)
        shape: Shape parameter
        rate: Rate parameter

    Returns:
        Log probability density
    """
    coef = shape * jnp.log(rate) - gammaln(shape)
    dterm = (shape - 1.0) * jnp.log(x) - rate * x
    return coef + dterm


def make_fixed_cov_mvn_logpdf(Sigma):
    """
    Create a log-PDF function for a multivariate normal with fixed covariance

    Args:
        Sigma: Covariance matrix

    Returns:
        Function that computes log-PDF given samples and mean
    """
    icholSigma = jnla.inv(jnla.cholesky(Sigma))
    return lambda x, mean: mvn_logpdf(x, mean, icholSigma)


def unpack_params(params):
    """
    Unpack parameters into mean and log_std

    Args:
        params: Parameters [mean, log_std], shape (2*D,)

    Returns:
        mean, log_std: Split parameters
    """
    mean, log_std = jnp.split(params, 2)
    return mean, log_std


def unconstrained_to_simplex(rhos):
    """
    Convert unconstrained parameters to a simplex (probabilities that sum to 1)

    Args:
        rhos: Unconstrained parameters, shape (K-1,)

    Returns:
        Simplex parameters, shape (K,)
    """
    rhosf = jnp.concatenate([rhos, jnp.zeros(1)])
    pis = jnp.exp(rhosf) / jnp.sum(jnp.exp(rhosf))
    return pis


def simplex_to_unconstrained(pis):
    """
    Convert simplex parameters to unconstrained parameters

    Args:
        pis: Simplex parameters, shape (K,)

    Returns:
        Unconstrained parameters, shape (K-1,)
    """
    lnpis = jnp.log(pis)
    return (lnpis - lnpis[-1])[:-1]