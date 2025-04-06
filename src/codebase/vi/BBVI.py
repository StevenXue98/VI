import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, jit, vmap, jvp, vjp, random
from functools import partial


class BBVI(object):
    def __init__(self, lnpdf, D, glnpdf=None, lnpdf_is_vectorized=False):
        """ Black Box Variational Inference using stochastic gradients

        Args:
            lnpdf: Log probability density function
            D: Dimensionality of the parameter space
            glnpdf: Gradient of lnpdf (optional, will be computed if not provided)
            lnpdf_is_vectorized: Whether lnpdf can handle batched inputs
        """
        self.D = D

        if lnpdf_is_vectorized:
            self.lnpdf = lnpdf
            if glnpdf is None:
                # JAX doesn't have elementwise_grad like autograd
                # Using vmap(grad) to achieve the same effect
                self.glnpdf = jax.vmap(jax.grad(lnpdf))
        else:
            # Create vectorized versions
            self.glnpdf_single = jax.grad(lnpdf)

            # In JAX, we use vmap for vectorization
            @jax.vmap
            def vec_lnpdf(z):
                return lnpdf(z)

            @jax.vmap
            def vec_glnpdf(z):
                return self.glnpdf_single(z)

            self.lnpdf = lambda z: vec_lnpdf(jnp.atleast_2d(z))
            self.glnpdf = lambda z: vec_glnpdf(jnp.atleast_2d(z))

        # We don't need gglnpdf in JAX implementation as it's not used in the provided methods
        # But including for completeness
        self.gglnpdf = jax.vmap(jax.grad(lambda x: self.glnpdf(jnp.expand_dims(x, 0))[0]))

        # Hessian
        # JAX approach: use jacfwd(jacrev()) for efficient Hessian computation
        self.hlnpdf = lambda x: jax.jacfwd(jax.jacrev(lnpdf))(x)

        # Hessian-vector product
        # In JAX, we use jvp(grad()) to compute HVP efficiently
        def hvp(x, v):
            """Hessian-vector product: H(x) @ v"""
            return jax.jvp(jax.grad(lnpdf), (x,), (v,))[1]

        self.hvplnpdf = hvp

        # Function that creates HVP functions for a specific point
        def hvp_maker(x):
            """Returns a function that computes H(x) @ v for any v"""

            def hvp_at_x(v):
                return jax.jvp(jax.grad(lnpdf), (x,), (v,))[1]

            return hvp_at_x

        self.hvplnpdf_maker = hvp_maker

    #################################################
    # BBVI exposes these gradient functions         #
    #################################################
    def elbo_grad_mc(self, lam, t, n_samps=1, eps=None):
        """ monte carlo approximation of the *negative* ELBO
            eps: seed randomness (could be uniform, could be Gaussian)
        """
        raise NotImplementedError

    def elbo_grad_delta_approx(self, lam, t):
        """ delta method approximation of the *negative* ELBO """
        raise NotImplementedError

    def elbo_grad_fixed_mixture_approx(self, lam, t, rho=.5):
        """ combine a sample w/ the elbo grad mean """
        raise NotImplementedError

    def elbo_grad_adaptive_mixture_approx(self, lam, t):
        raise NotImplementedError

    def elbo_mc(self, lam, n_samps=100, full_monte_carlo=False):
        """ approximate the ELBO with samples """
        raise NotImplementedError

    def true_elbo(self, lam, t):
        """ approximates the ELBO with 20k samples """
        raise NotImplementedError

    def sample_z(self, lam, n_samps=1, eps=None):
        """ Sample from variational distribution q(z|λ) """
        raise NotImplementedError

    def nat_grad(self, lam, standard_grad):
        """ converts standard gradient into a natural gradient at parameter
        value lam """
        raise NotImplementedError


# JAX equivalent of mvn_diag_logpdf and mvn_diag_entropy
def mvn_diag_logpdf(x, mu, log_sigma):
    """Multivariate normal log pdf with diagonal covariance matrix"""
    D = mu.shape[-1]
    sigma = jnp.exp(log_sigma)
    norm_const = -0.5 * D * jnp.log(2 * jnp.pi) - jnp.sum(log_sigma)
    return norm_const - 0.5 * jnp.sum(((x - mu) / sigma) ** 2, axis=-1)


def mvn_diag_entropy(log_sigma):
    """Entropy of a multivariate normal with diagonal covariance matrix"""
    D = log_sigma.shape[-1]
    return 0.5 * D * (1.0 + jnp.log(2 * jnp.pi)) + jnp.sum(log_sigma)


class DiagMvnBBVI(BBVI):
    def __init__(self, lnpdf, D, glnpdf=None, lnpdf_is_vectorized=False):
        """
        Implements MCVI --- exposes elbo gradient and sampling methods.
        This class breaks the gradient down into parts
        dg/dz = dlnpdf(z)/dz * dz/dlam - dlnq(z)/dz * dz/dlam - dlnq(z)/dlam
        Parameterizes with mean and log-std! (not variance!)
            lam = [mean, log-std]
        """
        # base class sets up the gradient function organization
        super(DiagMvnBBVI, self).__init__(lnpdf, D, glnpdf, lnpdf_is_vectorized)

        # we note that the second two terms, with probability one,
        # create the vector [0, 0, 0, ..., 0, 1., 1., ..., 1.]
        self.mask = jnp.concatenate([jnp.zeros(D), jnp.ones(D)])
        self.num_variational_params = 2 * D
        self.D = D

        # Add dlnp method if it doesn't exist in the base class
        if not hasattr(self, 'dlnp'):
            # This will need to be implemented based on your specific model
            # Placeholder implementation:
            self.dlnp = lambda lam, eps: self._compute_dlnp(lam, eps)

    def _compute_dlnp(self, lam, eps):
        """
        Compute dlnp/dlam for a batch of epsilon values
        This is a placeholder - implementation depends on your model
        """
        # Example implementation - replace with your actual computation
        D = self.D
        m_lam, s_lam = lam[:D], jnp.exp(lam[D:])
        zs = m_lam + s_lam * eps

        # Compute gradient of log posterior with respect to z
        dL_dz = self.glnpdf(zs)

        # Compute gradient components
        dL_dm = dL_dz
        dL_ds = dL_dz * eps * s_lam

        return jnp.column_stack([dL_dm, dL_ds])

    #####################################################################
    # Methods for various types of gradients of the ELBO                #
    #    -- that can be plugged into FilteredOptimization routines      #
    #####################################################################
    def elbo_grad_mc(self, lam, t, n_samps=1, eps=None, key=None):
        """ monte carlo approximation of the *negative* ELBO """
        if eps is None:
            if key is None:
                # In JAX, we need to explicitly pass a PRNG key
                key = random.PRNGKey(0)  # Default seed
            eps = random.normal(key, shape=(n_samps, self.D))
        return -1. * jnp.mean(self.dlnp(lam, eps) + self.mask, axis=0)

    def nat_grad(self, lam, standard_gradient):
        """Convert standard gradient to natural gradient"""
        finv = 1. / self.fisher_info(lam)
        return finv * standard_gradient

    def fisher_info(self, lam):
        """
        Fisher information matrix for the variational distribution
        For diagonal MVN with mean and log-std parameterization
        """
        # Placeholder - you'll need to provide the actual implementation
        # For diagonal MVN, this is typically a diagonal matrix
        D = self.D
        return jnp.concatenate([jnp.ones(D), 2 * jnp.ones(D)])

    #############################
    # ELBO objective functions  #
    #############################
    def elbo_mc(self, lam, n_samps=100, full_monte_carlo=False, key=None):
        """ approximate the ELBO with samples """
        D = self.D

        if key is None:
            key = random.PRNGKey(0)  # Default seed

        zs = self.sample_z(lam, n_samps=n_samps, key=key)

        if full_monte_carlo:
            elbo_vals = self.lnpdf(zs) - mvn_diag_logpdf(zs, lam[:D], lam[D:])
        else:
            elbo_vals = self.lnpdf(zs) + mvn_diag_entropy(lam[D:])

        return jnp.mean(elbo_vals)

    def true_elbo(self, lam, t, key=None):
        """ approximates the ELBO with 20k samples """
        return self.elbo_mc(lam, n_samps=20000, key=key)

    def sample_z(self, lam, n_samps=1, eps=None, key=None):
        """ sample from the variational distribution """
        D = self.D
        assert len(lam) == 2 * D, "bad parameter length"

        if eps is None:
            if key is None:
                key = random.PRNGKey(0)  # Default seed
            eps = random.normal(key, shape=(n_samps, D))

        z = jnp.exp(lam[D:]) * eps + lam[None, :D]
        return z