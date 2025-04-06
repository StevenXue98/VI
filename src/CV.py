"""
Functions for computing control variate noise reduced gradients
Translated from Autograd to JAX
"""
import jax
import jax.numpy as jnp
from jax import grad, jvp, vjp, jit


def construct_cv_grads(vbobj, lam, eps,
                       elbo_gsamps=None,
                       method="hessian"):
    """ main method to construct reduced variance reparameterization gradients
        using a variety of methods.

    Methods:
        - "mc"           : full monte carlo estimator
        - "hessian"      : uses full hessian information
        - "hessian_diag" : uses only hessian diag information
        - "hvp_with_loo_diag_approx" : uses other samples to approximate
        - "hvp_with_mc_variance"     :
    """
    # unpack variational parameters
    assert eps.ndim == 2, "epsilon needs to be nsamps x D"
    ns, D = eps.shape
    m_lam, s_lam = lam[:D], jnp.exp(lam[D:])

    # generate samples if necessary
    if elbo_gsamps is None:
        elbo_gsamps = elbo_grad_samps_mat(vbobj, lam, eps)

    if method == "mc":
        # full monte carlo --- this is a No-op
        return elbo_gsamps

    elif method == "hessian":
        """ full hessian approximation
        """
        # full hessian, including diagonal
        gmu = vbobj.glnpdf(m_lam)
        H = vbobj.hlnpdf(m_lam)
        Hdiag = jnp.diag(H)

        # construct normal approx samples of data term
        dLdz = gmu + jnp.dot(H, (s_lam * eps).T).T
        dLds = dLdz * eps * s_lam + 1.
        elbo_gsamps_tilde = jnp.column_stack([dLdz, dLds])

        # characterize the mean of the dLds component (and z comp)
        dLds_mu = (Hdiag * s_lam + 1 / s_lam) * s_lam
        gsamps_tilde_mean = jnp.concatenate([gmu, dLds_mu])

        # subtract mean to compute control variate
        elbo_gsamps_cv = elbo_gsamps - (elbo_gsamps_tilde - gsamps_tilde_mean)
        return elbo_gsamps_cv

    elif method == "hessian_diag":
        """ use only hessian diagonal for RV model """
        gmu = vbobj.glnpdf(m_lam)
        H = vbobj.hlnpdf(m_lam)
        Hdiag = jnp.diag(H)

        # construct normal approx samples of data term
        dLdz = gmu + Hdiag * s_lam * eps
        dLds = (dLdz * eps + 1 / s_lam[None, :]) * s_lam
        elbo_gsamps_tilde = jnp.column_stack([dLdz, dLds])

        # construct mean
        dLds_mu = (Hdiag * s_lam + 1 / s_lam) * s_lam
        gsamps_tilde_mean = jnp.concatenate([gmu, dLds_mu])
        elbo_gsamps_cv = elbo_gsamps - (elbo_gsamps_tilde - gsamps_tilde_mean)
        return elbo_gsamps_cv

    elif method == "hvp_with_loo_diag_approx":
        """ use other samples to estimate a per-sample diagonal
        expectation
        """
        assert ns > 1, "loo approximations require more than 1 sample"
        # compute hessian vector products using JAX's hvp implementation
        gmu = vbobj.glnpdf(m_lam)

        # In JAX, we create an hvp maker function using jvp and grad
        def hvp_lam(x, v):
            """Hessian-vector product for vbobj.lnpdf at point x with vector v"""
            return jax.jvp(jax.grad(vbobj.lnpdf), (x,), (v,))[1]

        hvps = jnp.array([hvp_lam(m_lam, s_lam * e) for e in eps])

        # construct normal approx samples of data term
        dLdz = gmu + hvps
        dLds = dLdz * (eps * s_lam) + 1

        # compute Leave One Out approximate diagonal (per-sample mean of dLds)
        Hdiag_sum = jnp.sum(eps * hvps, axis=0)
        Hdiag_s = (Hdiag_sum[None, :] - eps * hvps) / float(ns - 1)
        dLds_mu = (Hdiag_s + 1 / s_lam[None, :]) * s_lam

        # Modified in-place operation for JAX (creating new arrays instead)
        elbo_gsamps_updated = elbo_gsamps.at[:, :D].set(elbo_gsamps[:, :D] - hvps)
        elbo_gsamps_updated = elbo_gsamps_updated.at[:, D:].set(elbo_gsamps[:, D:] - (dLds - dLds_mu))
        return elbo_gsamps_updated

    elif method == "hvp_with_loo_direct_approx":
        # compute hessian vector products and save them for both parts
        assert ns > 1, "loo approximations require more than 1 sample"
        gmu = vbobj.glnpdf(m_lam)

        # Define hvp function using JAX
        def hvp_func(m, v):
            return jax.jvp(jax.grad(vbobj.lnpdf), (m,), (v,))[1]

        hvps = jnp.array([hvp_func(m_lam, s_lam * e) for e in eps])

        # construct normal approx samples of data term
        dLdz = gmu + hvps
        dLds = (dLdz * eps + 1 / s_lam[None, :]) * s_lam
        elbo_gsamps_tilde = jnp.column_stack([dLdz, dLds])

        # compute Leave One Out approximate diagonal (per-sample mean of dLds)
        dLds_sum = jnp.sum(dLds, axis=0)
        dLds_mu = (dLds_sum[None, :] - dLds) / float(ns - 1)

        # compute gsamps_cv - mean(gsamps_cv), and finally the var reduced
        elbo_gsamps_tilde_centered = jnp.column_stack([dLdz - gmu, dLds - dLds_mu])
        elbo_gsamps_cv = elbo_gsamps - elbo_gsamps_tilde_centered
        return elbo_gsamps_cv

    elif method == "hvp_with_mc_variance":
        # Define hvp function using JAX
        def hvp_lam(m, v):
            return jax.jvp(jax.grad(vbobj.lnpdf), (m,), (v,))[1]

        hvps = jnp.array([hvp_lam(m_lam, s_lam * e) for e in eps])

        # In JAX, we create a new array instead of modifying in place
        elbo_gsamps_updated = elbo_gsamps.at[:, :D].set(elbo_gsamps[:, :D] - hvps)
        return elbo_gsamps_updated

    # not implemented
    raise NotImplementedError(f"{method} not implemented")


def elbo_grad_samps_mat(vbobj, lam, eps):
    """ function to compute grad g = [g_m, g_lns]
            g_m   = dELBO / dm
            g_lns = dELBO / d ln-sigma]
    from some base randomness, eps

    Returns:
        - [dELBO/dm, dELBO/dlns] as a Nsamps x D array
    """
    assert eps.ndim == 2, "epsilon must be Nsamps x D"
    D = vbobj.D

    # generate samples
    m_lam, s_lam = lam[:D], jnp.exp(lam[D:])
    zs = m_lam[None, :] + s_lam[None, :] * eps

    # generate dElbo/dm (which happens to be dElbo / dz)
    dL_dz = vbobj.glnpdf(zs)
    dL_dm = dL_dz

    # generate dElbo/d sigma, convert via d sigma/d ln-sigma
    dL_ds = dL_dz * eps + 1 / s_lam
    dL_dlns = dL_ds * s_lam
    return jnp.column_stack([dL_dm, dL_dlns])