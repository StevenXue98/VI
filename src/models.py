import jax
import jax.numpy as jnp
from functools import partial
from codebase.models import frisk, uci, nn


def set_lnpdf(model="baseball", dset="boston"):
    """
    Create log probability density functions for different models

    Args:
        model: Type of model ("baseball", "frisk", "normal", "bnn")
        dset: Dataset name for BNN model

    Returns:
        tuple: (log_pdf_function, dimensions, model_name)
    """
    if model == "baseball":
        # This assumes baseball module has been adapted to JAX
        return lambda x: jnp.squeeze(baseball.lnpdf_flat(x, 0)), baseball.D, model

    if model == "frisk":
        # Assuming frisk module has been adapted to JAX
        lnpdf, unpack, num_params, frisk_df, param_names = \
            frisk.make_model_funs(crime=2., precinct_type=1)
        return lnpdf, num_params, model

    if model == "normal":
        D, r = 10, 2

        # Use JAX PRNG for reproducibility
        key = jax.random.PRNGKey(0)
        key, key1, key2 = jax.random.split(key, 3)

        mu0 = jnp.zeros(D)
        C_true = jax.random.normal(key1, (D, r)) * 2.
        v_true = jax.random.normal(key2, (D,))
        Sigma_true = jnp.dot(C_true, C_true.T) + jnp.diag(jnp.exp(v_true))

        print(Sigma_true)  # Note: In JAX, this won't print values during tracing

        # Define multivariate normal log PDF
        def mvn_logpdf(x, mean, cov):
            """Multivariate normal log PDF"""
            d = mean.shape[0]
            centered = x - mean
            # Numerically stable way to compute log(det(cov))
            sign, logdet = jnp.linalg.slogdet(cov)
            log_norm = -0.5 * (d * jnp.log(2 * jnp.pi) + logdet)
            # Solve system instead of explicit inverse
            solved = jnp.linalg.solve(cov, centered)
            return log_norm - 0.5 * jnp.sum(centered * solved)

        # Vectorize the log PDF
        vmvn_logpdf = jax.vmap(mvn_logpdf, in_axes=(0, None, None))

        # Create a fixed covariance MVN log PDF
        def fixed_cov_mvn_logpdf(x, mean=mu0):
            if x.ndim == 1:
                return mvn_logpdf(x, mean, Sigma_true)
            else:
                return vmvn_logpdf(x, mean, Sigma_true)

        return fixed_cov_mvn_logpdf, D, model

    if model == "bnn":
        # Assuming uci and nn modules have been adapted to JAX
        (Xtrain, Ytrain), (Xtest, Ytest) = \
            uci.load_dataset(dset, split_seed=0)

        lnpdf, predict, loglike, parser, (std_X, ustd_X), (std_Y, ustd_Y) = \
            nn.make_nn_regression_funs(Xtrain[:100], Ytrain[:100],
                                       layer_sizes=None, obs_variance=None)

        # Vectorize lnpdf using vmap
        lnpdf_vec = jax.vmap(lnpdf)

        # Wrapper to handle at least 2d input
        def lnpdf_wrapper(ths):
            ths = jnp.atleast_2d(ths)
            return lnpdf_vec(ths)

        return lnpdf_wrapper, parser.N, "-".join([model, dset])