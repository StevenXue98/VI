"""
Implementation of the hierarchical poisson glm model, with a precinct-specific
term, an ethnicity specific term, and an offset term.

The data are tuples of (ethnicity, precinct, num_stops, total_arrests), where
the count variables num_stops and total_arrests refer to the number of stops
and total arrests of an ethnicity in the specified precinct over a period of
15 months.  The rate we are measuring is the rate of stops-per-arrest
for certain ethnicities in different precincts.

    Y_ep       = num stops of ethnicity e in precinct p
    N_ep       = num arests of e in p
    log lam_ep = alpha_e + beta_p + mu + log(N_ep * 15/12)  #yearly correction term
    Y_ep       ~ Pois(lam_ep)
"""

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import gammaln
import pandas as pd
import os


# Process the dataset
def process_dataset():
    data_dir = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(data_dir, 'data/frisk/frisk_with_noise.dat'),
                     skiprows=6, delim_whitespace=True)

    # compute proportion black in precinct, black = 1
    # first aggregate by precinct/ethnicity, and sum over populations
    popdf = df[['pop', 'precinct', 'eth']].groupby(['precinct', 'eth'])['pop'].apply(sum)

    # Convert to JAX-friendly computation
    percent_black = jnp.array([
        popdf[i][1] / float(popdf[i].sum()) for i in range(1, 76)
    ])

    # Use pandas cut for binning
    precinct_type = pd.cut(percent_black, [0, .1, .4, 1.])
    df['precinct_type'] = precinct_type.codes[df.precinct.values - 1]

    return df


# Load the dataset
df = process_dataset()


def make_model_funs(crime=1., precinct_type=0):
    """ crime: 1=violent, 2=weapons, 3=property, 4=drug
        eth  : 1=black, 2 = hispanic, 3=white
        precincts: 1-75
        precinct_type = (0, .1], (.1, .4], (.4, 1.]
    """
    # subselect crime/precinct, set up design matrix
    sdf = df[(df['crime'] == crime) & (df['precinct_type'] == precinct_type)]

    # make dummies for precincts, etc
    def one_hot(x, k):
        return jnp.array(x[:, None] == jnp.arange(k)[None, :], dtype=int)

    precincts = jnp.sort(jnp.array(pd.unique(sdf['precinct'])))

    # Convert pandas data to JAX arrays
    Xprecinct = one_hot(jnp.array(sdf['precinct']), 76)[:, precincts]
    Xeth = one_hot(jnp.array(sdf['eth']), 4)[:, 1:-1]
    yep = jnp.array(sdf['stops'].values)
    lnep = jnp.log(jnp.array(sdf['past.arrests'].values)) + jnp.log(15. / 12)

    num_eth = Xeth.shape[1]
    num_precinct = Xprecinct.shape[1]

    # unpack a flat param vector
    aslice = slice(0, num_eth)
    bslice = slice(num_eth, num_eth + num_precinct)
    mslice = slice(bslice.stop, bslice.stop + 1)
    lnsa_slice = slice(mslice.stop, mslice.stop + 1)
    lnsb_slice = slice(lnsa_slice.stop, lnsa_slice.stop + 1)
    num_params = lnsb_slice.stop

    def pname(s, stub):
        return [f'{stub}_{i}' for i in range(s.stop - s.start)]

    param_names = [pname(s, stub)
                   for s, stub in zip([aslice, bslice, mslice, lnsa_slice, lnsb_slice],
                                      ['alpha', 'beta', 'mu', 'lnsigma_a', 'lnsigma_b'])]
    param_names = [s for pn in param_names for s in pn]

    def unpack(th):
        """ unpack vectorized lndf """
        th = jnp.atleast_2d(th)
        alpha_eth, beta_prec, mu, lnsigma_alpha, lnsigma_beta = \
            th[:, aslice], th[:, bslice], th[:, mslice], \
                th[:, lnsa_slice], th[:, lnsb_slice]
        return alpha_eth, beta_prec, mu, lnsigma_alpha, lnsigma_beta

    hyper_lnstd = jnp.array([[jnp.log(10.)]])

    def lnpdf(th):
        # params
        alpha, beta, mu, lns_alpha, lns_beta = unpack(th)

        # priors
        ll_alpha = normal_lnpdf(alpha, 0, lns_alpha)
        ll_beta = normal_lnpdf(beta, 0, lns_beta)
        ll_mu = normal_lnpdf(mu, 0, hyper_lnstd)
        ll_salpha = normal_lnpdf(jnp.exp(lns_alpha), 0, hyper_lnstd)
        ll_sbeta = normal_lnpdf(jnp.exp(lns_beta), 0, hyper_lnstd)
        logprior = ll_alpha + ll_beta + ll_mu + ll_salpha + ll_sbeta

        # likelihood
        lnlam = (mu + lnep[None, :]) + \
                jnp.dot(alpha, Xeth.T) + jnp.dot(beta, Xprecinct.T)
        loglike = jnp.sum(lnpoiss(yep, lnlam), axis=1)
        return jax.lax.cond(
            th.shape[0] == 1,
            lambda x: x[0],
            lambda x: x,
            loglike + logprior
        )

    return lnpdf, unpack, num_params, sdf, param_names


def lnpoiss(y, lnlam):
    """ log likelihood of poisson """
    return y * lnlam - jnp.exp(lnlam) - gammaln(y + 1)


def normal_lnpdf(x, mean, ln_std):
    """Log PDF of normal distribution with parameters in log space"""
    x = jnp.atleast_2d(x)
    D = x.shape[1]

    # Handle case where ln_std is not D-dimensional
    dcoef = jnp.where(ln_std.shape[1] != D, D, 1.)

    # Compute log PDF components
    qterm = -0.5 * jnp.sum((x - mean) ** 2 / jnp.exp(2. * ln_std), axis=1)
    coef = -0.5 * D * jnp.log(2. * jnp.pi) - dcoef * jnp.sum(ln_std, axis=1)

    return qterm + coef