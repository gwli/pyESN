import numpy as np
import statsmodels.base._penalties as smpen
import statsmodels.base.model as base_model


"""
A collection of smooth penalty functions.

Penalties on vectors take a vector argument and return a scalar
penalty.  The gradient of the penalty is a vector with the same shape
as the input value.

Penalties on covariance matrices take two arguments: the matrix and
its inverse, both in unpacked (square) form.  The returned penalty is
a scalar, and the gradient is returned as a vector that contains the
gradient with respect to the free elements in the lower triangle of
the covariance matrix.

All penalties are subtracted from the log-likelihood, so greater
penalty values correspond to a greater degree of penalization.

The penaties should be smooth so that they can be subtracted from log
likelihood functions and optimized using standard methods (i.e. L1
penalties do not belong here).
"""

class Penalty(object):
    """
    A class for representing a scalar-value penalty.

    Parameters
    ----------
    wts : array-like
        A vector of weights that determines the weight of the penalty
        for each parameter.


    Notes
    -----
    The class has a member called `alpha` that scales the weights.
    """

    def __init__(self, wts):
        self.wts = wts
        self.alpha = 1.

    def func(self, params):
        """
        A penalty function on a vector of parameters.

        Parameters
        ----------
        params : array-like
            A vector of parameters.

        Returns
        -------
        A scalar penaty value; greater values imply greater
        penalization.
        """
        raise NotImplementedError

    def grad(self, params):
        """
        The gradient of a penalty function.

        Parameters
        ----------
        params : array-like
            A vector of parameters

        Returns
        -------
        The gradient of the penalty with respect to each element in
        `params`.
        """
        raise NotImplementedError


class L2(Penalty):
    """
    The L2 (ridge) penalty.
    """

    def __init__(self, wts=None):
        if wts is None:
            self.wts = 1.
        else:
            self.wts = wts
        self.alpha = 1.

    def func(self, params):
        return np.sum(self.wts * self.alpha * params**2)

    def grad(self, params):
        return 2 * self.wts * self.alpha * params


class SCAD(Penalty):
    """
    The SCAD penalty of Fan and Li
    """

    def __init__(self, tau, c=3.7, wts=None):
        if wts is None:
            self.weights = 1.
        else:
            self.weights = wts
        self.tau = tau
        self.c = c

    def func(self, params):

        # 3 segments in absolute value
        tau = self.tau
        p_abs = np.atleast_1d(np.abs(params))
        res = np.empty(p_abs.shape)#, p_abs.dtype)
        res.fill(np.nan)
        mask1 = p_abs < tau
        mask3 = p_abs >= self.c * tau
        res[mask1] = tau * p_abs[mask1]
        mask2 = ~mask1 & ~mask3
        p_abs2 = p_abs[mask2]
        #tmp = (tau * p_abs2**2 - 2 * self.c * p_abs2 + tau**2)
        tmp = (p_abs2**2 - 2 * self.c * tau * p_abs2 + tau**2)
        res[mask2] = -tmp / (2 * (self.c - 1))
        #res[mask3] = (self.c + 1) / 2. * p_abs[mask3]**2
        res[mask3] = (self.c + 1) * tau**2 / 2.

        return (self.weights * res).sum(0)

    def grad(self, params):

        # 3 segments in absolute value
        tau = self.tau
        p = np.atleast_1d(params)
        p_abs = np.abs(p)
        p_sign = np.sign(p)
        res = np.empty(p_abs.shape)#, p_abs.dtype)
        res.fill(np.nan)

        mask1 = p_abs < tau
        mask3 = p_abs >= self.c * tau
        mask2 = ~mask1 & ~mask3
        res[mask1] = p_sign[mask1] * tau
        #tmp = p_sign[mask2] * (tau * p_abs[mask2] - self.c)
        tmp = p_sign[mask2] * (p_abs[mask2] - self.c * tau)
        res[mask2] = -tmp / (self.c - 1)
        res[mask3] = 0#(self.c + 1)

        return self.weights * res


    def deriv2(self, params):
        """Second derivative of function

        Warning: Not checked, possible problems in multivariate case
        This returns scalar or vector in same shape as params, not a square Hessian.

        """

        # 3 segments in absolute value
        tau = self.tau
        p = np.atleast_1d(params)
        p_abs = np.abs(p)
        p_sign = np.sign(p)
        res = np.zeros(p_abs.shape)#, p_abs.dtype)

        mask1 = p_abs < tau
        mask3 = p_abs >= self.c * tau
        mask2 = ~mask1 & ~mask3

        res[mask2] = -p_sign[mask2] / (self.c - 1)

        return self.weights * res


class SCADSmoothed(SCAD):
    """
    The SCAD penalty of Fan and Li, quadratically smoothed around zero

    This follows Fan and Li 2001 equation (3.7).

    Parameterization follows Boo, Johnson, Li and Tan 2011


    TODO: Use delegation instead of subclassing, so smoothing can be added to all
    penalty classes.

    """

    def __init__(self, tau, c=3.7, c0=None, wts=None, restriction=None):
        if wts is None:
            self.weights = 1.
        else:
            self.weights = wts
        self.alpha = 1.
        self.tau = tau
        self.c = c
        self.c0 = c0 if c0 is not None else tau * 0.1
        if self.c0 > tau:
            raise ValueError('c0 cannot be larger than tau')

        # get coefficients for quadratic approximation
        c0 = self.c0
        deriv_c0 = super(SCADSmoothed, self).grad(c0)
        value_c0 = super(SCADSmoothed, self).func(c0)
        self.aq1 = value_c0 - 0.5 * deriv_c0 * c0
        self.aq2 = 0.5 * deriv_c0 / c0
        self.restriction = restriction


    def func(self, params):
        # TODO: `and np.size(params) > 1` is hack for llnull, need better solution
        if self.restriction is not None and np.size(params) > 1:
            params = self.restriction.dot(params)
        # need to temporarily override weights for call to super
        # Note: we have the same problem with `restriction`
        weights = self.weights
        self.weights = 1.
        value = super(SCADSmoothed, self).func(params[None, ...])
        self.weights = weights

        #change the segment corrsponding to quadratic approximation
        p_abs = np.atleast_1d(np.abs(params))
        mask = p_abs < self.c0
        p_abs_masked = p_abs[mask]
        value[mask] = self.aq1 + self.aq2 * p_abs_masked**2

        return (self.weights * value).sum(0)


    def grad(self, params):
        if self.restriction is not None and np.size(params) > 1:
            params = self.restriction.dot(params)
        # need to temporarily override weights for call to super
        weights = self.weights
        self.weights = 1.
        value = super(SCADSmoothed, self).grad(params)
        self.weights = weights

        #change the segment corrsponding to quadratic approximation
        p = np.atleast_1d(params)
        mask = np.abs(p) < self.c0
        value[mask] = 2 * self.aq2 * p[mask]

        if self.restriction is not None and np.size(params) > 1:
            return weights * value.dot(self.restriction)
        else:
            return weights * value


    def deriv2(self, params):
        if self.restriction is not None and np.size(params) > 1:
            params = self.restriction.dot(params)
        # need to temporarily override weights for call to super
        weights = self.weights
        self.weights = 1.
        value = super(SCADSmoothed, self).deriv2(params)
        self.weights = weights

        #change the segment corrsponding to quadratic approximation
        p = np.atleast_1d(params)
        #p_abs = np.abs(p)
        mask = np.abs(p) < self.c0
        #p_abs_masked = p_abs[mask]
        value[mask] = 2 * self.aq2

        if self.restriction is not None and np.size(params) > 1:
            # note: super returns 1d array for diag, i.e. hessian_diag
            return (self.restriction.T * value).dot(self.restriction)
        else:
            return value


class L2SCADSmoothed(SCADSmoothed):
    def func(self, params):
        # TODO: `and np.size(params) > 1` is hack for llnull, need better solution
        if self.restriction is not None and np.size(params) > 1:
            params = self.restriction.dot(params)
        # need to temporarily override weights for call to super
        # Note: we have the same problem with `restriction`
        weights = self.weights
        self.weights = 1.
        value = super(SCADSmoothed, self).func(params[None, ...])
        self.weights = weights

        #change the segment corrsponding to quadratic approximation
        p_abs = np.atleast_1d(np.abs(params))
        mask = p_abs < self.c0
        p_abs_masked = p_abs[mask]
        value[mask] = self.aq1 + self.aq2 * p_abs_masked**2

        #change to L2
        return self.weights * np.linalg.norm(value)

class CovariancePenalty(object):

    def __init__(self, wt):
        self.wt = wt

    def func(self, mat, mat_inv):
        """
        Parameters
        ----------
        mat : square matrix
            The matrix to be penalized.
        mat_inv : square matrix
            The inverse of `mat`.

        Returns
        -------
        A scalar penalty value
        """
        raise NotImplementedError

    def grad(self, mat, mat_inv):
        """
        Parameters
        ----------
        mat : square matrix
            The matrix to be penalized.
        mat_inv : square matrix
            The inverse of `mat`.

        Returns
        -------
        A vector containing the gradient of the penalty
        with respect to each element in the lower triangle
        of `mat`.
        """
        raise NotImplementedError




class PenalizedMixin(object):
    """Mixin class for Maximum Penalized Likelihood


    TODO: missing **kwds or explicit keywords

    TODO: do we really need `pen_weight` keyword in likelihood methods?

    """

    def __init__(self, *args, **kwds):
        super(PenalizedMixin, self).__init__(*args, **kwds)

        penal = kwds.pop('penal', None)
        # I keep the following instead of adding default in pop for future changes
        if penal is None:
            # TODO: switch to unpenalized by default
            self.penal = SCADSmoothed(0.1, c0=0.0001)
        else:
            self.penal = penal

        # TODO: define pen_weight as average pen_weight? i.e. per observation
        # I would have prefered len(self.endog) * kwds.get('pen_weight', 1)
        # or use pen_weight_factor in signature
        #self.pen_weight =  kwds.get('pen_weight', len(self.endog))
        self.pen_weight =  np.average(self.endog)

        self._init_keys.extend(['penal', 'pen_weight'])



    def loglike(self, params, pen_weight=None):
       if pen_weight is None:
            pen_weight = self.pen_weight

       llf = super(PenalizedMixin, self).loglike(params)
       if pen_weight != 0:
           llf -= pen_weight * self.penal.func(params)

       return llf


    def loglikeobs(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        llf = super(PenalizedMixin, self).loglikeobs(params)
        nobs_llf = float(llf.shape[0])

        if pen_weight != 0:
            llf -= pen_weight / nobs_llf * self.penal.func(params)

        return llf


    def score(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        sc = super(PenalizedMixin, self).score(params)
        if pen_weight != 0:
            sc -= pen_weight * self.penal.grad(params)

        return sc


    def scoreobs(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        sc = super(PenalizedMixin, self).scoreobs(params)
        nobs_sc = float(sc.shape[0])
        if pen_weight != 0:
            sc -= pen_weight / nobs_sc  * self.penal.grad(params)

        return sc


    def hessian_(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight
            loglike = self.loglike
        else:
            loglike = lambda p: self.loglike(p, pen_weight=pen_weight)

        from statsmodels.tools.numdiff import approx_hess
        return approx_hess(params, loglike)


    def hessian(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight

        hess = super(PenalizedMixin, self).hessian(params)
        if pen_weight != 0:
            h = self.penal.deriv2(params)
            if h.ndim == 1:
                hess -= np.diag(pen_weight * h)
            else:
                hess -= pen_weight * h

        return hess


    def fit(self, method=None, trim=None, **kwds):
        # If method is None, then we choose a default method ourselves

        # TODO: temporary hack, need extra fit kwds
        # we need to rule out fit methods in a model that will not work with
        # penalization
        if hasattr(self, 'family'):  # assume this identifies GLM
            kwds.update({'max_start_irls' : 0})

        # currently we use `bfgs` by default
        if method is None:
            #method = 'bfgs'
            method = 'newton'


        if trim is None:
            trim = False  # see below infinite recursion in `fit_constrained

        res = super(PenalizedMixin, self).fit(method=method, **kwds)

        if trim is False:
            # note boolean check for "is False" not evaluates to False
            return res
        else:
            # TODO: make it penal function dependent
            # temporary standin, only works for Poisson and GLM,
            # and is computationally inefficient
            drop_index = np.nonzero(np.abs(res.params) < 1e-4) [0]
            keep_index = np.nonzero(np.abs(res.params) > 1e-4) [0]
            rmat = np.eye(len(res.params))[drop_index]

            # calling fit_constrained raise
            # "RuntimeError: maximum recursion depth exceeded in __instancecheck__"
            # fit_constrained is calling fit, recursive endless loop
            if drop_index.any():
                # todo : trim kwyword doesn't work, why not?
                #res_aux = self.fit_constrained(rmat, trim=False)
                res_aux = self._fit_zeros(keep_index, **kwds)
                return res_aux
            else:
                return res

class FixedGenericLikelihoodModel(base_model.GenericLikelihoodModel):
    def loglike(self,params):
        return 0

class scad(PenalizedMixin,FixedGenericLikelihoodModel):
    pass
class L2PenalizedMixin(PenalizedMixin):
    def __init__(self, *args, **kwds):
        super(L2PenalizedMixin, self).__init__(*args, **kwds)

        penal = kwds.pop('penal', None)
        # I keep the following instead of adding default in pop for future changes
        if penal is None:
            # TODO: switch to unpenalized by default
            self.penal = L2SCADSmoothed(0.1, c0=0.0001)
        else:
            self.penal = penal

        # TODO: define pen_weight as average pen_weight? i.e. per observation
        # I would have prefered len(self.endog) * kwds.get('pen_weight', 1)
        # or use pen_weight_factor in signature
        self.pen_weight =  kwds.get('pen_weight', len(self.endog))

        self._init_keys.extend(['penal', 'pen_weight'])

class l2scad(L2PenalizedMixin,FixedGenericLikelihoodModel):
    pass
