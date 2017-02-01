import numpy as np
from scipy import stats
import statsmodels.base._penalties as smpen
import statsmodels.base._penalized 
from statsmodels.base._penalized import PenalizedMixin as PM
import statsmodels.base.model as base_model
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit, DiscreteModel
import statsmodels.discrete.discrete_model as dm

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
            self.penal = smpen.SCADSmoothed(0.1, c0=0.0001)
        else:
            self.penal = penal

        # TODO: define pen_weight as average pen_weight? i.e. per observation
        # I would have prefered len(self.endog) * kwds.get('pen_weight', 1)
        # or use pen_weight_factor in signature
        self.pen_weight =  kwds.get('pen_weight', len(self.endog))

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
            method = 'bfgs'

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


class PoissonPenalized(PenalizedMixin, Poisson):
    pass

class LogitPenalized(PenalizedMixin, Logit):
    pass
    
class ProbitPenalized(PenalizedMixin, Probit):
    pass

# simulate data
np.set_printoptions(suppress=True)
np.random.seed(987865)

nobs, k_vars = 500, 20
k_nonzero = 4
x = (np.random.rand(nobs, k_vars) + 0.5* (np.random.rand(nobs, 1)-0.5)) * 2 - 1
x *= 1.2
x[:, 0] = 1
beta = np.zeros(k_vars)
beta[:k_nonzero] = 1. / np.arange(1, k_nonzero + 1)
linpred = x.dot(beta)
mu = np.exp(linpred)
y = np.random.poisson(mu)
import os
debug = raw_input("please attach to pid:{},then press any key".format(os.getpid()))
modp = Poisson(y, x)
resp = modp.fit()
print(resp.params)

mod = PoissonPenalized(y, x)
res = mod.fit(method='bfgs', maxiter=1000)
print(res.params)

############### Penalized Probit
y_star = linpred + 0.25 * np.random.randn(nobs)
y2 = (y_star > 0.75).astype(float)
y_star.mean(), y2.mean()

res0 = Probit(y2, x).fit()
print(res0.summary())
res_oracle = Probit(y2, x[:, :k_nonzero]).fit()
print(res_oracle.params)

res_oracle.pred_table()
margeff = res_oracle.get_margeff()
print(margeff.summary())

modl = ProbitPenalized(y2, x)
modl.penal.tau = 0
resl = modl.fit(method='newton', disp=True)
print(resl.params)
print(resl.params - res0.params)

res_regl = Probit(y2, x).fit_regularized(alpha = 10)
print(res_regl.params)

(np.abs(res_regl.params)< 1e-4).sum(), (np.abs(res_regl.params[k_nonzero:])< 1e-4).mean()
res_regl.pred_table()

margeff = res_regl.get_margeff()
print(margeff.summary())

################# my 

#from statsmodels.regression.linear_model import GLS
#
#class XXXXXPenalized(PenalizedMixin):
#    pass
#mod = XXXXXPenalized(y, x)
#
#res = mod.fit(method='bfgs', maxiter=1000)
##res = mod.fit(method='pinv', maxiter=1000)
#print(res.params)

print "#"*26 + "Ridge"
from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
#reg.fit(x,y)
print reg.coef_

print "#"*26 + "OLS"
import statsmodels.api as sm
mod = sm.GLM(y,x)
res = mod.fit()
print(res.summary())

print "#"*26 + "OLSPenalized"
class OlsPenalized(PM,base_model.GenericLikelihoodModel):
    pass
mod = OlsPenalized(y,x)
#res = mod.fit(method="newton")
res = mod.fit()
print(res.summary())

