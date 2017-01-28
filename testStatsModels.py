import numpy as np
from scipy import stats
import statsmodels.base._penalties as smpen
import statsmodels.base.model as base_model
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
import statsmodels.discrete.discrete_model as dm

class PenalizedMixin(object):

    def __init__(self, penal=None, *args,**kwds):
        super(PenalizedMixin, self).__init__(*args, **kwds)
        
        if penal is None:
            self.penal = smpen.SCADSmoothed(0.1, c0=0.0001)
        else:
            self.penal = penal
        self.pen_weight = len(self.endog) #100.

    def loglike(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight
            
        llf = super(PenalizedMixin, self).loglike(params)
        return llf - pen_weight * self.penal.func(params)

    def score(self, params, pen_weight=None):
        if pen_weight is None:
            pen_weight = self.pen_weight
            
        sc = super(PenalizedMixin, self).score(params)
        return sc - pen_weight * self.penal.grad(params)

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
            
        sc = super(PenalizedMixin, self).hessian(params)
        return sc - np.diag(pen_weight * self.penal.deriv2(params))


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