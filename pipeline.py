"""
MetaCausal pipeline: metabotype-aware heterogeneous treatment effect estimation.

Works identically on simulated data (simulate.py) and real NHANES (nhanes_prepare.py).
Core components
  1. Nuisance models with cross-fitting (gradient boosting)
  2. Average effect: doubly robust (AIPW / LinearDML)
  3. CATE learners: S, T, X, DR-learner, Causal Forest (DML) + linear-interaction baseline
  4. Heterogeneity inference (Chernozhukov, Demirer, Duflo, Fernandez-Val 2018/2025):
       BLP test, GATES (sorted group ATEs), CLAN (who is in top vs bottom group)
  5. Interpretable targeting: depth-2 doubly robust policy tree
  6. Explanation: SHAP values of the CATE function
Survey weights are supported via `w` (pass NHANES MEC/dietary weights).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingRegressor as HGBR,
                              HistGradientBoostingClassifier as HGBC,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from econml.dml import CausalForestDML, LinearDML, NonParamDML
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from econml.policy import DRPolicyTree


def reg(seed=0):
    # early-stopped boosting: adapts complexity to n (600 vs 2600+)
    return HGBR(max_iter=500, learning_rate=0.05, max_leaf_nodes=8,
                min_samples_leaf=20, l2_regularization=1.0, early_stopping=True,
                validation_fraction=0.2, n_iter_no_change=20, random_state=seed)


class ScaledLogit(ClassifierMixin, BaseEstimator):
    """Standardise + L2 logistic regression; accepts sample_weight (survey weights)."""
    def __init__(self, C=0.5):
        self.C = C

    def fit(self, X, y, sample_weight=None):
        self.sc_ = StandardScaler().fit(X, sample_weight=sample_weight)
        self.lr_ = LogisticRegression(C=self.C, max_iter=2000).fit(
            self.sc_.transform(X), y, sample_weight=sample_weight)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.sc_.transform(X))

    def predict(self, X):
        return self.lr_.predict(self.sc_.transform(X))


def clf(seed=0):
    # regularized logistic propensity: avoids the extreme weights that
    # flexible classifiers produce in small samples (overlap protection)
    return ScaledLogit(C=0.5)


# ---------------------------------------------------------------- nuisances
def crossfit_nuisances(X, T, y, w=None, k=5, seed=0, clip=0.03):
    """Out-of-fold mu0, mu1, e(x). Returns dict + AIPW pseudo-outcomes."""
    n = len(y)
    mu0, mu1, e = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        sw = None if w is None else w[tr]
        m = clf(seed).fit(X[tr], T[tr], sample_weight=sw)
        e[te] = m.predict_proba(X[te])[:, 1]
        for t, arr in ((0, mu0), (1, mu1)):
            idx = tr[T[tr] == t]
            r = reg(seed).fit(X[idx], y[idx],
                              sample_weight=None if w is None else w[idx])
            arr[te] = r.predict(X[te])
    e = np.clip(e, clip, 1 - clip)
    psi = mu1 - mu0 + T * (y - mu1) / e - (1 - T) * (y - mu0) / (1 - e)
    return dict(mu0=mu0, mu1=mu1, e=e, psi=psi)


def aipw_ate(psi, w=None):
    w = np.ones_like(psi) if w is None else w / w.mean()
    est = np.average(psi, weights=w)
    se = np.sqrt(np.average((psi - est) ** 2, weights=w) * np.sum(w ** 2)) / np.sum(w)
    return est, se


# ---------------------------------------------------------------- learners
def fit_learners(X, T, y, w=None, seed=0, which=None, modifier_idx=None):
    """Return dict name -> (fitted model, predict_fn). Out-of-sample CATE via predict_fn."""
    which = which or ["OLS-interaction", "S-learner", "T-learner", "X-learner",
                      "DR-learner", "CausalForest", "KG-CausalForest"]
    out = {}
    if "OLS-interaction" in which:
        Z = np.column_stack([X, T, X * T[:, None]])
        lr = LinearRegression().fit(Z, y, sample_weight=w)
        p = X.shape[1]
        coefT, coefXT = lr.coef_[p], lr.coef_[p + 1:]
        out["OLS-interaction"] = lambda Xn, a=coefT, b=coefXT: a + Xn @ b
    if "S-learner" in which:
        m = SLearner(overall_model=reg(seed)).fit(y, T, X=X)
        out["S-learner"] = m.effect
    if "T-learner" in which:
        m = TLearner(models=reg(seed)).fit(y, T, X=X)
        out["T-learner"] = m.effect
    if "X-learner" in which:
        m = XLearner(models=reg(seed), propensity_model=clf(seed)).fit(y, T, X=X)
        out["X-learner"] = m.effect
    if "DR-learner" in which:
        m = DRLearner(model_propensity=clf(seed), model_regression=reg(seed),
                      model_final=RandomForestRegressor(300, min_samples_leaf=30,
                                                        random_state=seed),
                      min_propensity=0.03, cv=5, random_state=seed)
        m.fit(y, T, X=X, sample_weight=w)
        out["DR-learner"] = m.effect
    if "CausalForest" in which:
        m = causal_forest(X, T, y, w, seed)
        out["CausalForest"] = m.effect
        out["_cf_model"] = m
    if "KG-CausalForest" in which:
        # Knowledge-guided CF: heterogeneity only along pre-registered modifiers
        # (metabotypes + a few clinical factors); all covariates adjust confounding.
        idx = modifier_idx if modifier_idx is not None else list(range(X.shape[1]))
        m = CausalForestDML(model_y=reg(seed), model_t=clf(seed), discrete_treatment=True,
                            n_estimators=1000, min_samples_leaf=50, max_samples=0.45,
                            honest=True, inference=True, cv=5, random_state=seed)
        m.fit(y, T, X=X[:, idx], W=X, sample_weight=w)
        out["KG-CausalForest"] = lambda Xn, m=m, idx=idx: m.effect(Xn[:, idx])
        out["_kgcf_model"] = m
    if "R-learner" in which:
        # Nie & Wager R-learner: cross-fitted residuals, flexible final stage
        m = NonParamDML(model_y=reg(seed), model_t=clf(seed), discrete_treatment=True,
                        model_final=RandomForestRegressor(300, min_samples_leaf=30, random_state=seed),
                        cv=5, random_state=seed)
        m.fit(y, T, X=X, sample_weight=w)
        out["R-learner"] = m.effect
    if "BCF" in which:
        # Bayesian causal forest (Hahn, Murray & Carvalho 2020) via stochtree, with a
        # gradient-boosted propensity score supplied as in the original paper
        from stochtree import BCFModel
        ps_model = clf(seed).fit(X, T)
        ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.03, 0.97)
        m = BCFModel()
        m.sample(X_train=X, Z_train=T.astype(float), y_train=y, propensity_train=ps,
                 observation_weights=None if w is None else w / np.mean(w),
                 num_gfr=10, num_burnin=0, num_mcmc=200,
                 general_params={"random_seed": seed})

        def bcf_effect(Xn, m=m, ps_model=ps_model):
            p = np.clip(ps_model.predict_proba(Xn)[:, 1], 0.03, 0.97)
            tau = m.predict(Xn, np.ones(len(Xn)), propensity=p, terms="cate")
            return np.asarray(tau).reshape(len(Xn), -1)                 # posterior draws

        out["BCF"] = lambda Xn, f=bcf_effect: f(Xn).mean(axis=1)        # posterior mean
        out["_bcf_draws"] = bcf_effect
    return out


def causal_forest(X, T, y, w=None, seed=0, n_estimators=1000):
    m = CausalForestDML(model_y=reg(seed), model_t=clf(seed), discrete_treatment=True,
                        n_estimators=n_estimators, min_samples_leaf=50,
                        max_samples=0.45, honest=True, inference=True, cv=5,
                        random_state=seed)
    m.fit(y, T, X=X, sample_weight=w)
    return m


# ---------------------------------------------------------------- inference on heterogeneity
def oof_cate(X, T, y, w=None, k=5, seed=0, learner="CausalForest", modifier_idx=None):
    """Out-of-fold CATE predictions (no unit's own outcome used for its CATE)."""
    tau = np.zeros(len(y))
    for tr, te in KFold(k, shuffle=True, random_state=seed + 99).split(X):
        f = fit_learners(X[tr], T[tr], y[tr], None if w is None else w[tr],
                         seed, which=[learner], modifier_idx=modifier_idx)
        tau[te] = np.ravel(f[learner](X[te]))
    return tau


def blp_test(tau_hat, nuis, T, w=None):
    """Best Linear Predictor: psi ~ b1 + b2 (tau_hat - mean).

    Heterogeneity test is ONE-SIDED, H0: b2 <= 0 vs H1: b2 > 0 (Chernozhukov et al.).
    A negative b2 means the predicted ranking reverses out of sample (noise fitting),
    which is evidence against detectable heterogeneity, not for it.
    Calibration test: H0: b2 = 1 (two-sided)."""
    from scipy.stats import norm
    Z = sm.add_constant(tau_hat - tau_hat.mean())
    fit = sm.WLS(nuis["psi"], Z, weights=np.ones_like(T) if w is None else w).fit(cov_type="HC3")
    b2, se2 = fit.params[1], fit.bse[1]
    return dict(beta1=fit.params[0], beta1_se=fit.bse[0], beta2=b2, beta2_se=se2,
                beta2_p=float(norm.sf(b2 / se2)),                       # one-sided, b2 > 0
                beta2_p_twosided=float(fit.pvalues[1]),
                beta2_eq1_p=float(2 * norm.sf(abs(b2 - 1) / se2)))      # calibration


def gates(tau_hat, nuis, n_groups=5, w=None):
    q = pd.qcut(pd.Series(tau_hat).rank(method="first"), n_groups, labels=False).values
    rows = []
    for g in range(n_groups):
        idx = q == g
        est, se = aipw_ate(nuis["psi"][idx], None if w is None else w[idx])
        rows.append(dict(group=g + 1, gate=est, se=se, n=idx.sum()))
    return pd.DataFrame(rows), q


def clan(df_X, groups, top=None, bottom=0):
    top = groups.max() if top is None else top
    a, b = df_X[groups == top], df_X[groups == bottom]
    return pd.DataFrame({"most_affected": a.mean(), "least_affected": b.mean(),
                         "difference": a.mean() - b.mean()})


def policy_tree(X, T, y, names, depth=2, seed=0, min_gain=3.0, W=None):
    """Treat only where the expected gain exceeds `min_gain` outcome points
    (a minimum worthwhile / clinically important difference). Implemented by
    charging the treated arm `min_gain`: y_adj = y - min_gain * T."""
    y = y - min_gain * T
    p = DRPolicyTree(max_depth=depth, min_samples_leaf=60, honest=True,
                     model_propensity=clf(seed), model_regression=reg(seed),
                     cv=5, random_state=seed)
    p.fit(y, T, X=X, W=W)   # W = full covariates for confounding adjustment
    return p


# ---------------------------------------------------------------- metrics (simulation only)
def cate_metrics(tau_hat, tau_true, top_frac=0.2):
    tau_hat = np.ravel(tau_hat)
    from scipy.stats import spearmanr
    k = int(len(tau_true) * top_frac)
    top_hat = np.argsort(-tau_hat)[:k]
    top_true = set(np.argsort(-tau_true)[:k])
    return dict(
        sqrt_pehe=float(np.sqrt(np.mean((tau_hat - tau_true) ** 2))),
        spearman=float(spearmanr(tau_hat, tau_true).correlation),
        ate_bias=float(tau_hat.mean() - tau_true.mean()),
        precision_top20=len(top_true.intersection(top_hat)) / k,
    )
