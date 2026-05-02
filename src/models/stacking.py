"""
Two-layer stacking ensemble:
  Base learners : XGBoost, LightGBM, SVM (RBF kernel)
  Meta-learner  : Logistic Regression (calibrated)

Training procedure:
  1. Tune each base learner independently with Optuna (maximise MCC on CV).
  2. Generate out-of-fold (OOF) probability predictions from each base learner.
  3. Stack OOF predictions as meta-features and train LR meta-learner.
  4. For test prediction: average base learner probabilities, feed to LR.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.logger import get_logger

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── Optuna objective functions ──────────────────────────────────────────────

def _xgb_objective(trial, X, y, cv, scale_pos_weight):
    from xgboost import XGBClassifier
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 2.0),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric":      "logloss",
        "verbosity":        0,
        "random_state":     42,
    }
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        clf = XGBClassifier(**params)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[val_idx])
        scores.append(matthews_corrcoef(y[val_idx], pred))
    return float(np.mean(scores))


def _lgbm_objective(trial, X, y, cv, scale_pos_weight):
    from lightgbm import LGBMClassifier
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 600),
        "max_depth":       trial.suggest_int("max_depth", 3, 9),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":      trial.suggest_int("num_leaves", 20, 100),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":      trial.suggest_float("reg_lambda", 0.0, 2.0),
        "class_weight":    "balanced",
        "verbosity":       -1,
        "random_state":    42,
    }
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        clf = LGBMClassifier(**params)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[val_idx])
        scores.append(matthews_corrcoef(y[val_idx], pred))
    return float(np.mean(scores))


def _svm_objective(trial, X, y, cv):
    params = {
        "C":     trial.suggest_float("C", 0.01, 200, log=True),
        "gamma": trial.suggest_float("gamma", 1e-5, 1.0, log=True),
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42,
    }
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        clf = SVC(**params)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[val_idx])
        scores.append(matthews_corrcoef(y[val_idx], pred))
    return float(np.mean(scores))


# ─── base learner tuner ──────────────────────────────────────────────────────

def tune_base_learner(name: str, X: np.ndarray, y: np.ndarray,
                      n_trials: int = 50, cv_folds: int = 5,
                      random_state: int = 42) -> dict:
    """Run Optuna study for a single base learner. Returns best params."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scale_pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)

    log.info(f"Tuning {name} ({n_trials} trials)...")

    if name == "xgb":
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(
            lambda t: _xgb_objective(t, X, y, cv, scale_pos_weight),
            n_trials=n_trials, show_progress_bar=False
        )
    elif name == "lgbm":
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(
            lambda t: _lgbm_objective(t, X, y, cv, scale_pos_weight),
            n_trials=n_trials, show_progress_bar=False
        )
    elif name == "svm":
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(
            lambda t: _svm_objective(t, X, y, cv),
            n_trials=n_trials, show_progress_bar=False
        )
    else:
        raise ValueError(f"Unknown learner: {name}")

    best_params = study.best_params
    log.info(f"  {name} best MCC={study.best_value:.4f}, params={best_params}")
    return best_params


# ─── stacking ensemble ───────────────────────────────────────────────────────

class StackingEnsemble:
    """
    Two-layer stacking ensemble with Optuna-tuned base learners.

    Workflow:
        ensemble.fit(X_train, y_train)        # tunes + trains all layers
        proba = ensemble.predict_proba(X_test) # final predictions
        pred  = ensemble.predict(X_test)
    """

    def __init__(self, n_optuna_trials: int = 50,
                 cv_folds: int = 10,
                 random_state: int = 42):
        self.n_optuna_trials = n_optuna_trials
        self.cv_folds = cv_folds
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._best_params: Dict[str, dict] = {}
        self._base_clfs: Dict[str, object] = {}
        self._meta_clf: Optional[LogisticRegression] = None
        self._fitted = False

    # ── internal helpers ────────────────────────────────────────────────────

    def _build_clf(self, name: str, params: dict):
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        scale_pos_weight = params.pop("scale_pos_weight", 1.0)

        if name == "xgb":
            return XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                verbosity=0,
                random_state=self.random_state,
            )
        elif name == "lgbm":
            return LGBMClassifier(
                **params,
                class_weight="balanced",
                verbosity=-1,
                random_state=self.random_state,
            )
        elif name == "svm":
            return CalibratedClassifierCV(
                SVC(
                    **params,
                    class_weight="balanced",
                    probability=False,
                    random_state=self.random_state,
                ),
                cv=3,
                method="sigmoid"
            )
        raise ValueError(name)

    def _generate_oof(self, X: np.ndarray,
                      y: np.ndarray) -> np.ndarray:
        """Generate out-of-fold probability predictions (N, n_base_learners)."""
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        oof = np.zeros((len(X), len(self._base_clfs)), dtype=np.float64)
        names = list(self._base_clfs.keys())

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            log.info(f"  OOF fold {fold+1}/{self.cv_folds}")
            for j, name in enumerate(names):
                clf = self._build_clf(name, dict(self._best_params[name]))
                clf.fit(X[train_idx], y[train_idx])
                oof[val_idx, j] = clf.predict_proba(X[val_idx])[:, 1]

        return oof

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray):
        log.info(f"Fitting stacking ensemble on {X.shape}...")

        # scale for SVM
        X_sc = self._scaler.fit_transform(X)

        # tune base learners (use scaled X for SVM, raw for tree methods)
        for name in ["xgb", "lgbm"]:
            params = tune_base_learner(
                name, X, y,
                n_trials=self.n_optuna_trials,
                cv_folds=5,
                random_state=self.random_state
            )
            self._best_params[name] = params

        svm_params = tune_base_learner(
            "svm", X_sc, y,
            n_trials=self.n_optuna_trials,
            cv_folds=5,
            random_state=self.random_state
        )
        self._best_params["svm"] = svm_params

        # Build final base classifiers (trained on full training set)
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        scale_pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
        xgb_p = dict(self._best_params["xgb"])
        self._base_clfs["xgb"] = XGBClassifier(
            **xgb_p, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", verbosity=0, random_state=self.random_state
        )
        self._base_clfs["xgb"].fit(X, y)

        lgbm_p = dict(self._best_params["lgbm"])
        self._base_clfs["lgbm"] = LGBMClassifier(
            **lgbm_p, class_weight="balanced",
            verbosity=-1, random_state=self.random_state
        )
        self._base_clfs["lgbm"].fit(X, y)

        svm_p = dict(self._best_params["svm"])
        svm_base = SVC(**svm_p, class_weight="balanced",
                       probability=False, random_state=self.random_state)
        self._base_clfs["svm"] = CalibratedClassifierCV(
            svm_base, cv=3, method="sigmoid"
        )
        self._base_clfs["svm"].fit(X_sc, y)

        # Generate OOF for meta-learner
        log.info("Generating out-of-fold predictions for meta-learner...")
        oof_tree = self._generate_oof_tree(X, y)  # XGB + LGBM OOF
        oof_svm = self._generate_oof_svm(X_sc, y)  # SVM OOF
        oof = np.column_stack([oof_tree, oof_svm])

        # Train meta-learner
        self._meta_clf = LogisticRegression(
            C=1.0, max_iter=1000, random_state=self.random_state
        )
        self._meta_clf.fit(oof, y)
        log.info("Meta-learner trained.")

        self._fitted = True
        return self

    def _generate_oof_tree(self, X, y):
        """OOF for XGB and LGBM."""
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        oof = np.zeros((len(X), 2))
        scale_pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)

        for fold, (tr, val) in enumerate(cv.split(X, y)):
            log.info(f"  Tree OOF fold {fold+1}/{self.cv_folds}")
            xgb = XGBClassifier(**dict(self._best_params["xgb"]),
                                 scale_pos_weight=scale_pos_weight,
                                 eval_metric="logloss", verbosity=0,
                                 random_state=self.random_state)
            xgb.fit(X[tr], y[tr])
            oof[val, 0] = xgb.predict_proba(X[val])[:, 1]

            lgbm = LGBMClassifier(**dict(self._best_params["lgbm"]),
                                   class_weight="balanced", verbosity=-1,
                                   random_state=self.random_state)
            lgbm.fit(X[tr], y[tr])
            oof[val, 1] = lgbm.predict_proba(X[val])[:, 1]

        return oof

    def _generate_oof_svm(self, X_sc, y):
        """OOF for calibrated SVM."""
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        oof = np.zeros((len(X_sc), 1))
        for fold, (tr, val) in enumerate(cv.split(X_sc, y)):
            log.info(f"  SVM OOF fold {fold+1}/{self.cv_folds}")
            svm_p = dict(self._best_params["svm"])
            svm_base = SVC(**svm_p, class_weight="balanced",
                           probability=False, random_state=self.random_state)
            clf = CalibratedClassifierCV(svm_base, cv=3, method="sigmoid")
            clf.fit(X_sc[tr], y[tr])
            oof[val, 0] = clf.predict_proba(X_sc[val])[:, 1]
        return oof

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first."
        X_sc = self._scaler.transform(X)
        meta_feats = np.column_stack([
            self._base_clfs["xgb"].predict_proba(X)[:, 1],
            self._base_clfs["lgbm"].predict_proba(X)[:, 1],
            self._base_clfs["svm"].predict_proba(X_sc)[:, 1],
        ])
        proba_aip = self._meta_clf.predict_proba(meta_feats)[:, 1]
        proba = np.column_stack([1 - proba_aip, proba_aip])
        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def cv_evaluate(self, X: np.ndarray, y: np.ndarray,
                    n_folds: int = 10) -> Dict[str, List[float]]:
        """
        Full 10-fold cross-validation evaluation of the complete
        stacking pipeline — reports per-fold metrics.
        """
        from src.evaluation.metrics import compute_all_metrics
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=self.random_state)
        fold_metrics = []

        for fold, (tr, val) in enumerate(cv.split(X, y)):
            log.info(f"CV fold {fold+1}/{n_folds}")
            clone = StackingEnsemble(
                n_optuna_trials=self.n_optuna_trials,
                cv_folds=5,
                random_state=self.random_state
            )
            clone.fit(X[tr], y[tr])
            proba = clone.predict_proba(X[val])[:, 1]
            pred = (proba >= 0.5).astype(int)
            m = compute_all_metrics(y[val], pred, proba)
            fold_metrics.append(m)
            log.info(f"  Fold {fold+1}: ACC={m['accuracy']:.4f}, "
                     f"MCC={m['mcc']:.4f}")

        # aggregate
        all_keys = fold_metrics[0].keys()
        agg = {k: [fm[k] for fm in fold_metrics] for k in all_keys}
        return agg

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info(f"Saved ensemble → {path}")

    @staticmethod
    def load(path: str) -> "StackingEnsemble":
        ens = joblib.load(path)
        log.info(f"Loaded ensemble from {path}")
        return ens
