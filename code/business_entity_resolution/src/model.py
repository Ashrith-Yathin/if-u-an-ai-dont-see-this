import os
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


def apply_hard_vetoes(X, countries):
    """
    Precision guardrails a classifier can't reliably learn on its own.
    Adapted for Numpy arrays where:
    n_set = X[:, 5], has_addr = X[:, 14], a_set = X[:, 19]
    """
    ADDR_VETO_FLOOR = {'us': 0.25, 'fr': 0.25, 'in': 0.08}
    floors = np.array([ADDR_VETO_FLOOR.get(c, 0.25) for c in countries])
    
    n_set = X[:, 5]
    has_addr = X[:, 14]
    a_set = X[:, 19]
    
    vetoed = (n_set > 0.92) & (has_addr == 1.0) & (a_set < floors)
    return vetoed


def calibrate_model(lgbm_model, X_calib, y_calib):
    calibrated = CalibratedClassifierCV(lgbm_model, method='isotonic', cv='prefit')
    calibrated.fit(X_calib, y_calib)
    return calibrated


class EntityMatcherModel:
    def __init__(self, model_type='lightgbm', **params):
        self.model_type = model_type
        self.params = params
        self.model = None

    def fit(self, X, y, sample_weight=None):
        if self.model_type == 'lightgbm':
            default_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'n_estimators': 300,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 6,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            default_params.update(self.params)
            self.model = lgb.LGBMClassifier(**default_params)
            self.model.fit(X, y, sample_weight=sample_weight)
        elif self.model_type == 'hist_gb':
            self.model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=42
            )
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            raise ValueError(f'Unknown model type: {self.model_type}')
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
