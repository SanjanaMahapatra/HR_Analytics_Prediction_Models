import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RandomCategoryImputer(BaseEstimator, TransformerMixin):
    def __init__(self, stratify_on=None, alpha=1.0, random_state=None):
        """
        stratify_on : column name or list of column names to condition on
        alpha       : Laplace smoothing constant (>= 0)
        random_state: seed for reproducibility
        """
        self.stratify_on = stratify_on
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y=None):
        # Always work on a DataFrame
        if isinstance(X, pd.Series):
            X = X.to_frame()
        self.global_categories_ = {
            col: X[col].dropna().unique().tolist()
            for col in X.columns
        }
        self.distributions_ = {}

        # Helper to compute smoothed probs
        def _get_probs(counts, categories):
            # reindex to include all categories
            counts = counts.reindex(categories, fill_value=0.0)
            smoothed = counts + self.alpha
            return smoothed / smoothed.sum()

        # No stratification: build one global distribution per column
        if not self.stratify_on:
            for col, cats in self.global_categories_.items():
                vc = X[col].value_counts(dropna=True)
                probs = _get_probs(vc, cats)
                self.distributions_[('global', col)] = {
                    'categories': cats,
                    'probs': probs.values
                }

        # Stratified: build a distribution for each (group, col)
        else:
            strat_cols = ([self.stratify_on]
                          if isinstance(self.stratify_on, str)
                          else list(self.stratify_on))

            for grp_vals, grp_df in X.groupby(strat_cols):
                for col, cats in self.global_categories_.items():
                    vc = grp_df[col].value_counts(dropna=True)
                    probs = _get_probs(vc, cats)
                    self.distributions_[(grp_vals, col)] = {
                        'categories': cats,
                        'probs': probs.values
                    }

        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame()
        X = X.copy()
        rng = np.random.RandomState(self.random_state)

        # Stratified path
        if self.stratify_on:
            strat_cols = ([self.stratify_on]
                          if isinstance(self.stratify_on, str)
                          else list(self.stratify_on))

            for grp_vals, _ in X.groupby(strat_cols):
                # build mask for group
                if len(strat_cols) == 1:
                    mask_grp = X[strat_cols[0]] == grp_vals
                else:
                    mask_grp = np.ones(len(X), dtype=bool)
                    for col_name, val in zip(strat_cols, grp_vals):
                        mask_grp &= (X[col_name] == val)

                for col in X.columns:
                    key = (grp_vals, col)
                    dist = self.distributions_.get(key)
                    if dist is None:
                        continue
                    mask_nan = mask_grp & X[col].isna()
                    cnt = mask_nan.sum()
                    if cnt:
                        draw = rng.choice(dist['categories'],
                                          size=cnt,
                                          p=dist['probs'])
                        X.loc[mask_nan, col] = draw

        # Global path
        else:
            for col in X.columns:
                dist = self.distributions_[('global', col)]
                mask_nan = X[col].isna()
                cnt = mask_nan.sum()
                if cnt:
                    draw = rng.choice(dist['categories'],
                                      size=cnt,
                                      p=dist['probs'])
                    X.loc[mask_nan, col] = draw

        # If single‐column input, return a Series
        return X.iloc[:, 0] if X.shape[1] == 1 else X