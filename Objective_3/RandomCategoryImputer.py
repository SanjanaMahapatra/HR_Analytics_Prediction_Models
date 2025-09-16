import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RandomCategoryImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=None, alpha=1.0, stratify_on=None):
        self.random_state = random_state
        self.alpha = alpha  # Smoothing parameter for Laplace smoothing
        self.category_probs_ = {}
        self.stratify_on = stratify_on
        
    def fit(self, X, y=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        for col in X.columns:
            # Calculate probabilities of each category with Laplace smoothing
            value_counts = X[col].value_counts()
            smoothed_counts = {k: v + self.alpha for k, v in value_counts.items()}
            total = sum(smoothed_counts.values())
            smoothed_probs = {k: v / total for k, v in smoothed_counts.items()}
            self.category_probs_[col] = smoothed_probs
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            mask = X[col].isna()
            n_missing = mask.sum()
            if n_missing > 0:
                categories = list(self.category_probs_[col].keys())
                probabilities = list(self.category_probs_[col].values())
                random_choices = np.random.choice(categories, size=n_missing, p=probabilities)
                X_transformed.loc[mask, col] = random_choices
        return X_transformed