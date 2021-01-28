"""
    Implementation of scikit-learn transformers for persistence
    landscapes.
"""
from operator import itemgetter
from sklearn.base import BaseEstimator, TransformerMixin
from .pers_landscape_exact import PersLandscapeExact
from .pers_landscape_approx import PersLandscapeApprox


__all__ = ["PLE", "PLA"]


class PLE(BaseEstimator, TransformerMixin):
    """A scikit-learn transformer class for exact persistence landscapes. The transform
    method returns the list of critical pairs for the landscape. For a vectorized
    encoding of the landscape, using the PL_grid transformer.
    """

    def __init__(self, hom_deg: int = 0):
        self.hom_deg = hom_deg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = PersLandscapeExact(dgms=X, hom_deg=self.hom_deg)
        return result.critical_pairs


class PLA(BaseEstimator, TransformerMixin):
    """A scikit-learn transformer for grid persistence landscapes."""

    def __init__(
        self,
        hom_deg: int = 0,
        start: float = None,
        stop: float = None,
        num_steps: int = 500,
        verbose: bool = True
    ):
        self.hom_deg = hom_deg
        self.start = start
        self.stop = stop
        self.num_steps = num_steps

    def fit(self, dgms, flatten: bool = False):
        # TODO: remove infinities 
        _dgm = dgms[self.hom_deg]
        if self.start is None:
            self.start = min(_dgm, key=itemgetter(0))[0]
        if self.stop is None:
            self.stop = max(_dgm, key=itemgetter(1))[1]
        return self
        

    def transform(self, dgms, flatten: bool = False):
        result = PersLandscapeApprox(
            dgms=dgms,
            start=self.start,
            stop=self.stop,
            num_steps=self.num_steps,
            hom_deg=self.hom_deg,
        )
        if flatten:
            return result.values.flatten()
        else:
            return result.values
