"""
    Implementation of scikit-learn transformers for persistence
    landscapes.
"""
from operator import itemgetter
import numpy as np
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

    def fit(self, dgms, flatten: bool = False, vectorize: bool = False):
        return self

    def transform(self, dgms, flatten: bool = False):
        result = PersLandscapeExact(dgms=dgms, hom_deg=self.hom_deg).critical_pairs
        if flatten:
            return np.array(result).flatten()
        else:
            return np.array(result)


class PLA(BaseEstimator, TransformerMixin):
    """A scikit-learn transformer for converting persistence diagrams into persistence landscapes.

    Parameters
    ----------
    hom_deg : int
        Homological degree of persistence landscape.

    start : float, optional
        Starting value of approximating grid.

    stop : float, optional
        Stopping value of approximating grid.

    num_steps : int, optional
        Number of steps of approximating grid.


    Examples
    --------
    First instantiate the PLA object::

        >>> from persim import PLA
        >>> pla = PLA(hom_deg=0, num_steps=10)
        >>> print(pla)

        PLA(hom_deg=1,num_steps=10)

    The `fit()` method is first called a list of (-,2) numpy.ndarrays to determine the `start` and `stop` parameters of the approximating grid::

        >>> ex_dgms = [np.array([[0,3],[1,4]]),np.array([[1,4]])]
        >>> pla.fit(ex_dgms)

        PLA(hom_deg=0, start=0, stop=4, num_steps=10)

    The `transform()` method will then compute the values of the landscape functions on the approximated grid. The `flatten` flag determines if the output should be a flattened numpy array::

        >>> ex_pl = pla.transform(ex_dgms, flatten=True)
        >>> ex_pl

        array([0.        , 0.44444444, 0.88888889, 1.33333333, 1.33333333,
       1.33333333, 1.33333333, 0.88888889, 0.44444444, 0.        ,
       0.        , 0.        , 0.        , 0.44444444, 0.88888889,
       0.88888889, 0.44444444, 0.        , 0.        , 0.        ])
    """

    def __init__(
        self,
        hom_deg: int = 0,
        start: float = None,
        stop: float = None,
        num_steps: int = 500,
    ):
        self.hom_deg = hom_deg
        self.start = start
        self.stop = stop
        self.num_steps = num_steps

    def __repr__(self):
        if self.start is None or self.stop is None:
            return f"PLA(hom_deg={self.hom_deg}, num_steps={self.num_steps})"
        else:
            return f"PLA(hom_deg={self.hom_deg}, start={self.start}, stop={self.stop}, num_steps={self.num_steps})"

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
