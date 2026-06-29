import numpy as np

from .base_reconstructor import BaseReconstructor
from ..spectral_calc.spectral_signatures import SpectralSignatures

from ..config import default

class DataReconstructor(BaseReconstructor):

    def __init__(self):

        data = BaseReconstructor()
        self.imzML = data.imzML
        self.max_x = data.max_x
        self.max_y = data.max_y
    
    
    def _nan_filter(self, vector):

        mask = ~np.isnan(vector)
        return np.vstack(vector)[mask]
    

    def _thresholded_BTS(self, AvS, threshold = default.THRESHOLD):
        return super()._thresholded_AvS(AvS, threshold)
    

    def spec_normalization(self, spectrum):

        norm_intA_i = spectrum * 1.0e8 / np.nansum(spectrum)
        return norm_intA_i
    
    
    def _fit(self, BTS, SpS):

        """ 
        BTS and SpS similarity is calculated
        
        """

        threshold = default.THRESHOLD

        # Applying threshold to BTS
        spec_thresh = np.where(BTS < threshold, 0.0, BTS)

        # Applying the same threshold mask to SpS columns
        X = SpS.T  # (nMz, nFeatures)
        X_thresh = np.where(spec_thresh[:, None] < threshold, 0.0, X)

        # Keeping only rows where neither BTS nor any SpS column is NaN
        data = np.column_stack((spec_thresh, X_thresh))
        valid = ~np.isnan(data).any(axis=1)

        if valid.sum() == 0:
            return np.zeros(SpS.shape[0])

        # Fitting BTS with SpS
        X_valid = X_thresh[valid, :]
        y_valid = spec_thresh[valid]

        # Computing coefficients
        w = np.linalg.pinv(X_valid, rcond=1e-6) @ y_valid

        return w
    
    
    def _clustering(self,bins_limits,mz,intensity):

        """
        Clustering is an important part of the preprocess in order to match indexes in mz and SpS's bmz. This process has to be done for each pixel.

        """

        bin_edges = np.arange(
            start = bins_limits[0]-0.006,
            stop = bins_limits[1]+0.012,
            step = 0.012
        )

        mz_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        intensity = self.spec_normalization(intensity)

        # Thresholding intensities
        intensity = self._thresholded_BTS(intensity.reshape(1,-1))
        intensity = intensity.reshape(-1)

        # Assign each m/z value to its corresponding bin
        indices = np.digitize(x = mz, bins = bin_edges) - 1

        # Filter out indices that are out of range (optional, for safety)
        valid = (indices >= 0) & (indices < len(bin_edges) - 1)
        indices = indices[valid]
        intensity_values = intensity[valid]
        
        # Sum intensities per bin using np.bincount
        intensity_binned = np.bincount(indices, weights=intensity_values, minlength=len(mz_bins))

        counts_per_bin = np.bincount(indices, minlength=len(mz_bins))

        # Set NaN only if a bin received no m/z values
        intensity_binned[counts_per_bin == 0] = np.nan

        return mz_bins, intensity_binned
        
            