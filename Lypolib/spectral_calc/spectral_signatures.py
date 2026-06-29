import numpy as np

from .base_preparator import BasePreparator

class SpectralSignatures(BasePreparator): # Computing each cell SpectralSignature (Sps)

    def __init__(self):

        data = BasePreparator()
        self.AvS = data.AvS
        self.bmz = data.bmz
        self.SpS = np.full_like(self.AvS,np.nan,dtype=float)
        self.T_AvS = self._thresholded_AvS(self.AvS)
        self.cell_types = data.cell_types

    def _thresholded_AvS(self,AvS,threshold=5.0e4): #Applying a THRESHOLD = 5.0e4
        """
        This functions returns thresholded AvS and bmz

        """

        T_AvS = []

        for spectrum in AvS:
            spec = spectrum.copy()  # evitar modificar el original
            spec[spec <= threshold] = 0.
            T_AvS.append(spec)
        
        return np.vstack(T_AvS)
    
    def _compute_coeff(self):
        
        """
        By taking the pseudoinverse of the T_AvS matrix we can easily compute de weights of each lipotype
        since AvS_i = sum(w_j * AvS_j) -> w_j = (AvS_i)^+ * AvS_j. And computing this for each lipotype gives us the caracteristic
        spectrum for each cell.
        """

        cell_types = self.T_AvS.shape[0]

        w = []

        non_nan_T_AvS = self.T_AvS

        for i in range(cell_types):

            # Filtering nan in order to get a common index
            nan_mask = ~np.isnan(non_nan_T_AvS[i])
            non_nan_T_AvS_masked = np.vstack([non_nan_T_AvS[j][nan_mask] for j in range(non_nan_T_AvS.shape[0])])
            non_nan_T_AvS = non_nan_T_AvS_masked

        for i in range(cell_types): 

            mask = np.arange(cell_types) != i
            X = non_nan_T_AvS[mask].T
            y = non_nan_T_AvS[i]

            w_j = np.linalg.pinv(X, rcond=1e-6) @ y

            w.append(w_j)

        return np.vstack(w)

    def _normalize(self,w_j):
        """
        Normalizing w_j vectors

        """

        norm = sum(w_j)
        if norm == 0:
            raise ValueError(f"Vector's values are all 0")
        return w_j / norm


    def _Sps_calc(self):

        """
        Using weights each cell spectrall signatures is calculated. 

        """

        cell_types = self.T_AvS.shape[0]

        # Obtaining coefficients 
        w = self._compute_coeff()

        for i in range(cell_types):
            mask = np.arange(cell_types) != i
            X = self.T_AvS[mask]
            y = self.T_AvS[i].copy()

            # Reconstruct the contribution of the other cell types
            signRescA = w[i] @ X         # (nMz,)

            residual = y - signRescA

            # FIX: keep only bins where both the residual AND the original
            # spectrum are positive (avoids keeping noise artefacts)
            mask_positive = (residual > 0) & (y > 0)
            self.SpS[i] = np.where(mask_positive, residual, 0.0)

        return self.SpS

