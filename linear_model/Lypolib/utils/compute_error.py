import numpy as np
from pathlib import Path

from ..spectral_calc.base_preparator import BasePreparator
from ..config.default import INPUT_BTS

class CompareBTS(): # npz1 contains a mask, it's important to put it in the first place!

    def __init__(self, npz1, npz2):

        self.npz1 = np.load(npz1)["coeffs"]
        self.npz2 = np.load(npz2)["coeffs"]

        self.mask = np.load(npz1)["mask"]

        if self.npz1.shape != self.npz2.shape:
            print(f"Adjusting dimensions of arrays in order to match.")
            self.npz2 = self.npz2.transpose(1,0,2)

        # Background pixels are set to nan and, then, masked
        self.npz1[~self.mask] = np.full_like(self.npz1[0][0],np.nan)

        self.clean1 = self.npz1[self.mask]
        self.clean2 = self.npz2[self.mask]

        # Print error
        cell_mse = self.MSE()
        cell_types = BasePreparator().cell_types

        print("#" * 77)
        print(" ")
        print(f"Computed MSE for all cell_types and global percentage in {INPUT_BTS}:")

        for i in range(len(cell_types)):
            print(f"{cell_types[i]:<30}: {cell_mse[i]:10.8f}")
        
        print(f"6 Global percentage           : {cell_mse[-1]:10.8f}")
        
        print(" ")
        print("#" * 77)

    def MSE(self):
        """
        MSE computes the mean square error for each cell type and global percentage in order to analize individually and the total error in each brain tissue.

        """

        diff_magnitude = (self.clean1 - self.clean2)**2

        n_items = diff_magnitude.shape[0]

        cell_mse = []

        for i in range(self.clean1.shape[1]):
            err = np.nansum(diff_magnitude[:,i])
            cell_mse.append(err / n_items)

        return cell_mse

