import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from .data_reconstructor import DataReconstructor
from ..spectral_calc.spectral_signatures import SpectralSignatures
from ..utils.clustering import KMeansClustering

from ..config import default

######################################################################

ROOT_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

save_results = True

######################################################################

class PlotCreator(DataReconstructor,SpectralSignatures):
    
    def __init__(self, bin_edges = default.INTERVAL):

        print(f"[INFO] Loading Brain Tissues.")

        data = DataReconstructor()
        self.imzML = data.imzML
        self.max_x = data.max_x
        self.max_y = data.max_y
        self.bin_edges = bin_edges

        print(f"[INFO] Loading all cell spectra and computing Spectral Signatures.")

        try:
            spectral = SpectralSignatures()
            self.SpS = spectral._Sps_calc()
            self.bmz = spectral.bmz

            self.cell_types = spectral.cell_types

            print(f"[INFO] Spectral Signatures done.")
        except:
            print(f"[WARNING] Unable to read Average Spectra!")

    def _clustering(self, bins_limits, mz, intensity):
        return super()._clustering(bins_limits, mz, intensity)

    def compute_pixel(self,i):

        # Clustering BTS
        spectrum = self.imzML.getspectrum(i)
        binned_mz, binned_BTS = self._clustering(self.bin_edges,spectrum[0],spectrum[1])

        mean = np.nanmean(binned_BTS)

        # Computing coefficients
        w = self._fit(binned_BTS,self.SpS)

        if np.isclose(0.0, np.sum(w), atol = 1e-8):
            coeff = w
            c = coeff
        else:
            coeff = np.clip(w,0,None)
            tot = sum(coeff)
            c = 100 * coeff / (tot + 1e-12)

        # Filtering SpS to compute cosine similarity 
        mask = ~np.isnan(binned_BTS)
        binned_BTS_valid = binned_BTS[mask]

        calc_BTS = coeff @ self.SpS
        mask = mask & ~np.isnan(calc_BTS) 

        calc_BTS_valid = calc_BTS[mask]
        binned_BTS_valid = binned_BTS[mask]

        # Cosine similarity
        cos_sim = cosine_similarity(calc_BTS_valid.reshape(1,-1),binned_BTS_valid.reshape(1,-1))

        # Correcting coefficients
        c = (c * cos_sim).reshape(-1)

        return c, cos_sim, mean

    def plot(self):

        # Initialize a matrix containing all cell contribution
        brain = []

        # Computing each pixel
        coords = self.imzML.coordinates

        kmatrix = np.zeros((self.max_x,self.max_y,1))

        print(f"[INFO] Computing BTS weights.")

        for i, (x,y,z) in enumerate(tqdm(coords, desc="Processing pixels", unit="px")):
            w_pixel, cos_sim, mean_val = self.compute_pixel(i)

            # Mean matrix for KMeans
            kmatrix[x-1][y-1] = mean_val
            new_coeff = np.append(w_pixel, np.sum(w_pixel))
            brain.append((x-1, y-1, new_coeff))

        print(f"[INFO] Clustering pixels in order to remove background.")

        # Computing clustering in order to remove the background
        cluster_map, max_index = KMeansClustering(kmatrix).KMeansCluster()

        print(f"[INFO] Plotting images.")
        
        # Plotting all data
        matrix = np.zeros((self.max_x, self.max_y, len(self.cell_types)+1))
        for x,y,coeff in brain:

            # Filtering background
            if cluster_map[x][y] != max_index:
                nan_list = np.full_like(coeff,np.nan)
                matrix[x,y,:] = nan_list
                continue
            
            matrix[x,y,:] = coeff

        # Saving .npz matrix in order to compare results
        if save_results:
            output_npz = ROOT_DIR / "npz"
            output_npz.mkdir(exist_ok=True)

            output_npz = output_npz / default.INPUT_BTS.replace("imzML","npz")

            np.savez(output_npz, coeffs = matrix)

        # Plotting images
        fig, axes = plt.subplots(1, len(self.cell_types)+1, figsize = (18,5))
        for i in range(len(self.cell_types)+1):
            im = axes[i].imshow(matrix[:,:,i].T,
                                origin = "lower",
                                cmap = "jet"
                                )
            if i < len(self.cell_types):
                axes[i].set_title(self.cell_types[i])
            else:
                axes[i].set_title("Global percentage")
            axes[i].axis("off")

            plt.colorbar(im, ax = axes[i], fraction = 0.046)
        

        plt.tight_layout()

        output_path = RESULTS_DIR / default.INPUT_BTS.replace("imzML","png")

        print(f"[INFO] Saving all images as: {output_path}")

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()