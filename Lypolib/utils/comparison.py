#####################################################################################################################################################################################################################
 
import numpy as np
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from ..spectral_calc.base_preparator import BasePreparator

#####################################################################################################################################################################################################################

ROOT_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

#####################################################################################################################################################################################################################


@dataclass
class PrepResults:
    """
    Data container for prepared spectral results.
    """
    bmz: np.ndarray
    T_AvS: np.ndarray
    SpS: np.ndarray
    cell_types: list



class CorrelationAnalysis(BasePreparator):

    def __init__(self, data: PrepResults, **kwargs):

        """
        CorrelationAnalysis provides tools to compare thresholded average spectra (T_AvS) 
        and spectral signatures (SpS) across different cell types.

        It generates:
        - Correlation heatmaps combining Pearson correlations of AvS and SpS.
        - Scatter plot matrices comparing intensity distributions.
        - Direct visual comparisons between T_AvS and SpS spectra.

        Args:
            data (PrepResults): Output from DataPreparator.
        """

        super().__init__(**kwargs)

        self.data = data

        self.bmz: np.ndarray = data.bmz
        self.T_AvS: np.ndarray = data.T_AvS
        self.SpS: np.ndarray = data.SpS
        self.cell_types: list = list(data.cell_types)

    
    def _data_preparation(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        """
        Convert numpy arrays (T_AvS, SpS) into Pandas DataFrames with cell type labels.
        
        Returns:
            tuple: (df_T_AvS, df_SpS)
        """
        
        df_SpS = pd.DataFrame(self.SpS.T, columns=self.cell_types)
        df_T_AvS = pd.DataFrame(self.T_AvS.T, columns=self.cell_types)
        
        return df_T_AvS, df_SpS

    def heatmap_correlation(self):
        
        """
        Generate a correlation heatmap:
        - Diagonal: Pearson correlation between T_AvS and SpS (per cell type).
        - Upper triangle: Pearson correlation between T_AvS across cell types.
        - Lower triangle: Pearson correlation between SpS across cell types.
        """
        
        df_T_AvS, df_SpS = self._data_preparation()

        # Compute correlations
        corr_X = df_T_AvS.corr().to_numpy()           # Upper triangle
        corr_Y = df_SpS.corr().to_numpy()            # Lower triangle
        #corr_XY = np.array([scipy.stats.pearsonr(self.T_AvS[i], self.SpS[i])[0] 
                            #for i in range(len(self.cell_types))])  # Diagonal

        corr_XY = []
        for i in range(len(self.cell_types)):
            x = self.T_AvS[i, :]
            y = self.SpS[i, :]
            
            mask = (~np.isnan(x)) & (~np.isnan(y))
            
            if np.sum(mask) > 1:  # pearson necesita al menos 2 puntos
                r = scipy.stats.pearsonr(x[mask], y[mask])[0]
            else:
                r = np.nan
                
            corr_XY.append(r)

        corr_XY = np.array(corr_XY)
        

        corr_map = np.zeros((len(self.cell_types), len(self.cell_types)))

        # Fill symmetric correlation matrix
        corr_map = np.triu(corr_X, k=1) + np.tril(corr_Y, k=-1)
        np.fill_diagonal(corr_map, corr_XY)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            corr_map, 
            annot=True, 
            cmap="hot", 
            vmin=0, vmax=1, 
            square=True, 
            xticklabels=self.cell_types, 
            yticklabels=self.cell_types, 
            ax=ax
        )
        ax.set_title("Similarity between spectra", fontsize=14, fontweight="bold")

        output_path = RESULTS_DIR / "heatmap.png"

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


    def scatter_plot(self):
        
        """
        Generate a scatter plot matrix:
        - Diagonal: T_AvS vs SpS for each cell type.
        - Upper triangle: T_AvS cross-comparison.
        - Lower triangle: SpS cross-comparison.
        """
        
        n = len(self.cell_types)
        fig, axes = plt.subplots(n, n, figsize=(12, 12))

        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if i == j:
                    ax.scatter(self.T_AvS[i, :], self.SpS[i, :], alpha=0.6, s=10, color="black")
                elif i > j:
                    ax.scatter(self.SpS[j, :], self.SpS[i, :], alpha=0.6, s=10, color="red")
                else:
                    ax.scatter(self.T_AvS[i, :], self.T_AvS[j, :], alpha=0.6, s=10, color="blue")

                # Labels only at the edges
                if i == 0:
                    ax.set_title(self.cell_types[j], fontweight='bold')
                if j == 0:
                    ax.set_ylabel(self.cell_types[i], fontweight='bold')

        # Global x-axis label
        fig.text(0.5, 0.04, "Intensity (a.u.)", ha="center", fontsize=12)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        
        output_path = RESULTS_DIR / "scatter.png"

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def AvS_SpS_comparison(self):
        
        """
        Compare thresholded average spectra (T_AvS) and spectral signatures (SpS) 
        side by side for each cell type.
        """
        
        fig, ax = plt.subplots(figsize=(12, 6), nrows=2, ncols=len(self.cell_types))

        for i in range(len(self.cell_types)):
            # Plot T_AvS
            ax[0, i].plot(self.bmz, self.T_AvS[i, :], color='blue')
            ax[0, i].set_title(self.cell_types[i])

            # Plot SpS
            ax[1, i].plot(self.bmz, self.SpS[i, :], color='red')
            ax[1, i].set_xlabel("m/z")

            # Y labels only for the first column
            if i == 0:
                ax[0, 0].set_ylabel("Intensity (a.u.)")
                ax[1, 0].set_ylabel("Intensity (a.u.)")

        fig.tight_layout()
        
        output_path = RESULTS_DIR / "comparison.png"

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot(self):
        try:
            # Heatmap
            self.heatmap_correlation()
            plt.show()

            # Scatter
            self.scatter_plot()
            plt.show()

            # Avs vs Sps
            self.AvS_SpS_comparison()
            plt.show()

        except Exception as e:
            ValueError(f"Error while performing correlation analysis: {e} ❌")
            
