import numpy as np
import argparse
from pathlib import Path

from Lypolib.spectral_calc.base_preparator import BasePreparator
from Lypolib.spectral_calc.spectral_signatures import SpectralSignatures
from Lypolib.reconstruction.base_reconstructor import BaseReconstructor
from Lypolib.reconstruction.data_reconstructor import DataReconstructor
from Lypolib.reconstruction.plot_creator import PlotCreator
from Lypolib.utils.comparison import CorrelationAnalysis
from Lypolib.utils.clustering import KMeansClustering
from Lypolib.utils.compute_error import CompareBTS


from Lypolib.config import default

import warnings


####################################################################

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pyimzml"
)

warnings.simplefilter(
    "ignore", 
    FutureWarning
)

####################################################################

parser = argparse.ArgumentParser(description="Sobrescribir valores de config desde consola")

parser.add_argument("--input", type=str)
parser.add_argument("--interval", type=float)
parser.add_argument("--clusters", type=int)
parser.add_argument("--threshold", type=int)

args = parser.parse_args()

# Input values
if args.input:
    default.INPUT_BTS = args.input
if args.interval:
    default.INTERVAL = args.interval
if args.clusters:
    default.N_CLUSTERS = args.clusters
if args.threshold:
    default.THRESHOLD = args.threshold

####################################################################

PATH_REFERENCE = Path("npz/Martin") / default.INPUT_BTS.replace("imzML","npz")
PATH_RESULTS = Path("npz") / default.INPUT_BTS.replace("imzML","npz")

####################################################################


def __main__():
    model = PlotCreator()
    return model

if __name__ == "__main__":
    a = __main__()
    a.plot()
    CompareBTS(PATH_REFERENCE,PATH_RESULTS)