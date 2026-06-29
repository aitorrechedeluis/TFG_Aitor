import numpy as np
import pathlib as Path
from pyimzml.ImzMLParser import ImzMLParser
import os

from ..spectral_calc.spectral_signatures import SpectralSignatures

class BaseReconstructor(SpectralSignatures): # Read and prepare data to reconstruct the images

    def __init__(self,path = "data/BTS/", brain = None):
        
        self.path = path
        self.brain = brain

        if brain is None:
            from ..config import default
            brain = default.INPUT_BTS
        self.brain = brain

        # Initialize all cell Spectral Signature
        try:
            cells = SpectralSignatures()
            self.SpS = cells._Sps_calc()
        except:
            print(f"❌ SpectralSignatures not working!")
        
        # Initialize BTS 
        self.imzML, self.max_x, self.max_y = self._load_BTS()

        
    def _load_BTS(self):

        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"❌ Base path does not exist: {self.path}")
            
        parser = ImzMLParser(os.path.join(self.path,self.brain))
        coords = parser.coordinates

        max_x = max(c[0] for c in coords)
        max_y = max(c[1] for c in coords)

        return parser, max_x, max_y