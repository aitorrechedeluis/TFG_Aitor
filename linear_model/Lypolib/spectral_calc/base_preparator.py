import numpy as np
import os
from scipy.io import loadmat



class BasePreparator():

    def __init__(self,base_path="data/cell_types_spectrum/",common="clean/represSpectr_calc.mat"):

        self.base_path = base_path
        self.common = common

        self.AvS = []
        self.bmz = None
        self.cell_types = []

        self._load()


    def _load(self):

        if not os.path.isdir(self.base_path):
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")
        
        for types in os.listdir(self.base_path):
            self.cell_types.append(types)
            types=types+"/"
            celltype_path = os.path.join(self.base_path,types,self.common)

            if not os.path.isfile(celltype_path):
                print(f"[WARNING] File not found: {celltype_path}")
                continue

            data = loadmat(celltype_path)

            self.AvS.append(data["reprSpec"].flatten())

            if self.bmz is None:
                self.bmz = data["M_Z"].flatten()
    
        if not self.AvS:
            raise ValueError(f"No spectra were loaded.")  
        
        return np.vstack(self.AvS),self.bmz
        
           