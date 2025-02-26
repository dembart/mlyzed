import numpy as np
from tqdm import tqdm, trange
from ase.cell import Cell
from ase.io import read
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from uncertainties import ufloat


def probability_density(trajectory, specie, resolution = 0.2):

    """
    Calculate a time-averaged probability density function for a selected specie.
    Warning! Works properly for NVT simulations only!

    Parameters
    ----------
    specie: str
        chemical symbol

    resolution: float
        grid resolution in angstroms

    Returns
    -------

    PDF object
        probability density distribution 

    Examples
    --------

    """
    
    specie_idx = np.argwhere(trajectory.symbols == specie).ravel()
    scaled_traj = np.stack([Cell(cell).scaled_positions(pos) for cell, pos in zip(trajectory.cells, trajectory.positions)])
    scaled_traj = scaled_traj[:, specie_idx, :]
    scaled_traj -=np.floor(scaled_traj)
    bins = list(map(int, (np.diag(trajectory.cells[-1]) / resolution))) # shouldn't we use lengths of translation vectors?
    voxels = (scaled_traj * bins).astype(int)
    hist = np.zeros((bins), np.uint16)
    for point in np.vstack(voxels):
        hist[point[0], point[1], point[2]] += 1
    hist = hist / hist.sum() * (hist.shape[0] * hist.shape[1] * hist.shape[2]) /  np.linalg.det(trajectory.cells[-1])
    return PDF(hist, trajectory.get_frame(-1))



class PDF:

    """

    diffusivity':    cm^2 / s
    conductivity':   S / cm
    msd:             Angstrom^2
    dt:              ps
    
    """
    
    def __init__(self, hist, atoms):
        
        """
        Parameters
        ----------
        
        hist: np.array, 3D matrix
            probability density function calculated with mlyzed.Trajectory.probability_density

        atoms: ase's Atoms object
            reference structure for which PDF was calculated

        """
        self.data = hist 
        self.atoms = atoms



    def write_grd(self, path):

        """
        Write probability density distribution volumetric file for VESTA 3.0.

        Parameters
        ----------
        data: np.array of size LxMxN
            volumetric data
        
        atoms: ase's Atoms object
            atomic structure

        path: str
            path to save the file

        """

        voxels = self.data.shape[0] - 1, self.data.shape[1] - 1, self.data.shape[2] - 1
        cellpars = self.atoms.cell.cellpar()

        with open(path, 'w+') as report:

            report.write('mlyzed generated chgcar' + '\n')
            report.write(''.join(str(p) + ' ' for p in cellpars).strip() + '\n')
            report.write(''.join(str(v) + ' ' for v in voxels).strip() + '\n')

            for i in range(voxels[0]):
                for j in range(voxels[1]):
                    for k in range(voxels[2]):
                        val = self.data[i, j, k]
                        report.write(str(val) + '\n')
        
        print(f'File was written to {path}\n')

