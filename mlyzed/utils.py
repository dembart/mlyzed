
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

def read_cfg(file):

    """
    Read .cfg file format used in MLIP package

    Parameters
    ----------
    file: str
        path to the file
    
    Returns
    ----------
    list of ase's Atoms object with a SinglePointCalculator
    """
    with open(file, 'r') as f:
        text = f.readlines()
    traj = []
    for i_s, i_e in zip(np.where(np.array(text) == 'BEGIN_CFG\n')[0], np.where(np.array(text) == 'END_CFG\n')[0]):
        subtext = text[i_s:i_e]
        grade = None
        for i, line in enumerate(subtext):
            if 'ize' in line:
                size = int(subtext[i+1])
            if 'cell' in line:
                x = subtext[i+1]
                y = subtext[i+2]
                z = subtext[i+3]
                cell = np.array([x.split(), y.split(), z.split()], dtype = float)
            if 'AtomData' in line:
                pos, forces, numbers = [], [], []
                for n in range(size):
                    if len(subtext[i+n+1].split()) == 8:
                        id_, num, pos_x, pos_y, pos_z, f_x, f_y, f_z = subtext[i+n+1].split()
                        pos.append(np.array([pos_x, pos_y, pos_z], dtype = float))
                        forces.append(np.array([f_x, f_y, f_z], dtype = float))
                        numbers.append(num)
                    elif len(subtext[i+n+1].split()) == 5:
                        id_, num, pos_x, pos_y, pos_z = subtext[i+n+1].split()
                        pos.append(np.array([pos_x, pos_y, pos_z], dtype = float))
                        forces.append(np.array([None, None, None], dtype = float))
                        numbers.append(num)
            if 'MV' in line: 
                grade = float(line.split()[-1])
            if 'Energy' in line:
                energy =  float(subtext[i+1].split()[-1])
            if 'ess' in line:
                stress = np.array(subtext[i+1].split(), dtype = float)
        atoms = Atoms(cell = cell, positions = np.array(pos), numbers = numbers, pbc = True)
        results = {'energy': energy, 'forces': forces, 'stress': stress}
        calc = SinglePointCalculator(atoms, **results)
        atoms.calc = calc
        traj.append(atoms)
    return traj



def write_grd(data, atoms, path):

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

    Returns
    ----------
    nothing
    """

    voxels = data.shape[0] - 1, data.shape[1] - 1, data.shape[2] - 1
    cellpars = atoms.cell.cellpar()

    with open(path + '.grd', 'w+') as report:

        report.write('mlyzed generated chgcar' + '\n')
        report.write(''.join(str(p) + ' ' for p in cellpars).strip() + '\n')
        report.write(''.join(str(v) + ' ' for v in voxels).strip() + '\n')

        for i in range(voxels[0]):
            for j in range(voxels[1]):
                for k in range(voxels[2]):
                    val = data[i, j, k]
                    report.write(str(val) + '\n')
    
    print(f'File was written to {path}.grd\n')



def diffusion_coefficient(slope, dim = 3):

    """
    Calculate diffusion coefficient from the slope [angstrom^2 / ps]
    Params
    ------

    slope: float
        slope of the MSD vs. dt fit line,  [A ^ 2 / ps]
    
    dim: int, 3 by default
        dimensionality of diffusion
        
    Returns
    -------
    diffusivity [cm ^ 2 / s]

    """
    d = 1 / (2 * dim) * slope * (1e-16) / (1e-12)
    return d



def conductivity(D, n, z, T):
    """
    Calculate conductivity [S/cm]
    Params
    ------

    D: float
        diffusivity
    
    n: float
        number of mobile species divided by the volume of simulation box [1 / cm^3]

    z: float
        formal charge of a mobile specie

    T: float
        temperature [K]

    Returns
    -------
    conductivity [S / cm ]
    
    """
    e = 1.60217663e-19
    k = 1.380649e-23
    sigma = (z * e) ** 2 * n * D / (k * T)
    return sigma



def _get_range(x, y, region):
    
    region = np.array(region)
    x_new = x[(x < region.max())&(x > region.min())]
    if len(y.shape) == 1:
        y_new = y[(x < region.max())&(x > region.min())]
    else:
        y_new = y[:,(x < region.max())&(x > region.min())]
    return x_new, y_new