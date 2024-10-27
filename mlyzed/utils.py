
import numpy as np





def read_cfg(file):
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
        atoms.info.update(
            {
                'grade': grade,
                'stress': stress,
                'energy': energy,
            }
        )
        atoms.set_array('forces', np.array(forces))
        atoms.forces = np.stack(forces)
        traj.append(atoms)
    return traj


def mean_absolute_error(y, y_hat):
    mae = abs(y - y_hat).mean()
    return mae


def root_mean_squared_error(y, y_hat):
    rmse = np.sqrt(np.square(y - y_hat).mean())
    return rmse


def _eval(traj_true, traj_pred):
    metrics = {
        'rmse_force': None,     # eV/Angstrom 
        'mae_force': None,      # eV/Angstrom 
        'median_force_error': None, # eV/Angstrom
        'rmse_energy': None,    # eV/atom
        'mae_energy': None,     # eV/atom
        'rmse_stress': None,    # GPa
        'mae_stress': None,     # GPa
        #'force_range' : None,
        #'energy_range': None,
        'force_samples': None,
        'energy_samples': None,
    }

    forces_true = np.array([np.linalg.norm(a.arrays['forces'], axis = 1) for a in traj_true])
    forces_pred = np.array([np.linalg.norm(a.arrays['forces'], axis = 1) for a in traj_pred])
    rmse_force = root_mean_squared_error(forces_true.ravel(), forces_pred.ravel())
    mae_force = mean_absolute_error(forces_true.ravel(), forces_pred.ravel())
    median_force_error = np.median(abs(forces_true.ravel() - forces_pred.ravel()))
    
    energies_true = np.array([a.info['energy']/len(a) for a in traj_true])
    energies_pred = np.array([a.info['energy']/len(a) for a in traj_pred])
    mae_energy = mean_absolute_error(energies_true, energies_pred)
    rmse_energy =root_mean_squared_error(energies_true, energies_pred)

    conv_factor = 160.22 # eV to GPa
    stress_true = np.array([a.info['stress'] * conv_factor/a.cell.volume for a in traj_true])
    stress_pred = np.array([a.info['stress'] * conv_factor/a.cell.volume for a in traj_pred])
    rmse_stress = root_mean_squared_error(stress_true.ravel(), stress_pred.ravel())
    mae_stress = mean_absolute_error(stress_true.ravel(), stress_pred.ravel())

    metrics['rmse_force'] = rmse_force
    metrics['mae_force'] = mae_force
    metrics['rmse_energy'] = rmse_energy
    metrics['mae_energy'] = mae_energy
    metrics['rmse_stress'] = rmse_stress
    metrics['mae_stress'] = mae_stress
    metrics['median_force_error'] = median_force_error
    metrics['force_min'] = forces_true.ravel().min()
    metrics['force_max'] = forces_true.ravel().max()
    metrics['energy_min'] = energies_true.min() # ( energies_true.max())
    metrics['energy_max'] = energies_true.max() # ( energies_true.max())
    metrics['force_samples'] = len(forces_true.ravel())
    metrics['energy_samples'] = len(energies_true)
    return metrics


def angle_between_vectors(v1, v2):
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    angle = 180 * np.arccos(np.dot(v1_unit, v2_unit)) / np.pi
    return angle



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