import numpy as np

def mean_absolute_error(y, y_hat):
    mae = abs(y - y_hat).mean()
    return mae


def root_mean_squared_error(y, y_hat):
    rmse = np.sqrt(np.square(y - y_hat).mean())
    return rmse

def r2_score(y, y_hat):
    residuals = y - y_hat
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared
    

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
