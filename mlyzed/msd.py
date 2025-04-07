import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from tqdm import tqdm, trange
from ase.io import read
from ase.data import atomic_masses



def classical_msd(trajectory, specie = None, timestep = 1.0,
                correct_drift = True, projection = 'xyz'):
    """
    Calculate classical MSD from dr = r(t = 0) - r(t)
    
    Parameters
    ----------

    trajectory: mlyzed.Trajectory object
        trajectory 
    
    specie: str, e.g. 'Li'
        species for which MSD should be calculated 

    timestep: float, 1.0 by default
        time step in fs

    correct_drift: boolean, True by default
        correct drift of the center of mass of mobile species
    
    projection: str, allowed values are 'xyz', 'x', 'y', 'z'
        for wich projection MSD will be calculated
        
    Returns
    -------

    result: mlyzed.MSD object

    Examples
    --------

    >>> import mlyzed as md
    >>> traj = md.Trajectory.from_file('MD_trajectory.traj')
    >>> result = md.classical_msd(trajectory, specie = 'Li', timestep = 2.0)
    >>> result.plot()
    >>> dt = result.dt
    >>> msd = result.msd
    """
    
    trajectory = trajectory[:, trajectory.symbols == specie, :].copy()
    if correct_drift:
        trajectory.correct_com_drift()
    disp = trajectory.positions[0,:,:][None, :] - trajectory.positions[:,:,:]
    disp_projected = disp[:, :, _projection_key_mapper(projection)].reshape(disp.shape[0], disp.shape[1], len(projection))
    dt = np.arange(disp.shape[0]) * timestep / 1000
    msd = np.square(disp_projected).sum(axis = -1)
    results = {'dt': dt, 'msd': msd.mean(axis = 1), 'msd_std': None, 'msd_by_particle': msd}
    return MSD(results)



def block_msd(trajectory, specie = None, timestep = 1.0, n_blocks = 10, correct_drift = True):

    """
    Split trajectory into n_blocks non-overlapping parts and calculate
    classical MSD for each split from dr = r(t = 0) - r(t). 
    Allows obtaining errors of MSD.
    
    Parameters
    ----------
    
    trajectory: mlyzed.Trajectory object
        trajectory 

    specie: str, e.g. 'Li'
        species for which MSD should be calculated 

    timestep: float, 1.0 by default
        time step in fs

    n_blocks: int, 10 by default
        size of the split in ps 

    correct_drift: boolean, True by default
        correct drift of the center of mass of mobile species
        
    Returns
    -------

    result: mlyzed.MSD object
    
    Examples
    --------

    >>> import mlyzed as md
    >>> traj = md.Trajectory.from_file('MD_trajectory.traj')
    >>> result = md.block_msd(trajectory, specie = 'Li', timestep = 2.0, n_blocks = 10)
    >>> result.plot()
    >>> dt = result.dt
    >>> msd = result.msd
    >>> msd_std = result.msd_std
    
    """
    trajectory = trajectory[:, trajectory.symbols == specie, :].copy()
    if correct_drift:
        trajectory.correct_com_drift()
    dts, msds = [], []
    positions = trajectory.positions
    step = positions.shape[0]// n_blocks    
    blocks = np.arange(0, (step + 1) * n_blocks, step)   
    for start, stop in zip(blocks[0:-1], blocks[1:]):
        block = positions[start:stop,:, :]
        disp = block[0,:,:][None, :] - block[:,:,:]
        msd = np.square(disp).sum(axis = -1).mean(axis = -1)
        dt = timestep * np.arange(0, len(msd)) / 1000
        dts.append(dt)
        msds.append(msd)
    results = {
                'dt': dt,
                'msd': np.mean(msds, axis = 0),
                'msd_std': np.std(msds, axis = 0),
                'msd_by_particle': None,
                'msd_list': msds
                }
    return MSD(results)



def fft_msd(trajectory, specie = None, timestep = 1.0, correct_drift = True):

    # adopted from
    # https://stackoverflow.com/questions/69738376/how-to-optimize-mean-square-displacement
    # -for-several-particles-in-two-dimensions/69767209#69767209

    """        
    Calculate MSD using a fast Fourier transform algorithm

    Parameters
    ----------
    
    trajectory: mlyzed.Trajectory object
        trajectory 

    specie: str, e.g. 'Li'
        species for which MSD should be calculated 

    timestep: float, 1.0 by default
        time step in fs

    correct_drift: boolean, True by default
        correct drift of the center of mass of mobile species

    Returns
    -------
    
    result: mlyzed.MSD object

    Examples
    --------

    >>> import mlyzed as md
    >>> traj = md.Trajectory.from_file('MD_trajectory.traj')
    >>> msd = md.fft_msd(trajectory, specie = 'Li', timestep = 2)
    >>> msd.plot()
    
    """

    trajectory = trajectory[:, trajectory.symbols == specie, :].copy()
    if correct_drift:
        trajectory.correct_com_drift()
    pos = trajectory.positions.swapaxes(0,1)
    nTime=pos.shape[1]        

    S2 = np.sum ( np.fft.ifft( np.abs(np.fft.fft(pos, n=2*nTime, axis = -2))**2, axis = -2  )[:,:nTime,:].real , axis = -1 ) / (nTime-np.arange(nTime)[None,:] )

    D=np.square(pos).sum(axis=-1)
    D=np.append(D, np.zeros((pos.shape[0], 1)), axis = -1)
    S1 = ( 2 * np.sum(D, axis = -1)[:,None] - np.cumsum( np.insert(D[:,0:-1], 0, 0, axis = -1) + np.flip(D, axis = -1), axis = -1 ) )[:,:-1] / (nTime - np.arange(nTime)[None,:] )

    msd = S1-2*S2

    Dt_r = np.arange(1, pos.shape[1]-1)
    msd = msd[:,Dt_r]
    dt = timestep * Dt_r / 1000
    msd.mean(axis = 0)
    results = {
                'dt': dt,
                'msd': msd.mean(axis = 0),
                'msd_std': None,
                'msd_by_particle': msd,
                }
    return MSD(results)


def windowed_msd(trajectory, specie = None,  timestep = 1.0,
                n_frames = 75, t_min = 0.1, t_max = 0.7, correct_drift=True,
                n_bootstraps = 200):
    """
    Calculate windowed (time-averaged) MSD for the selected specie. 

    Parameters
    ----------
    
    trajectory: mlyzed.Trajectory object
        trajectory 

    specie: str, e.g. 'Li'
        species for which MSD should be calculated 

    timestep: float, 1.0 by default
        time step in fs

    n_frames: int, 75 by default
        number of different lagtimes to calculate MSD
    
    t_min: float, 0.1 by default
        ratio of the min lagtime to the entire trajectory length

    t_max: float, 0.7 by default
        ratio of the max lagtime to the entire trajectory length

    correct_drift: boolean, True by default
        correct drift of the center of mass of mobile species

    n_bootstraps: int, 200 by default
        bootstrap squared displacements for each particle and lagtime
        allows calculating MSD errors

    Returns
    -------

    result: mlyzed.MSD object

    Examples
    --------

    >>> from mlyzed import Lyze
    >>> calc = Lyze()
    >>> calc.read_file('MD_trajectory.traj')
    >>> result = calc.windowed_msd(specie = 'Li', timestep = 2, n_frames = 75, n_bootstraps = 100)
    >>> result.plot()

    """
    trajectory = trajectory[:, trajectory.symbols == specie, :].copy()
    if correct_drift:
        trajectory.correct_com_drift()
    positions = trajectory.positions

    lagtimes = np.linspace(max(1, t_min * (positions.shape[0] - 1)),
                           t_max * (positions.shape[0] - 1),
                           n_frames).astype(np.int64)
    
    msd_mean = []
    squared_displacements = []
    for lag in tqdm(lagtimes, desc = 'Getting lagtime MSD'):
        disp = positions[lag:] - positions[:-lag] 
        squared_disp = np.sum(disp**2, axis=2) 
        squared_displacements.append(squared_disp)
        msd_mean.append(squared_disp.mean(axis = -1).mean())
    results = {
                'dt': timestep * lagtimes / 1000,
                'msd': np.array(msd_mean),
                'msd_std': None,
                'msd_by_particle': squared_displacements,
                }
    
    if n_bootstraps:
        bootstrap_means = np.zeros((n_bootstraps, n_frames))
        for i in tqdm(range(n_bootstraps), desc='Bootstrapping displacements'):
            resampled_msd = []
            for lagtime_idx, squared_disp in enumerate(squared_displacements):
                N_frames_lag, N_particles = squared_disp.shape
                resampled_particles = np.random.choice(N_particles, size=N_particles, replace=True)
                resampled_frames = np.random.choice(N_frames_lag, size=N_frames_lag, replace=True)
                resampled_squared_disp = squared_disp[resampled_frames[:, None], resampled_particles]
                resampled_msd.append(np.mean(resampled_squared_disp))
            bootstrap_means[i] = resampled_msd
        mean_msd = np.mean(bootstrap_means, axis=0)
        std_msd = np.std(bootstrap_means, axis=0)
        results['msd'] = mean_msd
        results['msd_std'] = std_msd

    return MSD(results)



class MSD:

    """
    diffusivity':    cm^2 / s
    msd:             Angstrom^2
    dt:              ps
    
    """
    
    def __init__(self, results):
        
        """
        
        Parameters
        ----------
        
        results: dict
            dict with results, should include 'dt', 'msd', and 'msd_std' numpy arrays
            msd_std can be None
        """
        self.results = results
        self.dt = results['dt']
        self.msd = results['msd']
        self.msd_std = results['msd_std']
        self.set_fit_parameters()



    def set_fit_parameters(self, range = None, dim = 3):
        
        if dim not in [1, 2, 3]:
            raise ValueError('dim can be only 1, 2, or 3')
        self.dim = dim
        if range:
            start, stop = range
            if min(range) < self.dt.min() or min(range) > self.dt.max():
                start = self.dt.min()
            if max(range) > self.dt.max() or max(range) < self.dt.min():
                stop = self.dt.max()
            self._fit_range = (start, stop)
        else:
            dt_min, dt_max = self.dt.min(), self.dt.max()
            self._fit_range = (                 
                            dt_min + (dt_max - dt_min) * 0.1,
                            dt_min + (dt_max - dt_min) * 0.7,
                            )
        


    def fit_line(self):
        
        def line(x, intercept, slope):
            y = slope * x + intercept
            return y
        
        dt, msd = self._get_range(self.dt, self.msd, self._fit_range)
        
        if np.any(self.msd_std):
            _, msd_std = self._get_range(self.dt, self.msd_std, self._fit_range)
            popt, pcov  = curve_fit(line, dt, msd, sigma = msd_std, absolute_sigma = True)
        else:
            popt, pcov  = curve_fit(line, dt, msd)
        intercept, slope = popt
        slope_err = np.sqrt(np.diag(pcov))[1]
        intercept_err = np.sqrt(np.diag(pcov))[0]
        residuals = msd - line(dt, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msd-np.mean(msd))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return slope, intercept, slope_err, intercept_err, r_squared
    


    @property
    def diffusivity(self):

        """
        Calculate diffusion coefficient from the slope [angstrom^2 / ps]
        Params
        ------
        
        dim: int, 3 by default
            dimensionality of diffusion
            
        Returns
        -------
        d: float 
            diffusivity [cm ^ 2 / s]

        d_std: float
            diffusivity std [cm ^ 2 / s]

        """
        slope, _, err, _, _ = self.fit_line()
        d = 1 / (2 * self.dim) * slope * (1e-16) / (1e-12)
        d_std = 1 / (2 * self.dim) * err * (1e-16) / (1e-12)
        return d, d_std
    


    @staticmethod
    def _get_range(x, y, region):
        
        region = np.array(region)
        x_new = x[(x < region.max())&(x > region.min())]
        if len(y.shape) == 1:
            y_new = y[(x < region.max())&(x > region.min())]
        else:
            y_new = y[:,(x < region.max())&(x > region.min())]
        return x_new, y_new



    def plot(self, ax = None, show=False, dpi = 150, figsize = (4, 4), fit = True):

        """
        Plot MSD vs. dt using matplotlib.pyplot

        Parameters
        ----------

        ax: pyplot's ax, None by default
            if None will create a new figures and axis
        
        dpi: int, 150 by default
            resolution of the figure

        figsize: tuple(width, height), (6, 3.8) by default
            figure size
        
        fit: boolean, True by default
            fit a line to the MSD vs. dt curve
    
        show: boolean, False by default
            show plot
        """
        

        #plt.style.use('seaborn-v0_8-deep')
        plt.rcParams['axes.grid'] = False

        colors = [

            '#96B896',
            '#e7b995',
            'k',
            '#8B666E',
            '#627097',
            'darkred'
        ]


        if ax is None:
            fig = plt.figure(figsize=figsize, dpi = dpi)
            ax = plt.gca()
        else:
            fig = plt.gcf()

        dt = self.dt
        msd = self.msd
        msd_std = self.msd_std
        if np.any(msd_std):
            ax.fill_between(dt, msd - msd_std, msd + msd_std, label = f'MSD ± Std', color = colors[2],
                            alpha = 0.3)
        ax.plot(dt, msd, label = f'MSD', color = colors[2])
        ax.set_xlabel('Time, ps')
        ax.set_ylabel('MSD, $\AA^2$')
        ax.grid(alpha = 0.3, linewidth = 0.5, color = 'k')
        ax.set_xlim(dt.min(), dt.max())
        ax.set_ylim(msd.min(), msd.max())

        if fit:
            slope, intercept, slope_err, intercept_err, r_squared = self.fit_line()
            d, d_err = self.diffusivity
            
            d_string = ufloat(d, d_err)
                
            text =  f'$D$ = {d_string} cm$^2$/s'
            ax.plot(self.dt, self.dt * slope + intercept,
                        label = f'Fit ($R^2$ = {round(r_squared, 3)}), {text}',
                        color = colors[-1],
                        #zorder = -1,
                        linewidth = 1.0
                        )
            bound_upper = dt * (slope + slope_err) + (intercept + intercept_err)
            bound_lower = dt * (slope - slope_err) + (intercept - intercept_err)
            ax.fill_between(dt, bound_lower, bound_upper,
                 color = colors[-1], alpha = 0.2,
                 label = '95% CI'
                 )
            
            ax.vlines([self._fit_range[0], self._fit_range[1]], 0,  msd.max(),
                        color = colors[3],
                        linestyle = '--',
                        linewidth = 1.0,
                        label = 'Fit range')
        ax.legend(loc = 'lower right', fontsize = 7)
        if show:
            plt.show()
        return fig, ax



def _projection_key_mapper(projection):
    # credit: https://github.com/bjmorgan/kinisi
    mapper = {
            'xyz': np.s_[:],
            'x': np.s_[0],
            'y': np.s_[1],
            'z': np.s_[2],
            'xy': np.s_[:2],
            'xz': np.s_[::2],
            'yz': np.s_[1:],
    }
    return mapper[projection]