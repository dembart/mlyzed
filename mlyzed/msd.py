import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from tqdm import tqdm, trange
from ase.io import read



def classical_msd(trajectory, specie = None, timestep = 1,
                correct_drift = True, projection = 'xyz', com = False):
    
    """
    Calculate classical MSD from dr = r(t = 0) - r(t)
    
    Parameters
    ----------

    specie: str, e.g. 'Li'
        species for which MSD should be calculated 

    timestep: int, 1 by default
        time step in fs

    correct_drift: boolean, True by default
        correct drift, used for tests
    
    projection: str, allowed values are 'xyz', 'x', 'y', 'z'
        for wich projection MSD will be calculated
        
    com: boolean, False by default
        calculate msd of the center of mass
        Note: Not well tested!

    Returns
    -------
    
    dt: np.array
        time in ps
    msd: np.array
        MSD

    Examples
    --------

    >>> import mlyzed as md
    >>> traj = md.Trajectory.from_file('MD_trajectory.traj')
    >>> msd = md.classical_msd(trajectory, specie = 'Li', timestep = 2)
    >>> msd.plot()
    
    """

    positions = trajectory.positions.copy().swapaxes(0,1)
    specie_idx = np.argwhere(trajectory.symbols == specie).ravel()
    framework_idx = np.argwhere(trajectory.symbols != specie).ravel()
    disp = positions[:,0,:][:, None] - positions[:,:,:]
    if correct_drift:
        disp -= disp[framework_idx, :, :].mean(axis = 0)[None, :]
    disp = disp[specie_idx, :, :]
    disp_projected = disp[:, :, _projection_key_mapper(projection)].reshape(disp.shape[0], disp.shape[1], len(projection))
    dt = np.arange(disp.shape[1]) * timestep / 1000
    if com:
        msd_com = np.square(np.linalg.norm(disp_projected.mean(axis = 0), axis = 1))
        return dt, msd_com
    msd = np.square(disp_projected).sum(axis = -1)
    results = {'dt': dt, 'msd': msd.mean(axis = 0), 'msd_std': None, 'msd_by_particle': msd}
    return MSD(results)



def block_msd(trajectory, specie = None, timestep = 1, n_blocks = 10):

    """
    Split trajectory into n_blocks non-overlapping parts and calculate
    classical MSD for each split from dr = r(t = 0) - r(t). 
    Allows obtaining errors of MSD.
    
    Parameters
    ----------
    
    timestep: int, 1 by default
        time step in fs
        
    specie: str, e.g. 'Li'
        species for which MSD should be calculated 

    n_blocks: int, 10 by default
        size of the split in ps 
        
    Returns
    -------
    
    dt: np.array
        time in ps
    msd_mean: np.array
        average MSD
    msd_std: np.array
        std of MSD
    

    Examples
    --------

    >>> import mlyzed as md
    >>> traj = md.Trajectory.from_file('MD_trajectory.traj')
    >>> msd = md.block_msd(trajectory, specie = 'Li', timestep = 2, n_blocks = 10)
    >>> msd.plot()
    
    """
    specie_idx = np.argwhere(trajectory.symbols == specie).ravel()
    framework_idx = np.argwhere(trajectory.symbols != specie).ravel()

    dts, msds = [], []
    positions = trajectory.positions.swapaxes(0,1)
    step = (positions.shape[1])// n_blocks    
    blocks = np.arange(0, (step + 1) * n_blocks, step)   
    for start, stop in zip(blocks[0:-1], blocks[1:]):
        block = positions[:, start:stop, :]
        disp = block[:,0,:][:, None] - block[:,:,:]
        disp -= disp[framework_idx, :, :].mean(axis = 0)[None, :]
        msd = np.square(disp[specie_idx, :, :]).sum(axis = -1).mean(axis = 0)
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



def fft_msd(trajectory, specie = None, timestep = 1.0):

    # adopted from
    # https://stackoverflow.com/questions/69738376/how-to-optimize-mean-square-displacement
    # -for-several-particles-in-two-dimensions/69767209#69767209

    """        
    Calculate MSD using a fast Fourier transform algorithm

    Parameters
    ----------
    
    timestep: int, 1 by default
        time step in fs
        
    specie: str, e.g. 'Li'
        species for which MSD should be calculated 

    Returns
    -------
    
    dt: np.array
        time in ps
    msd: np.array
        MSD

    Examples
    --------

    >>> import mlyzed as md
    >>> traj = md.Trajectory.from_file('MD_trajectory.traj')
    >>> msd = md.fft_msd(trajectory, specie = 'Li', timestep = 2)
    >>> msd.plot()
    
    """

    specie_idx = np.argwhere(trajectory.symbols == specie).ravel()
    framework_idx = np.argwhere(trajectory.symbols != specie).ravel()

    pos = trajectory.positions.swapaxes(0,1)[specie_idx, :, :]
    #pos = trajectory.positions[specie_idx,: , :]
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

    

def windowed_msd(trajectory, specie = 'Na', n_frames = 75, timestep = 1.0, min_frames = 10, n_bootstraps = 0):

    """ 
    Calculate windowed (time averaged) MSD for the selected specie. 
    Supposed to work the same way as MDAnalysis. Not tested  well.


    Parameters
    ----------

    n_frames: int, 75 by default
        number of different lagtimes to calculated MSD
        lagtimes = np.linspace(1, number of steps - 1, n_frames)

    Returns
    -------
    dt: np.array
        lagtimes
    msd: np.array
        mean squared displacements of selected specie
    msd_std: np.array
        standard deviation errors of msd

    msd_all: list
        msds for all frames

    Examples
    --------

    >>> from mlyzed import Lyze
    >>> calc = Lyze()
    >>> calc.read_file('MD_trajectory.traj')
    >>> dt, msd, msd_std, msd_all = calc.windowed_msd(specie = 'Li', timestep = 2, n_frames = 75)

    """


    specie_idx = np.argwhere(trajectory.symbols == specie).ravel()
    framework_idx = np.argwhere(trajectory.symbols != specie).ravel()

    positions = trajectory.positions.swapaxes(0,1)
    lagtimes = np.round(np.linspace(min_frames, positions.shape[1] - 1, n_frames))
    windows = {}
    msds_mean = []
    msds_std = []
    msds_all = []
    #msd_std = None
    for lag in tqdm(lagtimes, desc = 'Getting MSD vs. lagtime'):
        lag = int(lag)
        disp = positions[:,:-lag,:] - positions[:,lag:,:]
        #disp -= disp[framework_idx, :, :].mean(axis = 0)[None, :]
        msds_by_specie = np.square(disp[specie_idx, :, :]).sum(axis = 2)
        msd = np.square(disp[specie_idx, :, :]).sum(axis = 2).mean(axis = 0)
        
        msds_mean.append(msd.mean())
        #msds_std.append(msd.std()) #
        msds_all.append(msd)
        windows.update({lag: msds_by_specie})

    #trajectory.windows = windows
    msds_mean = np.array(msds_mean)
    #msds_std = np.array(msds_std)
    dt = timestep * lagtimes / 1000

    if n_bootstraps:
        data = windows
        msds = np.zeros((n_bootstraps, len(data.keys())))
        msds_mean = []
        msds_std = []
        #dt = []
        for i, t in enumerate(tqdm(data.keys())):
            msd = data[t].ravel()
            n_ind = data[t].shape[0] * int(np.floor(positions.shape[1] / t)) # N_atoms * non-overlapping trajectories
            msd_mean = msd.mean()
            #msd_std = msd.std()
            resample = []
            for _ in range(n_bootstraps):
                resample.append(np.random.choice(msd, n_ind).mean())
            resample = np.array(resample)
            msds[:, i] = resample
            msds_mean.append(msd_mean)
            msds_std.append(resample.std())

    results = {
                'dt': dt,
                'msd': np.array(msds_mean),
                'msd_std': msds_std,
                'msd_by_particle': None,
                'windows': windows,
                'msds_all': msds_all,
                }
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
        


    def plot(self, ax = None, show=False, dpi = 150, figsize = (7, 3), fit = True):

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

        ax.plot(dt, msd, label = f'Data', color = colors[2])
        ax.set_xlabel('Time, ps')
        ax.set_ylabel('MSD, $\AA^2$')
        ax.grid(alpha = 0.3, linewidth = 0.5, color = 'k')
        #ax.set_grid(False)
        ax.set_xlim(dt.min(), dt.max())
        ax.set_ylim(msd.min(), msd.max())

        # if np.any(msd_std):
        #     ax.fill_between(dt, msd - msd_std, msd + msd_std, alpha = 0.3,
        #                     color = colors[1])
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
                 color = 'k', alpha = 0.2,
                 label = '1-$\sigma$ interval'
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




class Arrhenius:

    def __init__(self, temperatures, msd_list):
        self.temperatures = np.array(temperatures)
        self.d = np.array([msd.diffusivity[0] for msd in msd_list])
        self.d_err = np.array([msd.diffusivity[1] for msd in msd_list])
        self.msd_list = msd_list
        self._fit()


    def _fit(self):
        
        temp = np.array(self.temperatures)
        d = self.d
        d_err = self.d_err
        
        def line(x, intercept, slope):
            y = slope * x + intercept
            return y
        
        def exponent(x, intercept, slope):
            return intercept * np.exp(-slope/x)
        
        #x = 1/temp
        #y = np.log(d)
        #yerr = d_err/d
        x = temp
        y = d
        yerr = d_err
        popt, pcov  = curve_fit(exponent, x, y, sigma = yerr, absolute_sigma = True)
        intercept, slope = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))
        #intercept_err = np.sqrt(np.diag(pcov))[0]

        residuals = y - line(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        self.slope = slope 
        self.slope_err = slope_err 
        self.intercept = intercept
        self.intercept_err = intercept_err
        self.r_squared = r_squared


    @property
    def barrier(self):
        return ufloat(-self.slope * 8.617e-5, self.slope_err * 8.617e-5)
    
    @property
    def factor(self):
        #d(lnx) = dx/x -> dx = x * d(lnx), x = D0, d(lnx) = self.intercept_err
        return ufloat(np.exp(self.intercept),  self.intercept_err * np.exp(self.intercept))


    def predict_diffusivity(self, T):

        d = np.exp(self.slope * 1/T + self.intercept)
        d_lower = np.exp((self.slope + self.slope_err) * 1/T + self.intercept - self.intercept_err)
        d_upper = np.exp((self.slope - self.slope_err) * 1/T + self.intercept + self.intercept_err)
        return d, d_lower, d_upper
    
    def predict_conductivity():
        pass
    
    def equation(self):
        text = f'Fit (R^2 = {np.round(self.r_squared, 3)}), D = {self.factor} exp (-{self.barrier} eV / kT) cm^2/s'
        return text


    def plot(self, show = False, axes = None, dpi = 150, figsize = (7, 3)):

        if axes is None:
            fig, (ax1, ax2) = plt.subplots(dpi = dpi, figsize = figsize, ncols = 2)
        else:
            fig = plt.gcf()
            ax1, ax2 = axes
        
        for (t, msd) in zip(self.temperatures, self.msd_list):
            ax1.plot(msd.dt, msd.msd, label = f'{t} K')
            if np.any(msd.msd_std):
                ax1.fill_between(msd.dt, msd.msd - msd.msd_std, msd.msd + msd.msd_std,
                                    alpha = 0.3)
            slope, intercept, slope_err, intercept_err, r_squared = msd.fit_line()
            ax1.plot(msd.dt, msd.dt * slope + intercept, color = 'k', linewidth = 0.75, linestyle = '--')
        ax1.set_xlabel('lagtime, ps')
        ax1.set_ylabel('MSD, $\AA^{2}$')
        ax1.grid(alpha = 0.5)
        ax1.set_ylim(0, max([msd.msd.max() for msd in self.msd_list]))
        ax1.set_xlim(min([msd.dt.min() for msd in self.msd_list]),
                     max([msd.dt.max() for msd in self.msd_list]))
        ax1.legend()

        x = 1000/self.temperatures
        y = np.log10(self.d)
        yerr = self.d_err / (np.log(10) * self.d)
        ax2.errorbar(x, y, yerr = yerr, linestyle = '',
                        capsize=2,
                        color = 'darkred',
                        markeredgewidth = 0.75, 
                        linewidth = 0.75,
                        marker = 'o',
                        markersize = 4
                    )
        
        ax2.set_xlabel('1000/T, 1/K')
        ax2.set_ylabel('log$_{10}$$D$, $cm^{2}/s$')


        ax2.plot(x, (1e-3/np.log(10) * self.slope * x + np.log10(np.exp(self.intercept))), color = 'k')
        #ax2.legend()
        ax2.grid(alpha = 0.5, linewidth = 0.5)
        plt.tight_layout()
        return fig, (ax1, ax2)
