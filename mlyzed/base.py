
"""
Lyze - the main class of mlyzed library. Used to calculate MSD.  
"""

import numpy as np
from tqdm import tqdm, trange
from ase.io import read
#from sklearn.linear_model import LinearRegression as linregress_w

__version__ = "0.1"

class Lyze:
    
    def __init__(self, verbose = True):
        
        """
        Initialization of Lyze object
        
        Parameters
        ----------
        
        verbose: boolean, True by default
            print additional information
        """
        
        self.verbose = verbose
        # the trick was taken from the kinisi project (https://github.com/bjmorgan/kinisi)
        self._projection_key_mapper = {
                                        'xyz': np.s_[:],
                                        'x': np.s_[0],
                                        'y': np.s_[1],
                                        'z': np.s_[2],
                                        'xy': np.s_[:2],
                                        'xz': np.s_[::2],
                                        'yz': np.s_[1:],
        }



    def _map_atomic_types(self, atom_types_mapper, numbers):
        u,inv = np.unique(numbers,return_inverse = True)
        return np.array([atom_types_mapper[x] for x in u])[inv].reshape(numbers.shape)



    def read_atoms(self, atoms_list, unwrap = True, atom_types_mapping = None):

        """
        Read list of Ase's atoms
        
        Parameters
        ----------
        
        atoms_list: list of Ase's atoms
            trajectory

        unwrap: boolean, True by default
            perform unwrapping of the coordinates

        Examples
        --------

        >>> from ase.io import read
        >>> from mlyzed import Lyze
        >>> atoms_list = read('MD_file.traj', index = ':')
        >>> calc = Lyze()
        >>> calc.read_atoms(atoms_list, unwrap = True)

        """
        structures = atoms_list
        self.atoms = structures[-1]
        if atom_types_mapping:
            for st in structures:
                st.symbols = self._map_atomic_types(atom_types_mapping, st.numbers)

        self._cells = np.array([st.cell for st in structures])
        self.symbols = np.array(structures[0].symbols)
        
        if unwrap:
            unwrapped = self.unwrap(structures)
            trajectory = np.array([np.dot(unwrapped[:, i, :], cell) for i, cell in enumerate(self._cells)])
            self.trajectory_scaled = unwrapped
        else:
            trajectory = np.array([atoms.positions for atoms in structures])
            self.trajectory_scaled = np.hstack([a.cell.scaled_positions(a.positions)[:, None] for a in structures])
            
        self.trajectory = trajectory.swapaxes(0,1)



    def read_file(self, file, index = ':', unwrap = True,  atom_types_mapping = None, **kwargs):
        
        """
        Read VASP/LAMMPS generated XDATCAR/.xml/.lammpstrj file
        
        Parameters
        ----------
        
        file: str
            path to the file (XDATCAR, .xml)

        unwrap: boolean, True by default
            perform unwrapping of the coordinates
            
        kwargs: dict
            aditional parameters of ase.io.read method

        Examples
        --------

        >>> from mlyzed import Lyze
        >>> calc = Lyze()
        >>> calc.read_file('MD_file.traj', unwrap = True)
        
        """

        structures = read(file, index = index, **kwargs)
        self.read_atoms(structures, unwrap = unwrap, atom_types_mapping=atom_types_mapping)



    def read_files(self, files, index = ':', unwrap = True, atom_types_mapping = None, **kwargs):
        
        """
        Read VASP/lammps generated XDATCAR/.xml/.lammpstrj file
        
        Parameters
        ----------
        
        file: str
            path to the file (XDATCAR, .xml)

        unwrap: boolean, True by default
            perform unwrapping of the coordinates
            
        kwargs: dict
            aditional parameters of ase.io.read method

        Examples
        --------

        >>> from mlyzed import Lyze
        >>> calc = Lyze()
        >>> files = ['MD_file_1.traj', 'MD_file_2.traj']
        >>> calc.read_files(files)
        
        """
        structures = []
        for file in files:
            structures.extend(read(file, index = index, **kwargs))
        self.read_atoms(structures, unwrap = unwrap, atom_types_mapping=atom_types_mapping)


    def unwrap(self, structures, sequence = None):

        """ Unwrapper of the MD sequence of atomic coordinates subject 
        to periodic boundary conditions.

        Minimum image principle is used to unwrap coordinates. It means, 
        method can be applied for NVT ensemble, but use it with caution for NpT

        Parameters
        ----------

        sequence: None (be default) or np.array of shape (n_atoms, n_steps, n_dimension)
            sequence of fractional atomic positions wrapped into the periodic box
            self.traj will be used to generate sequence

        Returns
        ----------
        unwrapped: np.array of shape (n_atoms, n_steps, n_dimension)
            unwrapped sequence of coordinates

        """
        if not np.any(sequence):
            positions = [a.cell.scaled_positions(a.positions)[:, None] for a in structures]
            positions = np.hstack(positions)
        else:
            positions = sequence
        self._wrapped_trajectory = positions
        images_list = [np.expand_dims(np.zeros(positions[:, 0, :].shape), axis = 1)]
        for i in trange(len(positions[0, :]) - 1, desc = 'Unwrapping coordinates'):
            d = positions[:, i + 1, :] - positions[:, i, :]
            images = np.where(d < -0.5, 1, 0)
            images = np.where(d > 0.5, images - 1, images)
            images = np.expand_dims(images, axis = 1)
            images_list.append(images)
        images_list = np.hstack(images_list)
        shift = np.cumsum(images_list, axis = 1)
        unwrapped = positions + shift
        return unwrapped
        

    def classical_msd(self, specie = None, timestep = 1, correct_drift = True, projection = 'xyz', mean = True, skip = 0, com = False):
        
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
        ---------- 
        
        dt: np.array
            time in ps
        msd: np.array
            MSD

        Examples
        --------

        >>> from mlyzed import Lyze
        >>> calc = Lyze()
        >>> calc.read_file('MD_trajectory.traj')
        >>> dt, msd = calc.classical_msd(specie = 'Li', timestep = 2)
        
        """

        if projection not in self._projection_key_mapper.keys():
            raise
        traj = self.trajectory[:, skip:, :]
        specie_idx = np.argwhere(self.symbols == specie).ravel()
        framework_idx = np.argwhere(self.symbols != specie).ravel()
        disp = traj[:,0,:][:, None] - traj[:,:,:]
        if correct_drift:
            disp -= disp[framework_idx, :, :].mean(axis = 0)[None, :]
        disp = disp[specie_idx, :, :]
        disp_projected = disp[:, :, self._projection_key_mapper[projection]].reshape(disp.shape[0], disp.shape[1], len(projection))
        dt = timestep * np.arange(0, disp.shape[1]) / 1000
        if com:
            msd_com = np.square(np.linalg.norm(disp_projected.mean(axis = 0), axis = 1))
            return dt, msd_com
        msd = np.square(disp_projected).sum(axis = -1)
        if mean:
            msd = msd.mean(axis = 0)
        return dt, msd
    


    def fft_msd(self, specie = None, timestep = 1.0):

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
        ---------- 
        
        dt: np.array
            time in ps
        msd: np.array
            MSD

        Examples
        --------

        >>> from mlyzed import Lyze
        >>> calc = Lyze()
        >>> calc.read_file('MD_trajectory.traj')
        >>> dt, msd = calc.fft_msd(specie = 'Li', timestep = 2)
        
        """

        specie_idx = np.argwhere(self.symbols == specie).ravel()
        framework_idx = np.argwhere(self.symbols != specie).ravel()

        pos = self.trajectory[specie_idx,: , :]
        nTime=pos.shape[1]        

        S2 = np.sum ( np.fft.ifft( np.abs(np.fft.fft(pos, n=2*nTime, axis = -2))**2, axis = -2  )[:,:nTime,:].real , axis = -1 ) / (nTime-np.arange(nTime)[None,:] )

        D=np.square(pos).sum(axis=-1)
        D=np.append(D, np.zeros((pos.shape[0], 1)), axis = -1)
        S1 = ( 2 * np.sum(D, axis = -1)[:,None] - np.cumsum( np.insert(D[:,0:-1], 0, 0, axis = -1) + np.flip(D, axis = -1), axis = -1 ) )[:,:-1] / (nTime - np.arange(nTime)[None,:] )

        MSD = S1-2*S2

        Dt_r = np.arange(1, pos.shape[1]-1)
        MSD = MSD[:,Dt_r]
        dt = timestep * Dt_r / 1000
        msd = MSD.mean(axis = 0)
        return dt, msd


    def block_msd(self, specie = None, timestep = 1, skip = 0, n_blocks = 10):

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

        skip: int
            skip timesteps

        n_blocks: int, 10 by default
            size of the split in ps 
            
        Returns
        ---------- 
        
        dt: np.array
            time in ps
        msd_mean: np.array
            average MSD
        msd_std: np.array
            std of MSD
        

        Examples
        --------

        >>> from mlyzed import Lyze
        >>> calc = Lyze()
        >>> calc.read_file('MD_trajectory.traj')
        >>> dt, msd, msd_std = calc.block_msd(specie = 'Li', timestep = 2, n_blocks = 10)
        
        """
        specie_idx = np.argwhere(self.symbols == specie).ravel()
        framework_idx = np.argwhere(self.symbols != specie).ravel()

        dts, msds = [], []
        blocks = np.arange(skip, self.trajectory.shape[1], (self.trajectory.shape[1] - skip)// n_blocks)
        for start, stop in zip(blocks[0:-1], blocks[1:]):
            traj = self.trajectory[:, start:stop, :]
            disp = traj[:,0,:][:, None] - traj[:,:,:]
            disp -= disp[framework_idx, :, :].mean(axis = 0)[None, :]
            msd = np.square(disp[specie_idx, :, :]).sum(axis = -1).mean(axis = 0)
            dt = timestep * np.arange(0, len(msd)) / 1000
            dts.append(dt)
            msds.append(msd)
        return dts[0], np.mean(msds, axis = 0), np.std(msds, axis = 0)
    
        

    def windowed_msd(self, n_frames = 75, timestep = 1.0, specie = 'Na', N_bootstraps = 0):

        """ 
        Calculate windowed (time averaged) MSD for the selected specie. 
        Supposed to work the same way as MDAnalysis.


        Parameters
        ----------

        n_frames: int, 75 by default
            number of different lagtimes to calculated MSD
            lagtimes = np.linspace(1, number of steps - 1, n_frames)

        Returns
        ----------
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

        self.timestep = timestep

        specie_idx = np.argwhere(self.symbols == specie).ravel()
        framework_idx = np.argwhere(self.symbols != specie).ravel()


        lagtimes = np.round(np.linspace(10, self.trajectory.shape[1] - 1, n_frames))
        windows = dict()
        msds = []
        msds_std = []
        msds_all = []
        for lag in tqdm(lagtimes, desc = 'Getting MSD vs. lagtime'):
            lag = int(lag)
            disp = self.trajectory[:,:-lag,:] - self.trajectory[:,lag:,:]
            msds_by_specie = np.square(disp[specie_idx, :, :]).sum(axis = 2)
            msd = np.square(disp[specie_idx, :, :]).sum(axis = 2).mean(axis = 0)
            msd_mean = msd.mean()
            msd_std = msd.std()
            msds.append(msd_mean)
            msds_all.append(msd)
            msds_std.append(msd_std)
            windows.update({lag: msds_by_specie})

        #self.windows = windows
        msds = np.array(msds)
        msds_std = np.array(msds_std)
        dt = timestep * lagtimes / 1000

        if N_bootstraps:
            data = windows
            msds = np.zeros((N_bootstraps, len(data.keys())))
            msds_mean = []
            #dt = []
            for i, t in enumerate(data.keys()):
                msd = data[t].ravel()
                n_ind = data[t].shape[0] * int(np.floor(self.trajectory.shape[1] / t)) # N_atoms * non-overlapping trajectories
                msd_mean = msd.mean()
                resample = list()
                for _ in range(N_bootstraps):
                    resample.append(np.random.choice(msd, n_ind).mean())
                resample = np.array(resample)
                msds[:, i] = resample
                msds_mean.append(msd_mean)

        return dt, msds, msds_std, msds_all

    

    def probability_density(self, specie, resolution = 0.2):

        """
        Calculate a time-average probability density function of a selected specie.
        Warning! Works properly for NVT simulations only!

        Parameters
        ----------
        specie: str
            chemical symbol

        resolution: float
            grid resolution in angstroms

        Returns
        -------

        hist: np.array
            probability density distribution 

        Examples
        --------

        >>> from mlyzed import Lyze
        >>> from mlyzed.utils import write_grd
        >>> calc = Lyze()
        >>> calc.read_file('MD_trajectory.traj')
        >>> hist = calc.probability_density('Li', resolution = 0.2)
        >>> write_grd(hist, calc.atoms, 'proba_file.grd')

        """
        
        specie_idx = np.argwhere(self.symbols == specie).ravel()
        scaled_traj = self.trajectory_scaled[specie_idx, :, :]
        scaled_traj -=np.floor(scaled_traj)
        bins = list(map(int, (np.diag(self._cells[0]) / resolution)))
        voxels = (scaled_traj * bins).astype(int)
        hist = np.zeros((bins), np.uint16)
        for point in np.vstack(voxels):
            hist[point[0], point[1], point[2]] += 1
        hist = hist / hist.sum() * (hist.shape[0] * hist.shape[1] * hist.shape[2]) /  np.linalg.det(self._cells[-1])
        return hist