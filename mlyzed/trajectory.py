import numpy as np
from ase import Atoms
from ase.io import read, write
from tqdm import trange, tqdm 
from ase.data import atomic_masses, atomic_numbers, chemical_symbols



class Trajectory:
    
    def __init__(self, numbers = None, positions = None, cells = None):

        assert positions.ndim == 3
        self.numbers = numbers
        if np.any(numbers):
            self.symbols = np.array([chemical_symbols[n] for n in numbers])
        self.positions = positions
        self.cells = cells



    @classmethod
    def from_atoms_list(cls, atoms_list, unwrap = True):

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

        >>> import mlyzed as md
        >>> from ase.io import read
        >>> atoms_list = read('MD_file.traj', index = ':')
        >>> traj = md.Trajectory.from_atoms_list(atoms_list)
        >>> del atoms_list

        """
        numbers = np.array(atoms_list[0].numbers)
        cells = np.array([st.cell for st in atoms_list])
        if unwrap:
            unwrapped = cls.unwrap(atoms_list)
            positions = np.array([np.dot(unwrapped[:, i, :], cell) for i, cell in enumerate(cells)])
        else:
            positions = np.array([atoms.positions for atoms in atoms_list])
        return cls(numbers = numbers, positions = positions, cells = cells)
        

    
    @staticmethod
    def unwrap(atoms_list, sequence = None):

        """
        Unwrapper of the MD sequence of atomic coordinates subject 
        to periodic boundary conditions.

        Minimum image principle is used to unwrap coordinates. It means, 
        method can be applied for NVT ensemble, but use it with caution for NpT

        Parameters
        ----------

        sequence: None (be default) or np.array of shape (n_atoms, n_steps, n_dimension)
            sequence of fractional atomic positions wrapped into the periodic box
            self.traj will be used to generate sequence

        Returns
        -------
        unwrapped: np.array of shape (n_atoms, n_steps, n_dimension)
            unwrapped sequence of coordinates

        """
        if not np.any(sequence):
            positions = [a.cell.scaled_positions(a.positions)[:, None] for a in atoms_list]
            positions = np.hstack(positions)
        else:
            positions = sequence
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
    


    def calculate_com(self):
        """Calculate center of mass (COM).
        
        Returns
        -------
        
        com: np.array of (N_frames, 3) shape
            COM position evolution 

        """
        masses = np.array([atomic_masses[atomic_numbers[s]] for s in self.symbols])
        total_mass = np.sum(masses)
        com = np.sum(self.positions * masses[np.newaxis, :, np.newaxis], axis=1) / total_mass
        return com



    def correct_com_drift(self):
        """Correct drift of the center of mass. In place operation"""
        com = self.calculate_com()
        self.positions -= com[:, np.newaxis, :]



    def map_atomic_types(self, atomic_types_mapper):
        
        """
        Update atomic types of the system.

        Parameters
        ----------
        atomic_types_mapper: dict 
            old atomic numbers are keys, new atomic numbers are values
        
        Examples
        --------

        >>> import mlyzed as md
        >>> traj = md.Trajectory.from_file('traj.lammpstrj')
        >>> traj.map_atomic_types({0: 3, 1: 8, 2: 15})

        """
        u,inv = np.unique(self.numbers,return_inverse = True)
        self.numbers = np.array([atomic_types_mapper[x] for x in u])[inv].reshape(self.numbers.shape)
        self.symbols = np.array([chemical_symbols[n]for n in self.numbers])



    def number_density(self, specie):
        """
        Calculate number density of a given specie
        
        Update atomic types of the system

        Parameters
        ----------
        specie: str
            chemical symbol, e.g. "Li"
        
        Returns
        -------
        number density at each time step measured in cm^-3
        Examples
        --------

        >>> import mlyzed as md
        >>> traj = md.Trajectory.from_file('file.traj')
        >>> n_mean = traj.number_density('Na').mean()

        """
        return (self.symbols == specie).sum() / np.linalg.det(self.cells) * 1e24



    @classmethod
    def from_file(cls, file):

        """
        Read trajectory file.

        Parameters
        ----------
        file: str
            trajectory fil
        
        Returns
        -------

        trajectory: mlyzed.Trajectory object
            trajectory

        Examples
        --------

        >>> import mlyzed as md
        >>> traj = md.Trajectory.from_file('traj.lammpstrj')

        """
        atoms_list = read(file, index = ':')
        return cls.from_atoms_list(atoms_list)
    
    @classmethod
    def from_files(cls, files):
        atoms_list = []
        for file in files:
            atoms_list.extend(read(file, index = ':'))
        return cls.from_atoms_list(atoms_list)
    
    def to_file(self, file):
        atoms_list = []
        for i in tqdm(range(self.n_frames)):
            atoms_list.append(self.get_frame(i))
        write(file, atoms_list)
    
    @property
    def n_atoms(self):
        return self.positions.shape[1]
    
    @property
    def n_frames(self):
        return self.positions.shape[0]
    
    def get_frame(self, index):
        cell = self.cells[index]
        positions = self.positions[index]
        numbers = self.numbers
        return Atoms(numbers = numbers, positions = positions, cell = cell, pbc = True)
    
    
    def copy(self):
        """Get copy of the trajectory"""
        numbers = self.numbers.copy()
        positions = self.positions.copy()
        cells = self.cells.copy()
        return self.__class__(numbers = numbers, positions = positions, cells = cells)


    def __getitem__(self, key):

        # Parse the key into frame, atom, and coordinate parts
        if isinstance(key, tuple):
            frame_key = key[0] if len(key) >= 1 else slice(None)
            atom_key = key[1] if len(key) >= 2 else slice(None)
            coord_key = key[2] if len(key) >= 3 else slice(None)
        else:
            frame_key = key
            atom_key = slice(None)
            coord_key = slice(None)

        new_positions = self.positions.copy()
        if isinstance(frame_key, int):
            new_positions = np.take(new_positions, [frame_key], axis=0)
        else:
            new_positions = new_positions[frame_key, :, :]

        if isinstance(atom_key, int):
            new_positions = np.take(new_positions, [atom_key], axis=1)
        else:
            new_positions = new_positions[:, atom_key, :]

        if isinstance(coord_key, int):
            new_positions = np.take(new_positions, [coord_key], axis=2)
        else:
            new_positions = new_positions[:, :, coord_key]

        if isinstance(frame_key, int):
            new_cells = np.take(self.cells.copy(), [frame_key], axis=0)
        else:
            new_cells = self.cells.copy()[frame_key]

        if isinstance(atom_key, int):
            new_numbers = self.numbers.copy()[[atom_key]]  # Keep as array to preserve dimension
        else:
            new_numbers = self.numbers.copy()[atom_key]
        new_numbers = np.atleast_1d(new_numbers)
        return self.__class__(new_numbers, new_positions, new_cells)
    

    def __repr__(self):
        return f'{self.positions.shape}'

    def __len__(self):
        return self.n_frames
    
