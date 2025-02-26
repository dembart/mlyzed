#### About

mlyzed is the python library for post-processing molecular dynamics (MD) simulations. The main features of the code are trajectory unwrapping, FFT calculation of the MSD, analysis of the hop events, and ease of use. 

<i>Note: The library is not guaranteed to be bug free. If you observe unexpected results, errors, please report  an issue at the github.</i>


For more details, see the [documentation](https://mlyzed.readthedocs.io/en/latest/). (Not ready yet)

#### Installation

```python
git clone https://github.com/dembart/mlyzed
cd mlyzed
pip install .
```
#### Minimum usage example

```python
import mlyzed as md
traj = md.Trajectory.from_file('traj.lammpstrj') # can be XDATCAR, OUTCAR, extxyz, etc

msd = md.classical_msd(traj[100:], # skip first 100 steps
                       specie = 'Li', # specie of interest
                       timestep = 2,  # in fs
                       )
msd.plot()
```


#### Alternatives:

Here are some alternatives and inspirations for this library (see below). You may find them better in some ways.
* [mdtraj](https://github.com/mdtraj/mdtraj) - a huge inspiration to this library
* [pymatgen-diffusion-analysis](https://github.com/materialsvirtuallab/pymatgen-analysis-diffusion)
* [kinisi](https://github.com/bjmorgan/kinisi)
* [MDAnalysis](https://www.mdanalysis.org/)
* [LLC_Membranes](https://github.com/shirtsgroup/LLC_Membranes)



