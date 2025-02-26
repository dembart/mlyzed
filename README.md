#### About

mlyzed is the python library for post-processing molecular dynamics (MD) simulations. The main features of the code are trajectory unwrapping, FFT calculation of the MSD, analysis of the hop events, and ease of use. 

<i>Note: The library is not guaranteed to be bug free. If you observe unexpected results, errors, please report  an issue at the github.</i>


For more details, see the [documentation](https://mlyzed.readthedocs.io/en/latest/). (Not ready yet)

#### Installation

```
pip install mlyzed
```

or

```python
git clone https://github.com/dembart/mlyzed
cd mlyzed
pip install .
```
#### Minimum usage example

```python
import mlyzed as md

traj = md.Trajectory.from_file('traj.lammpstrj') # any ASE readable format
msd = md.classical_msd(traj[100:], specie = 'Li', timestep = 2)
msd.plot()
```


#### Alternatives:

Here are some alternatives and inspirations for this library (see below). You may find them better in some ways.
* [mdtraj](https://github.com/mdtraj/mdtraj) - a huge inspiration to this library
* [pymatgen-diffusion-analysis](https://github.com/materialsvirtuallab/pymatgen-analysis-diffusion)
* [kinisi](https://github.com/bjmorgan/kinisi)
* [MDAnalysis](https://www.mdanalysis.org/)
* [LLC_Membranes](https://github.com/shirtsgroup/LLC_Membranes)



