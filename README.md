#### About

mlyzed is a small library for post-processing molecular dynamics (MD) simulations. The main features of the code are trajectory unwrapping, FFT calculation of the MSD, and ease of use. 

<i>Note: The library is not guaranteed to be bug free. If you observe unexpected results, errors, please report  an issue at the github.</i>


For more details, see the [documentation](https://mlyzed.readthedocs.io/en/latest/).

#### Installation

```python
git clone https://github.com/dembart/mlyzed
cd mlyzed
pip install .
```
#### Minimum usage example

```python
from mlyzed import Lyze
calc = Lyze()
traj = calc.read_file('traj.lammpstrj') # can be XDATCAR, OUTCAR, extxyz
dt, msd = calc.classical_msd(
                            specie = 'Li', # specie of interest
                            timestep = 2,  # in fs
                            skip = 1000,   # skip first 1000 steps
                            )
```


#### Alternatives:

Here are some alternatives and inspirations for this library (see below). You may find them better in some ways.

* [pymatgen-diffusion-analysis](https://github.com/materialsvirtuallab/pymatgen-analysis-diffusion)  
* [kinisi](https://github.com/bjmorgan/kinisi)
* [MDAnalysis](https://www.mdanalysis.org/)  
* [LLC_Membranes](https://github.com/shirtsgroup/LLC_Membranes)  



