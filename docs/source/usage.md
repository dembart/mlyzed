#### Minimum usage exmaple

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