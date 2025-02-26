import numpy as np
import networkx as nx
from scipy.ndimage import uniform_filter1d
from vesin import NeighborList
from tqdm import tqdm



def moving_average(positions, n_steps):
    return uniform_filter1d(positions, n_steps, axis = 0, mode='constant', origin=-(n_steps//2))



def _reference_site(reference_frame, current_frame):

    ref_states = []
    for i, p in enumerate(current_frame.positions):
        base = reference_frame.copy()
        base.append('X')
        base.positions[-1] = p
        dd = base.get_distances(len(base)-1, np.arange(0, len(base)-1), mic = True)
        closest_site = np.argmin(dd)
        ref_states.append(closest_site)
    return ref_states



def _detect_hop_events(reference_frame, current_frame, cutoff):
    reference_state = np.arange(len(reference_frame))
    current_state = np.array(_reference_site(reference_frame, current_frame))
    hoppers = np.where(reference_state - current_state)[0]
    calc = NeighborList(cutoff = cutoff, full_list=True)

    i_ref, j_ref, d_ref = calc.compute(points = reference_frame.positions, box = reference_frame.cell, periodic = True, quantities='jid')
    i_cur, j_cur, d_cur = calc.compute(points = current_frame.positions, box = current_frame.cell, periodic = True, quantities='jid')
    hoppers_moving_neighbors = {}
    for hopper in hoppers:
        nn_list_reference = np.unique(j_ref[i_ref == hopper])
        nn_list_current = np.unique(j_cur[i_cur == hopper])
        moving_neighbors = list(set(nn_list_reference) & set(nn_list_current) & set(hoppers))
        moving_neighbors.append(hopper)
        moving_neighbors = np.array(moving_neighbors, dtype = int)
        hoppers_moving_neighbors.update({hopper: (moving_neighbors)})

    edgelist = []
    for key in hoppers_moving_neighbors.keys():
        edgelist.extend([(key, nn) for nn in hoppers_moving_neighbors[key]])

    G = nx.from_edgelist(edgelist)
    total_hops = {}
    for cc in nx.connected_components(G):
        hops = len(cc)
        if hops not in total_hops.keys():
            total_hops.update({hops: 1})
        else:
            total_hops[hops] += 1
    return total_hops




def hops_statistics(trajectory, specie = None, filter_step = 1, frame_step = 1, cutoff = 4.0):

    """
    Calculate statistics of the hop events

    Parameters:

    trajectory: mlyzed Trajectory
        trajectory

    specie: str
        specie of interest, e.g. "Na"

    filter_step: int
        uniform filter step

    frame_step: int
        calculate hope events each frame_step

    cutoff: float
        collect neighbors within cutoff to count cooperative migration

    start_id: int, 0 by default
        start analysis from the start_id

    Returns
    -------

    hops_stats: dict
        statistics of the hop events, where
        key -> number of simultaneous ions jumped
        values -> number of such events between sequential frames
    """

    traj = trajectory[:, trajectory.symbols == specie]
    if filter_step > 1:
        traj.trajectory = moving_average(traj.trajectory.copy(), filter_step)
        traj = traj[:-(filter_step-1)]

    n_steps = len(traj) // frame_step - 1
    hops_stats = {}
    for i in tqdm(range(n_steps)):
        reference_frame, current_frame = traj[i * frame_step].get_frame(0), traj[(i+1) * frame_step].get_frame(0)
        total_hops_per_step = _detect_hop_events(reference_frame, current_frame, cutoff)
        for key in total_hops_per_step.keys():
            if key not in hops_stats.keys():
                hops_stats.update({key: total_hops_per_step[key]})
            else:
                hops_stats[key] += total_hops_per_step[key]
    hops_stats.update({'total': sum(hops_stats.values())})
    return hops_stats

