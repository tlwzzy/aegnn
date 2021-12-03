import numba
import numpy as np
from torch_geometric.nn.pool import radius_graph
import h5py
import torch
import time


@numba.jit(nopython=True)
def _fill_event_queues(sample, event_queues, events_per_pixel):
    max_num_events_per_queue = event_queues.shape[0]
    for i, (x, y, _) in enumerate(sample):
        queue_idx = events_per_pixel[y, x]
        if queue_idx < max_num_events_per_queue:            
            event_queues[queue_idx, y, x] = i
            events_per_pixel[y, x] += 1

@numba.jit(nopython=True)
def _perform_radius_search(sample, event_queues, events_per_pixel, connections, r, delta_t_us, max_num_neighbors, edges):
    height, width = events_per_pixel.shape
    num_edges = 0
    for i, (x, y, t) in enumerate(sample):
        if connections[i] >= max_num_neighbors:
            continue

        for r_idx in range(2*r+1):
            for c_idx in range(2*r+1):
                x_idx = x + c_idx - r
                y_idx = y + r_idx - r

                # skip if out of fov
                if not (0 <= x_idx and 0 <= y_idx and x_idx < width and y_idx < height):
                    continue

                for queue_idx in range(events_per_pixel[y_idx, x_idx]):
                    # this avoids connecting nodes twice or connecting to itself
                    # and avoids adding more connections than allowed.
                    j = event_queues[queue_idx, y_idx, x_idx]
                    if connections[j] >= max_num_neighbors or i <= j:
                        continue

                    t_nb = sample[j, 2]

                    # do not double-count edges
                    if abs(t - t_nb) < delta_t_us:
                        connections[i] += 1
                        connections[j] += 1
                        edges[num_edges,:] = [i,j]
                        num_edges += 1

    return edges[:num_edges]

def radius_graph_numba(sample, r, delta_t_us, max_num_neighbors, return_intermediate_results=False):
    MAX_QUEUE_SIZE = 32
    width, height, _ = sample.max(0).astype("int16")+1

    event_queues = np.zeros((MAX_QUEUE_SIZE, height, width), dtype="uint16")
    events_per_pixel = np.zeros((height, width), dtype="uint8")
    _fill_event_queues(sample, event_queues, events_per_pixel)

    connections_per_event = np.zeros(shape=(len(sample),), dtype="uint8")
    edges = np.zeros((max_num_neighbors*len(sample)//2, 2), dtype="uint16")
    edges = _perform_radius_search(sample,
                                   event_queues,
                                   events_per_pixel,
                                   connections_per_event,
                                   r,
                                   delta_t_us,
                                   max_num_neighbors,
                                   edges)

    if return_intermediate_results:
        return edges, {
            "connections_per_event": connections_per_event,
            "event_queues": event_queues,
            "events_per_pixel": events_per_pixel
        }
    else:
        return edges


r = 3
delta_t_us = 600000
max_num_neighbors = 128

events = np.load("events.npy")
events_xyt = events[:,:3]


# rescale time to be similar to pixels
points = torch.Tensor(events_xyt.astype("float32"))
points[:,2] = (points[:,2] - points[0,2]) * 5e-6


print("Go radius_graph_pytorch!")
N = 10
start = time.time()
for _ in range(N):
    edges = radius_graph(points, r=r, max_num_neighbors=max_num_neighbors)
end = time.time()
print(f"Used {end - start} seconds, {(end - start) / N} seconds per loop.")

print("Compiling radius_graph_numba...")
edges, data = radius_graph_numba(events_xyt, r=r, max_num_neighbors=max_num_neighbors, delta_t_us=delta_t_us, return_intermediate_results=True)

print("Go radius_graph_numba!")
start = time.time()
for _ in range(N):
    edges = radius_graph_numba(events_xyt, r=r, max_num_neighbors=max_num_neighbors, delta_t_us=delta_t_us)
end = time.time()
print(f"Used {end - start} seconds, {(end - start) / N} seconds per loop.")


