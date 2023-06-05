import torch

from benchmarks.utils import benchmark

WITH_MAP_INDEX = True
try:
    from torch_geometric.utils.map import map_index
except:
    map_index = object
    WITH_MAP_INDEX = False


def trivial_map(src, index, max_index, inclusive):
    if max_index is None:
        max_index = max(src.max(), index.max())

    if inclusive:
        assoc = src.new_empty(max_index + 1)
    else:
        assoc = src.new_full((max_index + 1, ), -1)
    assoc[index] = torch.arange(index.numel(), device=index.device)
    out = assoc[src]

    if inclusive:
        return out, None
    else:
        mask = out != -1
        return out[mask], mask


class TrivialMap:

    def setup_cache(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        src = torch.randint(0, 100_000_000, (100_000, ), device=device)
        index = src.unique()
        return src, index

    def track_trivial_map_inclusive(self, output):
        src, index = output
        ts = benchmark(
            funcs=[trivial_map],
            func_names=['trivial_map'],
            args=(src, index, None, True),
            num_steps=100,
            num_warmups=50,
        )
        return ts[0][1]

    def track_trivial_map_exclusive(self, output):
        src, index = output
        ts = benchmark(
            funcs=[trivial_map],
            func_names=['trivial_map'],
            args=(src, index[:50_000], None, False),
            num_steps=100,
            num_warmups=50,
        )
        return ts[0][1]


class MapIndex:
    def setup(self):
        if not WITH_MAP_INDEX:
            raise NotImplementedError

    def setup_cache(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        src = torch.randint(0, 100_000_000, (100_000, ), device=device)
        index = src.unique()
        return src, index

    def track_map_index_inclusive(self, output):
        src, index = output
        ts = benchmark(
            funcs=[map_index],
            func_names=['map_index'],
            args=(src, index, None, True),
            num_steps=100,
            num_warmups=50,
        )
        return ts[0][1]

    def track_map_index_exclusive(self, output):
        src, index = output
        ts = benchmark(
            funcs=[map_index],
            func_names=['max_index'],
            args=(src, index[:50_000], None, False),
            num_steps=100,
            num_warmups=50,
        )
        return ts[0][1]
