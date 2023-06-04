import argparse
import ast
import gc
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Union

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv
from torch_geometric.profile import rename_profile_file, timeit, torch_profile
from tqdm import tqdm

from benchmark.utils import (emit_itt, get_dataset, get_model, get_split_masks,
                             save_benchmark_data, test, write_to_csv)

supported_sets = {
    'ogbn-mag': ['rgat', 'rgcn'],
    'ogbn-products': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
    'Reddit': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
}


# https://github.com/pyg-team/pytorch_geometric/blob/2.3.1/benchmark/runtime/train.py
# https://github.com/akihironitta/lightning-benchmarks/blob/adbe9ab/benchmarks/common_utils.py
def train_runtime(
    model: torch.nn.Module,
    data: torch_geometric.data.Data,
    device: Union[str, torch.device],
    num_epochs: int,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model = model.to(device)
    data = data.to(device)
    model.train()
    mask = data.train_mask if 'train_mask' in data else data.train_idx
    y = data.y[mask] if 'train_mask' in data else data.train_y

    with Recorder() as r:
        for _ in range(num_epochs):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[mask], y)
            loss.backward()
            optimizer.step()

    t, peakmem = r.get_result()
    return t, peakmem, loss


class Recorder:
    def __init__(self) -> None:
        self.t0 = None
        self.t1 = None
        self.peakmem = None

    def __enter__(self):
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_accumulated_memory_stats()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.t1 = time.perf_counter()

        if torch.cuda.is_available():
            self.peakmem = torch.cuda.max_memory_allocated()

    def get_result(self):
        return self.t1 - self.t0, self.peakmem


def train_homo(model, loader, optimizer, device, trim=False):
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        if hasattr(batch, 'adj_t'):
            edge_index = batch.adj_t
        else:
            edge_index = batch.edge_index
        if not trim:
            out = model(batch.x, edge_index)
        else:
            out = model(
                batch.x,
                edge_index,
                num_sampled_nodes_per_hop=batch.num_sampled_nodes,
                num_sampled_edges_per_hop=batch.num_sampled_edges,
            )
        batch_size = batch.batch_size
        out = out[:batch_size]
        target = batch.y[:batch_size]
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()


def train_hetero(model, loader, optimizer, device, trim=False):
    if trim:
        warnings.warn("Trimming not yet implemented for heterogeneous graphs")

    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        if len(batch.adj_t_dict) > 0:
            edge_index_dict = batch.adj_t_dict
        else:
            edge_index_dict = batch.edge_index_dict
        out = model(batch.x_dict, edge_index_dict)
        batch_size = batch['paper'].batch_size
        out = out['paper'][:batch_size]
        target = batch['paper'].y[:batch_size]
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()
