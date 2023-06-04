import torch

from benchmarks import PATH_DATASETS
from benchmarks.utils import train_runtime, Recorder, train_homo
from torch_geometric.nn import GCN
from torch_geometric.datasets import Reddit


class TrainBenchmark:
    timeout = 10 * 60  # 60 min

    def setup_cache(self):
        data = Reddit(root=PATH_DATASETS)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = GCN(in_channels=64, hidden_channels=64, num_layers=3)

        with Recorder() as r:
            loss = train_homo(model, loader, optimizer, device)

        t, peakmem = r.get_result()
        return t, peakmem, loss

    def track_time(self, results):
        return results[0]

    def track_peakmem(self, results):
        return results[1]

    def track_loss(self, results):
        return results[2]
