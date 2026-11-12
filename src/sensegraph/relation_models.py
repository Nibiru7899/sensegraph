"""Relation predictors whose forward API accepts observable tensors only."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv

from .synthetic import RELATIONS


@dataclass
class RelationInputs:
    x: torch.Tensor
    candidate_index: torch.Tensor
    message_index: torch.Tensor


def mask_features(x, mode="all"):
    result = x.clone()
    if mode == "no_geometry":
        result[:, :8] = 0
    elif mode == "no_color":
        result[:, 8:] = 0
    return result


def pair_features(x, candidates):
    a, b = x[candidates[0]], x[candidates[1]]
    delta = b[:, 4:6] - a[:, 4:6]
    size_delta = b[:, 6:8] - a[:, 6:8]
    intersection = (torch.minimum(a[:, 2:4], b[:, 2:4]) - torch.maximum(a[:, :2], b[:, :2])).clamp_min(0).prod(1)
    area_a, area_b = a[:, 6:8].prod(1), b[:, 6:8].prod(1)
    iou = intersection / (area_a + area_b - intersection).clamp_min(1e-8)
    color = (a[:, 8:] - b[:, 8:]).square().sum(1).sqrt()
    return torch.cat((delta, delta.abs(), size_delta, iou[:, None], color[:, None]), dim=1)


def message_graph(x, connectivity="spatial", neighbors=6):
    n = x.shape[0]
    if connectivity == "complete":
        source, target = torch.where(~torch.eye(n, dtype=torch.bool, device=x.device))
        return torch.stack((source, target))
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    distance = torch.cdist(x[:, 4:6], x[:, 4:6])
    distance.fill_diagonal_(float("inf"))
    source = torch.argsort(distance, dim=1, stable=True)[:, :min(neighbors, n - 1)]
    target = torch.arange(n, device=x.device)[:, None].expand_as(source)
    return torch.stack((source.reshape(-1), target.reshape(-1)))


def pack_samples(samples, device="cpu", mode="all", connectivity="spatial", neighbors=6):
    xs, pairs, messages, labels, slices = [], [], [], [], []
    node_offset, edge_offset = 0, 0
    for sample in samples:
        x = mask_features(sample["inputs"]["x"], mode)
        candidate = sample["inputs"]["candidate_index"]
        graph = message_graph(x, "complete" if mode == "no_geometry" else connectivity, neighbors)
        xs.append(x)
        pairs.append(candidate + node_offset)
        messages.append(graph + node_offset)
        labels.append(sample["targets"]["edge_y"])
        slices.append((edge_offset, edge_offset + candidate.shape[1]))
        node_offset += len(x)
        edge_offset += candidate.shape[1]
    inputs = RelationInputs(torch.cat(xs).to(device), torch.cat(pairs, dim=1).to(device), torch.cat(messages, dim=1).to(device))
    return inputs, torch.cat(labels).to(device), slices


class RelationNet(nn.Module):
    def __init__(self, architecture, config):
        super().__init__()
        self.architecture = architecture
        hidden = config["hidden"]
        dropout = config["dropout"]
        if architecture == "mlp":
            self.project = nn.Sequential(nn.Linear(11, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
            head_hidden = config["mlp_head_hidden"]
        elif architecture == "gat":
            self.project = nn.Linear(11, hidden)
            self.layers = nn.ModuleList([GATv2Conv(hidden, hidden // config["heads"], heads=config["heads"], dropout=dropout, add_self_loops=True) for _ in range(config["layers"])])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.layers])
            head_hidden = config["gat_head_hidden"]
        else:
            raise ValueError("Unknown neural architecture")
        self.head = nn.Sequential(nn.Linear(hidden * 2 + 8, head_hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(head_hidden, hidden), nn.ReLU(), nn.Linear(hidden, len(RELATIONS)))

    def forward(self, inputs):
        h = self.project(inputs.x)
        if self.architecture == "gat":
            for layer, norm in zip(self.layers, self.norms):
                h = norm(h + F.elu(layer(h, inputs.message_index)))
        source, target = inputs.candidate_index
        features = torch.cat((h[source], h[target], pair_features(inputs.x, inputs.candidate_index)), dim=1)
        return self.head(features)


def rule_scores(inputs):
    """Fixed, deliberately approximate geometry/color rules; no oracle-type lookup."""
    x, pairs = inputs.x, inputs.candidate_index
    source, target = pairs
    a, b = x[source], x[target]
    af, bf = a[:, 6:8], b[:, 6:8]
    area_a, area_b = af.prod(1), bf.prod(1)
    saturation = (x[:, 8:].max(1).values - x[:, 8:].min(1).values)
    mark = (saturation > 0.07) & (x[:, 6] > 0.002) & (x[:, 7] > 0.002)
    tiny = (x[:, 6] < 0.035) & (x[:, 7] < 0.04)
    thin = (x[:, 6] < 0.012) | (x[:, 7] < 0.012)
    text_like = (x[:, 7] < 0.065) & (x[:, 6] > 0.015) & ~mark
    contains = (b[:, :2] <= a[:, :2] + 0.005).all(1) & (b[:, 2:4] >= a[:, 2:4] - 0.005).all(1)
    inside = (a[:, :2] <= b[:, 4:6] + 0.01).all(1) & (a[:, 2:4] >= b[:, 4:6] - 0.01).all(1)
    delta = b[:, 4:6] - a[:, 4:6]
    close_x, close_y = delta[:, 0].abs() < 0.025, delta[:, 1].abs() < 0.025
    close = delta.square().sum(1).sqrt() < 0.10
    similar_color = (a[:, 8:] - b[:, 8:]).square().sum(1).sqrt() < 0.12
    # Estimate a shared bar baseline from repeated visible rectangle boundaries.
    candidates = x[mark & (x[:, 6] > 0.025) & (x[:, 7] > 0.035)]
    anchor = x[:, 5].clone()
    if len(candidates) >= 3:
        boundaries = torch.cat((candidates[:, 1], candidates[:, 3]))
        rounded = (boundaries * 200).round().long()
        unique, counts = rounded.unique(return_counts=True)
        baseline = unique[counts.argmax()].float() / 200
        rect = mark & (x[:, 6] > 0.025) & (x[:, 7] > 0.035)
        top_distance, bottom_distance = (x[:, 1] - baseline).abs(), (x[:, 3] - baseline).abs()
        anchor[rect] = torch.where(top_distance > bottom_distance, x[:, 1], x[:, 3])[rect]
    scores = torch.zeros((pairs.shape[1], len(RELATIONS)), device=x.device)
    def assign(name, condition):
        scores[:, RELATIONS.index(name)] = condition.float()
    assign("labels", text_like[source] & thin[target] & close & (close_x | close_y))
    assign("belongs_to", (mark[source] & contains & (area_b > area_a * 1.25)) | (tiny[source] & thin[target] & (close_x | close_y)))
    assign("maps_to_axis", mark[source] & thin[target] & ((close_x & (bf[:, 1] > 0.2)) | (close_y & (bf[:, 0] > 0.2))))
    assign("maps_to_tick", mark[source] & tiny[target] & close_x)
    assign("connected_to", mark[source] & tiny[target] & mark[target] & inside & (area_a > area_b * 2))
    assign("next_in_series", mark[source] & mark[target] & similar_color & (delta[:, 0] > 0.01) & (delta[:, 0] < 0.30))
    both = mark[source] & mark[target]
    difference = anchor[target] - anchor[source]
    assign("greater_than", both & (difference > 0.004))
    assign("less_than", both & (difference < -0.004))
    assign("equal_to", both & (difference.abs() <= 0.004))
    return scores
