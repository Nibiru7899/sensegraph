"""Untrained architecture skeleton: tensor verification only, never loads weights.

Proposals are explicit observable inputs at the discrete decoder boundary.
No renderer annotations, training loop, OCR decoder or numerical executor is hidden here.
"""
from collections import OrderedDict
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision.ops import FeaturePyramidNetwork, roi_align
from torch_geometric.nn import GATv2Conv
from .relation_models import message_graph, pair_features


def conv_block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.ReLU())


def dense_head(out):
    return nn.Sequential(conv_block(128, 128), nn.Conv2d(128, out, 1))


class CropCTCRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(conv_block(1, 32, (2, 2)), conv_block(32, 64, (2, 2)),
                                 conv_block(64, 128, (2, 1)), conv_block(128, 128, (2, 1)))
        self.gru = nn.GRU(128, 64, bidirectional=True, batch_first=True)
        self.output = nn.Linear(128, 96)

    def forward(self, crops):
        assert crops.ndim == 4 and crops.shape[1:] == (1, 32, 128)
        if len(crops) == 0:
            return crops.new_empty((0, 32, 96))
        x = self.cnn(crops)
        assert x.shape[1:] == (128, 2, 32)
        return self.output(self.gru(x.mean(2).transpose(1, 2))[0])


class CharacterBiGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(98, 64, padding_idx=97)
        self.gru = nn.GRU(64, 64, bidirectional=True, batch_first=True)

    def forward(self, tokens, lengths):
        assert tokens.ndim == 2 and tokens.shape[1] == 128
        assert lengths.shape == (len(tokens),) and ((lengths > 0) & (lengths <= 128)).all()
        if not len(tokens):
            return self.embedding.weight.new_empty((0, 128))
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return torch.cat((h[0], h[1]), -1)

    def recognized_embedding(self, ids, available):
        assert ids.ndim == 2 and available.shape == (len(ids),)
        mask = (ids >= 1) & (ids <= 95)
        embeddings = self.embedding(ids)
        pooled = (embeddings * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        return pooled * available[:, None]


class GraphStack128(nn.Module):
    def __init__(self, architecture="gat"):
        super().__init__()
        if architecture not in ("gat", "mlp"):
            raise ValueError("Expected gat or mlp")
        self.architecture = architecture
        self.layers = nn.ModuleList([
            GATv2Conv(128, 32, heads=4, concat=True, dropout=.15, edge_dim=8,
                       negative_slope=.2, share_weights=False, add_self_loops=True,
                       fill_value=0., bias=True, residual=False)
            if architecture == "gat" else nn.Sequential(nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128))
            for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(3)])

    def forward(self, h, messages, attributes):
        assert h.ndim == 2 and h.shape[1] == 128
        assert messages.shape[0] == 2 and attributes.shape == (messages.shape[1], 8)
        if not len(h):
            return h
        for layer, norm in zip(self.layers, self.norms):
            update = layer(h, messages, attributes) if self.architecture == "gat" else layer(h)
            h = norm(h + F.elu(update))
        return h


class QuestionNodeReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(128, 4, dropout=.1, batch_first=True)
        self.operation = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 7))
        self.projections = nn.ModuleList([nn.Linear(128, 128, bias=False) for _ in range(5)])
        self.null = nn.Parameter(torch.zeros(128))

    def forward(self, q, h, node_chart, question_chart, batch_size):
        assert q.shape == (len(question_chart), 128) and node_chart.shape == (len(h),)
        assert ((question_chart >= 0) & (question_chart < batch_size)).all()
        k = max(1, max((int((node_chart == b).sum()) for b in range(batch_size)), default=0))
        padded = h.new_zeros((batch_size, k, 128))
        valid = torch.zeros((batch_size, k), dtype=torch.bool, device=h.device)
        for b in range(batch_size):
            selected = h[node_chart == b]
            padded[b, :len(selected)] = selected
            valid[b, :len(selected)] = True
        nodes, mask = padded[question_chart], valid[question_chart]
        # Always-valid null key prevents all-masked softmax on empty charts.
        null = self.null.expand(len(q), 1, 128)
        keys = torch.cat((nodes, null), 1)
        key_mask = torch.cat((~mask, torch.zeros((len(q), 1), dtype=torch.bool, device=h.device)), 1)
        if len(q):
            context = self.attention(q[:, None], keys, keys, key_padding_mask=key_mask, need_weights=False)[0][:, 0]
        else:
            context = q
        qa, qb, operand_key, eq, ek = self.projections
        pointer_a = torch.einsum("qd,qkd->qk", qa(q), operand_key(keys)) / math.sqrt(128)
        pointer_b = torch.einsum("qd,qkd->qk", qb(q), operand_key(keys)) / math.sqrt(128)
        evidence = torch.einsum("qd,qkd->qk", eq(q), ek(nodes)) / math.sqrt(128)
        return {"operation_logits": self.operation(torch.cat((q, context), -1)),
                "operand_a_logits": pointer_a.masked_fill(key_mask, -torch.inf),
                "operand_b_logits": pointer_b.masked_fill(key_mask, -torch.inf),
                "evidence_logits": evidence.masked_fill(~mask, -torch.inf),
                "node_valid": mask, "unsupported_empty_chart": ~mask.any(1)}


class SenseGraphNet(nn.Module):
    """Forward skeleton across the explicitly supplied discrete-proposal boundary.

    images: ImageNet-normalized RGB; rgb: unnormalized [0,1] pixels.
    boxes: per-image predicted, detached xyxy tensors in padded-image pixels.
    probabilities: [N,7]; confidence [N,1]; recognized_ids [N,L]; available [N].
    All parameters are RANDOM, including ResNet. This is not learned inference.
    """
    def __init__(self, architecture="gat", connectivity="spatial"):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], 128)
        self.fuse = conv_block(128, 128)
        self.heads = nn.ModuleDict({k: dense_head(v) for k, v in {
            "center": 5, "size": 10, "offset": 10, "keypoint": 2,
            "key_offset": 4, "key_extent": 4, "segmentation": 2}.items()})
        self.node_fuse = nn.Sequential(nn.Linear(212, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.15))
        self.graph = GraphStack128(architecture)
        self.relations = nn.Sequential(nn.Linear(264, 256), nn.ReLU(), nn.Dropout(.15),
                                       nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
        self.ocr = CropCTCRecognizer()
        self.questions = CharacterBiGRU()
        self.reasoner = QuestionNodeReasoner()
        self.connectivity = connectivity
        self.apply(self._initialize)
        for name in ("center", "keypoint"):
            nn.init.constant_(self.heads[name][-1].bias, math.log(.01/.99))
        for name in ("size", "key_extent"):
            nn.init.constant_(self.heads[name][-1].bias, math.log(math.expm1(4.)))
        nn.init.zeros_(self.questions.embedding.weight[97])
        self.train()

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, value in module.named_parameters():
                if "weight_hh" in name:
                    for gate in value.chunk(3):
                        nn.init.orthogonal_(gate)
                elif "weight_ih" in name:
                    nn.init.xavier_uniform_(value)
                else:
                    nn.init.zeros_(value)

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def set_stage(self, stage):
        """Exact freeze audit policy; segmentation stays inactive without new masks."""
        if stage not in ("warmup", "perception", "ocr", "structural", "reasoning", "joint"):
            raise ValueError(stage)
        for p in self.parameters():
            p.requires_grad_(False)
        groups = {
            "warmup": [self.fpn, self.fuse, self.heads],
            "perception": [self.backbone.layer3, self.backbone.layer4, self.fpn, self.fuse, self.heads],
            "ocr": [self.ocr],
            "structural": [self.node_fuse, self.graph, self.relations],
            "reasoning": [self.questions, self.reasoner],
            "joint": [self.node_fuse, self.graph, self.relations, self.questions, self.reasoner],
        }
        for module in groups[stage]:
            for p in module.parameters():
                p.requires_grad_(True)
        for p in self.heads["segmentation"].parameters():
            p.requires_grad_(False)
        if stage in ("perception",):
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for p in module.parameters():
                        p.requires_grad_(False)

    def encode_image(self, images):
        assert images.ndim == 4 and images.shape[1] == 3 and images.shape[-1] % 32 == images.shape[-2] % 32 == 0
        m = self.backbone
        x = m.maxpool(m.relu(m.bn1(m.conv1(images))))
        features = OrderedDict()
        for name, layer in zip(("p2", "p3", "p4", "p5"), (m.layer1, m.layer2, m.layer3, m.layer4)):
            x = layer(x); features[name] = x
        pyramid = self.fpn(features)
        fused = self.fuse(torch.stack([F.interpolate(p, size=pyramid["p2"].shape[-2:], mode="bilinear", align_corners=False)
                                       for p in pyramid.values()]).mean(0))
        return pyramid, {name: head(fused) for name, head in self.heads.items()}

    def forward(self, images, rgb, boxes, probabilities, confidence, recognized_ids, available,
                question_ids, lengths, question_chart, ocr_crops=None):
        assert rgb.shape == images.shape and len(boxes) == len(images)
        boxes = [b.detach() for b in boxes]
        n = sum(len(b) for b in boxes)
        assert probabilities.shape == (n, 7) and confidence.shape == (n, 1) and available.shape == (n,)
        assert recognized_ids.ndim == 2 and len(recognized_ids) == n
        pyramid, dense = self.encode_image(images)
        appearance = roi_align(pyramid["p2"], boxes, (7, 7), spatial_scale=.25,
                               sampling_ratio=2, aligned=True).mean((-1, -2))
        xs, charts, messages, candidates = [], [], [], []
        offset = 0
        height, width = images.shape[-2:]
        for b, proposed in enumerate(boxes):
            assert proposed.ndim == 2 and proposed.shape[1] == 4 and len(proposed) <= 128
            assert torch.isfinite(proposed).all() and (proposed[:, :2] >= 0).all()
            assert (proposed[:, 2:] <= proposed.new_tensor([width, height])).all()
            assert (proposed[:, 2:] > proposed[:, :2]).all()
            rows = []
            for box in proposed:
                xy = box / box.new_tensor([width, height, width, height])
                left, top = box[:2].floor().long(); right, bottom = box[2:].ceil().long()
                color = rgb[b, :, top:bottom, left:right].mean((1, 2))
                rows.append(torch.cat((xy, (xy[:2]+xy[2:])/2, xy[2:]-xy[:2], color)))
            x = torch.stack(rows) if rows else images.new_empty((0, 11))
            xs.append(x); charts.append(torch.full((len(x),), b, dtype=torch.long, device=images.device))
            messages.append(message_graph(x, self.connectivity) + offset)
            candidates.append(message_graph(x, "complete") + offset)
            offset += len(x)
        x = torch.cat(xs); node_chart = torch.cat(charts)
        message = torch.cat(messages, 1); candidate = torch.cat(candidates, 1)
        text = self.questions.recognized_embedding(recognized_ids, available)
        h = self.node_fuse(torch.cat((x, probabilities.detach(), confidence.detach(), appearance, text, available[:, None]), -1))
        h = self.graph(h, message, pair_features(x, message))
        chunks = [self.relations(torch.cat((h[c[0]], h[c[1]], pair_features(x, c)), -1)) for c in candidate.split(4096, 1)]
        logits = torch.cat(chunks) if chunks else h.new_empty((0, 10))
        q = self.questions(question_ids, lengths)
        return {"dense": dense, "nodes": h, "observable": x, "node_chart": node_chart,
                "messages": message, "candidates": candidate, "relation_logits": logits,
                "reasoning": self.reasoner(q, h, node_chart, question_chart, len(images)),
                "ocr_logits": self.ocr(ocr_crops) if ocr_crops is not None else None}
