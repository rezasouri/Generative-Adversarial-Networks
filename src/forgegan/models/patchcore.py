"""PatchCore-style local feature memory bank for industrial anomaly detection."""

from __future__ import annotations

from typing import Any

import numpy as np


class ResNetFeatureExtractor:
    """ImageNet ResNet-18 layer2/layer3 patch embeddings."""

    def __init__(self, *, pretrained: bool = True, device: str = "cpu") -> None:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        self.model.eval().to(device)
        self.device = device
        self._features: dict[str, Any] = {}
        self.model.layer2.register_forward_hook(
            lambda _module, _inputs, output: self._features.__setitem__("layer2", output)
        )
        self.model.layer3.register_forward_hook(
            lambda _module, _inputs, output: self._features.__setitem__("layer3", output)
        )
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], device=device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], device=device
        ).view(1, 3, 1, 1)

    @property
    def embedding_dim(self) -> int:
        return 256 + 512

    @property
    def weights(self) -> str:
        return "imagenet1k_v1" if self.model.fc.weight.shape[0] == 1000 else "random"

    @property
    def device_name(self) -> str:
        return self.device

    def __call__(self, images: Any) -> Any:
        import torch.nn.functional as functional

        images = (images - self.mean) / self.std
        self._features.clear()
        with torch.no_grad():
            self.model(images)
            shallow = self._features["layer2"]
            deep = self._features["layer3"]
            deep = functional.interpolate(
                deep, size=shallow.shape[-2:], mode="bilinear", align_corners=False
            )
            embedding = torch_cat((shallow, deep))
            embedding = functional.avg_pool2d(
                embedding, kernel_size=3, stride=1, padding=1
            )
        return embedding


def torch_cat(values: tuple[Any, ...]) -> Any:
    """Small local indirection to keep torch import optional at module import."""
    import torch

    return torch.cat(values, dim=1)


def greedy_coreset(
    embeddings: np.ndarray,
    *,
    max_points: int = 2048,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic farthest-first coreset selection on a bounded candidate set."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("embeddings must be a nonempty finite matrix")
    if type(max_points) is not int or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if len(values) <= max_points:
        return values.copy()

    rng = np.random.default_rng(seed)
    candidate_count = min(len(values), max(max_points * 10, 10000))
    if candidate_count < len(values):
        candidates = np.sort(rng.choice(len(values), candidate_count, replace=False))
        values = values[candidates]
    norms = np.sum(values * values, axis=1)
    center = values.mean(axis=0, keepdims=True)
    min_distance = np.maximum(
        norms + float(np.sum(center * center)) - 2 * (values @ center.T).ravel(), 0
    )
    selected = np.empty(max_points, dtype=np.int64)
    for index in range(max_points):
        farthest = int(np.argmax(min_distance))
        selected[index] = farthest
        delta = values - values[farthest]
        distance = np.einsum("ij,ij->i", delta, delta)
        min_distance = np.minimum(min_distance, distance)
        min_distance[farthest] = -1
    return values[selected]


def nearest_neighbor_scores(
    queries: np.ndarray,
    memory_bank: np.ndarray,
    *,
    query_batch_size: int = 1024,
    memory_batch_size: int = 4096,
) -> np.ndarray:
    """Exact chunked L2 nearest-neighbor distances with bounded temporary memory."""
    query = np.asarray(queries, dtype=np.float32)
    memory = np.asarray(memory_bank, dtype=np.float32)
    if query.ndim != 2 or memory.ndim != 2 or query.shape[1] != memory.shape[1]:
        raise ValueError("query and memory embeddings must be compatible matrices")
    if not len(query) or not len(memory) or not np.isfinite(query).all() or not np.isfinite(memory).all():
        raise ValueError("query and memory embeddings must be nonempty and finite")
    if query_batch_size < 1 or memory_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    memory_norm = np.sum(memory * memory, axis=1)
    output = np.empty(len(query), dtype=np.float32)
    for q_start in range(0, len(query), query_batch_size):
        q = query[q_start : q_start + query_batch_size]
        q_norm = np.sum(q * q, axis=1, keepdims=True)
        best = np.full(len(q), np.inf, dtype=np.float32)
        for m_start in range(0, len(memory), memory_batch_size):
            bank = memory[m_start : m_start + memory_batch_size]
            distances = (
                q_norm + memory_norm[m_start : m_start + memory_batch_size][None, :]
                - 2 * (q @ bank.T)
            )
            np.maximum(distances, 0, out=distances)
            best = np.minimum(best, distances.min(axis=1))
        output[q_start : q_start + len(q)] = np.sqrt(best)
    return output
