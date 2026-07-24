from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbbox_overlaps


@MODELS.register_module()
class CPSQ(nn.Module):
    """Coverage-Preserving Semantic-Geometric Query Selection.

    CPSQ scores intermediate decoder queries, builds a semantic-geometric
    relation graph, greedily selects a compact representative set, and adds
    soft losses that teach the quality predictor to preserve GT coverage.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 hidden_dims: int = 256,
                 coverage_target: float = 0.95,
                 min_queries: int = 128,
                 max_queries: int = 500,
                 topk_neighbors: int = 32,
                 semantic_weight: float = 0.5,
                 geometric_weight: float = 0.5,
                 quality_loss_weight: float = 1.0,
                 coverage_loss_weight: float = 1.0,
                 budget_loss_weight: float = 0.1,
                 survival_threshold: float = 0.5,
                 initial_temperature: float = 1.0,
                 min_temperature: float = 0.1,
                 temperature_decay: float = 0.999,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.coverage_target = coverage_target
        self.min_queries = min_queries
        self.max_queries = max_queries
        self.topk_neighbors = topk_neighbors
        self.semantic_weight = semantic_weight
        self.geometric_weight = geometric_weight
        self.quality_loss_weight = quality_loss_weight
        self.coverage_loss_weight = coverage_loss_weight
        self.budget_loss_weight = budget_loss_weight
        self.survival_threshold = survival_threshold
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.eps = eps
        self.register_buffer('temperature_step', torch.zeros((), dtype=torch.long))

        quality_in_dims = embed_dims + 4
        self.quality_mlp = nn.Sequential(
            nn.Linear(quality_in_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, hidden_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims // 2, 1))

    def select(self,
               query: Tensor,
               cls_score: Tensor,
               bboxes: Tensor,
               gt_bboxes: Optional[Tensor] = None,
               gt_labels: Optional[Tensor] = None,
               max_select: Optional[int] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        if query.numel() == 0:
            empty = query.new_zeros((0,), dtype=torch.long)
            zero = query.sum() * 0
            return empty, dict(
                loss_cpsq_quality=zero,
                loss_cpsq_coverage=zero,
                loss_cpsq_budget=zero)

        probs = cls_score.sigmoid()
        quality = self.predict_quality(query, probs, bboxes)
        relation = self.build_relation(probs, bboxes)
        max_queries = (
            self.max_queries if max_select is None
            else min(self.max_queries, max_select))
        keep = self.greedy_select(relation, quality, max_queries)
        losses = self.loss(query, probs, bboxes, quality, gt_bboxes, gt_labels)
        return keep, losses

    def predict_quality(self, query: Tensor, probs: Tensor, bboxes: Tensor) -> Tensor:
        max_prob = probs.max(dim=-1).values
        entropy = self.entropy(probs)
        area = (bboxes[:, 2].clamp_min(self.eps) * bboxes[:, 3].clamp_min(self.eps)).log()
        aspect = (bboxes[:, 2].clamp_min(self.eps) / bboxes[:, 3].clamp_min(self.eps)).log().abs()
        quality_input = torch.cat([
            query, max_prob[:, None], entropy[:, None], area[:, None],
            aspect[:, None]
        ], dim=-1)
        return self.quality_mlp(quality_input).sigmoid().squeeze(-1)

    def build_relation(self, probs: Tensor, bboxes: Tensor) -> Tensor:
        semantic = F.cosine_similarity(probs[:, None, :], probs[None, :, :], dim=-1)
        semantic = semantic.clamp(min=0, max=1)
        geometric = torch.exp(-self.bhattacharyya_distance(bboxes, bboxes)).clamp(0, 1)
        relation = semantic.pow(self.semantic_weight) * geometric.pow(self.geometric_weight)
        relation.fill_diagonal_(1)
        return self.keep_topk_neighbors(relation)

    def greedy_select(self, relation: Tensor, quality: Tensor, max_queries: int) -> Tensor:
        num_queries = relation.size(0)
        max_queries = min(max_queries, num_queries)
        min_queries = min(self.min_queries, max_queries)
        selected = []
        covered = relation.new_zeros(num_queries)
        total_quality = quality.sum().clamp_min(self.eps)
        selected_mask = torch.zeros(
            num_queries, device=relation.device, dtype=torch.bool)

        for _ in range(max_queries):
            uncovered_weight = quality * (1 - covered)
            gain = (relation * uncovered_weight[None, :]).sum(dim=1)
            gain = gain.masked_fill(selected_mask, -1)
            next_idx = int(gain.argmax().item())
            selected.append(next_idx)
            selected_mask[next_idx] = True
            covered = 1 - (1 - covered) * (1 - relation[next_idx])
            coverage = (covered * quality).sum() / total_quality
            if len(selected) >= min_queries and coverage >= self.coverage_target:
                break

        if len(selected) < min_queries:
            _, ranked = quality.topk(min(min_queries, num_queries))
            seen = set(selected)
            for idx in ranked.tolist():
                if idx not in seen:
                    selected.append(idx)
                    seen.add(idx)
                if len(selected) >= min_queries:
                    break

        return torch.tensor(selected, device=relation.device, dtype=torch.long)

    def loss(self,
             query: Tensor,
             probs: Tensor,
             bboxes: Tensor,
             quality: Tensor,
             gt_bboxes: Optional[Tensor],
             gt_labels: Optional[Tensor]) -> Dict[str, Tensor]:
        zero = quality.sum() * 0
        if gt_bboxes is None or gt_labels is None or gt_bboxes.numel() == 0:
            return dict(
                loss_cpsq_quality=zero,
                loss_cpsq_coverage=zero,
                loss_cpsq_budget=quality.mean() * self.budget_loss_weight)

        ious = rbbox_overlaps(bboxes, gt_bboxes.detach())
        gt_probs = probs[:, gt_labels.long()].clamp(0, 1)
        match = ious * gt_probs
        quality_target = match.max(dim=1).values.detach()
        loss_quality = F.binary_cross_entropy(
            quality.clamp(self.eps, 1 - self.eps), quality_target)

        temperature = self.current_temperature()
        if self.training:
            self.temperature_step += 1
        survival = torch.sigmoid(
            (quality - self.survival_threshold) / temperature.clamp_min(self.eps))
        gt_coverage = 1 - torch.prod(1 - survival[:, None] * match.clamp(0, 1), dim=0)
        loss_coverage = -torch.log(gt_coverage.clamp_min(self.eps)).mean()
        loss_budget = survival.mean()

        return dict(
            loss_cpsq_quality=loss_quality * self.quality_loss_weight,
            loss_cpsq_coverage=loss_coverage * self.coverage_loss_weight,
            loss_cpsq_budget=loss_budget * self.budget_loss_weight)

    def current_temperature(self) -> Tensor:
        decayed = self.initial_temperature * (
            self.temperature_decay ** self.temperature_step.float())
        return decayed.clamp_min(self.min_temperature)

    @staticmethod
    def entropy(probs: Tensor) -> Tensor:
        probs = probs.clamp_min(1e-6)
        entropy = -(probs * probs.log()).sum(dim=-1)
        normalizer = torch.log(probs.new_tensor(probs.size(-1))).clamp_min(1e-6)
        return entropy / normalizer

    def keep_topk_neighbors(self, relation: Tensor) -> Tensor:
        if self.topk_neighbors <= 0 or self.topk_neighbors >= relation.size(1):
            return relation
        values, indices = relation.topk(
            min(self.topk_neighbors, relation.size(1)), dim=1)
        sparse = relation.new_zeros(relation.shape)
        sparse.scatter_(1, indices, values)
        return torch.maximum(sparse, sparse.t())

    def bhattacharyya_distance(self, boxes1: Tensor, boxes2: Tensor) -> Tensor:
        mean1, cov1 = self.rbox_to_gaussian(boxes1)
        mean2, cov2 = self.rbox_to_gaussian(boxes2)
        cov = (cov1[:, None] + cov2[None, :]) * 0.5
        diff = mean1[:, None, :] - mean2[None, :, :]
        inv_cov = torch.linalg.inv(cov)
        term1 = 0.125 * (diff.unsqueeze(-2) @ inv_cov @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        det_cov = torch.linalg.det(cov).clamp_min(self.eps)
        det_cov1 = torch.linalg.det(cov1).clamp_min(self.eps)
        det_cov2 = torch.linalg.det(cov2).clamp_min(self.eps)
        term2 = 0.5 * torch.log(det_cov / torch.sqrt(det_cov1[:, None] * det_cov2[None, :]))
        return (term1 + term2).clamp_min(0)

    def rbox_to_gaussian(self, boxes: Tensor) -> Tuple[Tensor, Tensor]:
        mean = boxes[:, :2]
        wh = boxes[:, 2:4].clamp_min(self.eps)
        angle = boxes[:, 4]
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        rot = boxes.new_zeros((boxes.size(0), 2, 2))
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = -sin
        rot[:, 1, 0] = sin
        rot[:, 1, 1] = cos
        scale = boxes.new_zeros((boxes.size(0), 2, 2))
        scale[:, 0, 0] = (wh[:, 0] * 0.5).pow(2)
        scale[:, 1, 1] = (wh[:, 1] * 0.5).pow(2)
        cov = rot @ scale @ rot.transpose(-1, -2)
        eye = torch.eye(2, device=boxes.device, dtype=boxes.dtype).unsqueeze(0)
        return mean, cov + eye * self.eps
