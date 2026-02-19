from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import resnet101, resnet18

from .geometry import GeometryConsistencyAdapter


def _make_resnet(name: str) -> nn.Module:
    if name == "resnet18":
        return resnet18(weights=None)
    if name == "resnet101":
        return resnet101(weights=None)
    raise ValueError(f"不支持的主干网络: {name}")


def _build_default_y_steps(num_points: int) -> Tensor:
    y = torch.linspace(5.0, 100.0, steps=num_points)
    return torch.round(y * 10.0) / 10.0


def _safe_linspace(start: float, end: float, num: int) -> list[float]:
    if num <= 1:
        return [float((start + end) * 0.5)]
    return [float(v) for v in torch.linspace(start, end, steps=num).tolist()]


def _resample_to_count(indices: Tensor, count: int) -> Tensor:
    if indices.numel() >= count:
        pos = torch.linspace(0, indices.numel() - 1, steps=count).long()
        return indices[pos]
    repeat = (count + indices.numel() - 1) // indices.numel()
    tiled = indices.repeat(repeat)
    return tiled[:count]


class AnchorBankBuilder:
    """锚点库构建器，输出物理坐标系下的3D锚点。"""

    def __init__(self, y_steps: Tensor, anchor_cfg: dict) -> None:
        self.y_steps = y_steps
        self.x_min = float(anchor_cfg.get("x_min", -10.0))
        self.x_max = float(anchor_cfg.get("x_max", 10.0))
        self.num_x = int(anchor_cfg.get("num_x", 16))
        self.pitches = list(anchor_cfg.get("pitches", [-3.0, 0.0, 3.0]))
        self.yaws = list(anchor_cfg.get("yaws", [-12.0, -6.0, 0.0, 6.0, 12.0]))
        self.start_z = float(anchor_cfg.get("start_z", 0.0))
        self.min_visible_ratio = float(anchor_cfg.get("min_visible_ratio", 0.4))

    def _build_one(self, start_x: float, pitch_deg: float, yaw_deg: float) -> Tensor | None:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        x = start_x + self.y_steps * math.tan(yaw)
        z = self.start_z + self.y_steps * math.tan(pitch)
        vis = (x > self.x_min) & (x < self.x_max)
        if vis.float().mean().item() < self.min_visible_ratio:
            return None
        y = self.y_steps.clone()
        return torch.stack([x, y, z], dim=-1)

    def build(self, query_count: int) -> Tensor:
        starts = _safe_linspace(self.x_min, self.x_max, self.num_x)
        anchors: list[Tensor] = []
        for start_x in starts:
            for pitch in self.pitches:
                for yaw in self.yaws:
                    lane = self._build_one(start_x, pitch, yaw)
                    if lane is not None:
                        anchors.append(lane)
        if not anchors:
            x = torch.linspace(self.x_min, self.x_max, steps=query_count).view(query_count, 1)
            y = self.y_steps.view(1, -1).expand(query_count, -1)
            z = torch.zeros_like(y)
            return torch.stack([x.expand_as(y), y, z], dim=-1)
        dense = torch.stack(anchors, dim=0)
        inds = _resample_to_count(torch.arange(dense.shape[0]), query_count)
        return dense.index_select(0, inds)


class ResNetFPNBackbone(nn.Module):
    """ResNet + FPN重型骨干。"""

    def __init__(self, backbone_name: str, hidden_dim: int) -> None:
        super().__init__()
        net = _make_resnet(backbone_name)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        c2_dim = 64 if backbone_name == "resnet18" else 256
        c3_dim = 128 if backbone_name == "resnet18" else 512
        c4_dim = 256 if backbone_name == "resnet18" else 1024
        c5_dim = 512 if backbone_name == "resnet18" else 2048
        self.lat2 = nn.Conv2d(c2_dim, hidden_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(c3_dim, hidden_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(c4_dim, hidden_dim, kernel_size=1)
        self.lat5 = nn.Conv2d(c5_dim, hidden_dim, kernel_size=1)
        self.out2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.out3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.out4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.out5 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.global_proj = nn.Linear(hidden_dim * 4, hidden_dim)

    def _merge(self, top: Tensor, lateral: Tensor) -> Tensor:
        up = F.interpolate(top, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
        return up + lateral

    def _global_feature(self, p2: Tensor, p3: Tensor, p4: Tensor, p5: Tensor) -> Tensor:
        pools = [F.adaptive_avg_pool2d(level, output_size=1).flatten(1) for level in [p2, p3, p4, p5]]
        return self.global_proj(torch.cat(pools, dim=1))

    def forward(self, image: Tensor) -> tuple[list[Tensor], Tensor]:
        x = self.stem(image)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p5 = self.lat5(c5)
        p4 = self._merge(p5, self.lat4(c4))
        p3 = self._merge(p4, self.lat3(c3))
        p2 = self._merge(p3, self.lat2(c2))
        p2, p3 = self.out2(p2), self.out3(p3)
        p4, p5 = self.out4(p4), self.out5(p5)
        return [p2, p3, p4, p5], self._global_feature(p2, p3, p4, p5)


class ProjectionSampler(nn.Module):
    """将3D锚点投影到特征图并采样。"""

    def __init__(self, x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
        super().__init__()
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

    def _project_with_matrix(self, project_matrix: Tensor, lane_xyz: Tensor) -> tuple[Tensor, Tensor]:
        bsz, lanes, points, _ = lane_xyz.shape
        ones = torch.ones_like(lane_xyz[..., :1])
        homo = torch.cat([lane_xyz, ones], dim=-1).reshape(bsz, -1, 4).transpose(1, 2)
        proj = torch.bmm(project_matrix, homo)
        depth = proj[:, 2, :].clamp(min=1e-4)
        u = (proj[:, 0, :] / depth).reshape(bsz, lanes, points)
        v = (proj[:, 1, :] / depth).reshape(bsz, lanes, points)
        return u, v

    def _project_with_range(self, lane_xyz: Tensor, feat_h: int, feat_w: int) -> tuple[Tensor, Tensor]:
        x = lane_xyz[..., 0]
        y = lane_xyz[..., 1]
        u = (x - self.x_min) / max(self.x_max - self.x_min, 1e-4) * (feat_w - 1)
        v = (1.0 - (y - self.y_min) / max(self.y_max - self.y_min, 1e-4)) * (feat_h - 1)
        return u, v

    def forward(self, features: Tensor, lane_xyz: Tensor, project_matrix: Tensor | None) -> tuple[Tensor, Tensor]:
        bsz, _, feat_h, feat_w = features.shape
        if project_matrix is None:
            u, v = self._project_with_range(lane_xyz, feat_h, feat_w)
        else:
            u, v = self._project_with_matrix(project_matrix, lane_xyz)
        u_norm = (u / max(feat_w - 1, 1)) * 2.0 - 1.0
        v_norm = (v / max(feat_h - 1, 1)) * 2.0 - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1)
        sampled = F.grid_sample(features, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = sampled.permute(0, 2, 3, 1).contiguous()
        valid = (u_norm > -1.0) & (u_norm < 1.0) & (v_norm > -1.0) & (v_norm < 1.0)
        return sampled, valid.float()


class DepthAwareTokenAggregator(nn.Module):
    """创新点：基于y深度与可见性的自适应token聚合。"""

    def __init__(self, in_dim: int, y_steps: Tensor) -> None:
        super().__init__()
        y_min = float(y_steps.min().item())
        y_max = float(y_steps.max().item())
        y_norm = (y_steps - y_min) / max(y_max - y_min, 1e-6)
        self.register_buffer("y_norm", y_norm, persistent=True)
        self.weight_mlp = nn.Sequential(
            nn.Linear(in_dim + 1, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, sampled: Tensor, valid: Tensor) -> Tensor:
        y = self.y_norm.view(1, 1, -1, 1).expand(sampled.shape[0], sampled.shape[1], -1, -1)
        feat = torch.cat([sampled, y], dim=-1)
        logits = self.weight_mlp(feat).squeeze(-1)
        logits = logits.masked_fill(valid < 0.5, -1e4)
        weights = torch.softmax(logits, dim=-1) * valid
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        return (sampled * weights.unsqueeze(-1)).sum(dim=2)


class GeometryAwareRefiner(nn.Module):
    """创新点：曲率与可见性驱动的几何门控。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.geo_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def _geo_feature(self, lane_xyz: Tensor, lane_vis: Tensor) -> Tensor:
        x = lane_xyz[..., 0]
        z = lane_xyz[..., 2]
        x_mean = x.mean(dim=-1)
        z_mean = z.mean(dim=-1)
        if x.shape[-1] > 2:
            dx = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
            dz = z[..., 2:] - 2 * z[..., 1:-1] + z[..., :-2]
            curvature = torch.sqrt(dx.square() + dz.square() + 1e-6).mean(dim=-1)
        else:
            curvature = x_mean.new_zeros(x_mean.shape)
        vis_ratio = lane_vis.mean(dim=-1)
        return torch.stack([x_mean, z_mean, curvature, vis_ratio], dim=-1)

    def forward(self, token: Tensor, lane_xyz: Tensor, lane_vis: Tensor) -> Tensor:
        geo = self._geo_feature(lane_xyz, lane_vis)
        geo_embed = self.geo_mlp(geo)
        weight = self.gate(geo_embed)
        return token * (1.0 + weight)


class IterativeDecodeStage(nn.Module):
    """单阶段回归头。"""

    def __init__(self, hidden_dim: int, num_points: int, num_category: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.cls_layer = nn.Linear(hidden_dim, num_category)
        self.reg_x = nn.Linear(hidden_dim, num_points)
        self.reg_z = nn.Linear(hidden_dim, num_points)
        self.reg_vis = nn.Linear(hidden_dim, num_points)
        self.reg_logvar_x = nn.Linear(hidden_dim, num_points)
        self.reg_logvar_z = nn.Linear(hidden_dim, num_points)

    def forward(self, token: Tensor) -> dict[str, Tensor]:
        feat = self.shared(token)
        return {
            "cls_logits": self.cls_layer(feat),
            "delta_x": self.reg_x(feat),
            "delta_z": self.reg_z(feat),
            "vis": torch.sigmoid(self.reg_vis(feat)),
            "logvar_x": self.reg_logvar_x(feat),
            "logvar_z": self.reg_logvar_z(feat),
        }


class LaneMaster3DNet(nn.Module):
    """论文级主链路：锚点投影采样 + 迭代回归。"""

    def __init__(
        self,
        hidden_dim: int,
        query_count: int,
        num_points: int,
        use_gca: bool,
        num_category: int = 21,
        backbone_name: str = "resnet101",
        dynamic_anchor_enabled: bool = True,
        dynamic_anchor_delta_scale: float = 0.25,
        y_steps: list[float] | None = None,
        anchor_cfg: dict | None = None,
        iter_reg: int = 2,
        anchor_feat_channels: int = 64,
        feature_level: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_count = query_count
        self.num_points = num_points
        self.num_category = num_category
        self.iter_reg = iter_reg
        self.use_gca = use_gca
        self.dynamic_anchor_enabled = dynamic_anchor_enabled
        self.dynamic_anchor_delta_scale = float(dynamic_anchor_delta_scale)
        self.anchor_feat_channels = anchor_feat_channels
        self.feature_level = feature_level
        y_tensor = torch.tensor(y_steps, dtype=torch.float32) if y_steps is not None else _build_default_y_steps(num_points)
        if y_tensor.numel() != num_points:
            raise ValueError("y_steps 长度必须等于 num_points")
        self.register_buffer("y_steps", y_tensor, persistent=True)
        self.backbone = ResNetFPNBackbone(backbone_name=backbone_name, hidden_dim=hidden_dim)
        self.anchor_projection = nn.Conv2d(hidden_dim, anchor_feat_channels, kernel_size=1)
        cfg = anchor_cfg or {}
        bank = AnchorBankBuilder(self.y_steps, cfg).build(query_count)
        self.register_buffer("base_lane_anchors", bank, persistent=True)
        self.sampler = ProjectionSampler(
            x_range=(float(cfg.get("x_min", -10.0)), float(cfg.get("x_max", 10.0))),
            y_range=(float(self.y_steps.min().item()), float(self.y_steps.max().item())),
        )
        self.depth_agg = DepthAwareTokenAggregator(anchor_feat_channels, self.y_steps)
        token_in_dim = anchor_feat_channels * (num_points + 1)
        self.token_proj = nn.Sequential(nn.Linear(token_in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.geo_refiner = GeometryAwareRefiner(hidden_dim)
        if use_gca:
            self.gca = GeometryConsistencyAdapter(anchor_feat_channels, 3, anchor_feat_channels)
        self.decode_stages = nn.ModuleList(
            [IterativeDecodeStage(hidden_dim, num_points, num_category=num_category) for _ in range(iter_reg + 1)]
        )
        if dynamic_anchor_enabled:
            self.dynamic_anchor_delta = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, query_count * num_points * 2),
            )

    def _proposal_template(self, batch_size: int, lane_xyz: Tensor, lane_vis: Tensor) -> Tensor:
        proposal = lane_xyz.new_zeros(batch_size, self.query_count, 5 + self.num_points * 3)
        proposal[:, :, 3] = 1.0
        proposal[:, :, 4] = float(self.num_points)
        proposal[:, :, 5:5 + self.num_points] = lane_xyz[..., 0]
        proposal[:, :, 5 + self.num_points:5 + self.num_points * 2] = lane_xyz[..., 2]
        proposal[:, :, 5 + self.num_points * 2:5 + self.num_points * 3] = lane_vis
        return proposal

    def _lane_from_proposal(self, proposal: Tensor) -> tuple[Tensor, Tensor]:
        x = proposal[:, :, 5:5 + self.num_points]
        z = proposal[:, :, 5 + self.num_points:5 + self.num_points * 2]
        vis = proposal[:, :, 5 + self.num_points * 2:5 + self.num_points * 3].clamp(0.0, 1.0)
        y = self.y_steps.view(1, 1, -1).expand_as(x)
        lane_xyz = torch.stack([x, y, z], dim=-1)
        return lane_xyz, vis

    def _geo_tokens(self, lane_xyz: Tensor) -> Tensor:
        x_mean = lane_xyz[..., 0].mean(dim=-1)
        z_mean = lane_xyz[..., 2].mean(dim=-1)
        if lane_xyz.shape[2] > 2:
            dx = lane_xyz[:, :, 2:, 0] - 2 * lane_xyz[:, :, 1:-1, 0] + lane_xyz[:, :, :-2, 0]
            dz = lane_xyz[:, :, 2:, 2] - 2 * lane_xyz[:, :, 1:-1, 2] + lane_xyz[:, :, :-2, 2]
            curve = torch.sqrt(dx.square() + dz.square() + 1e-6).mean(dim=-1)
        else:
            curve = x_mean.new_zeros(x_mean.shape)
        return torch.stack([x_mean, z_mean, curve], dim=-1)

    def _build_anchor_priors(self, global_feat: Tensor, batch_size: int) -> Tensor:
        anchors = self.base_lane_anchors.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        if not self.dynamic_anchor_enabled:
            return anchors
        delta = self.dynamic_anchor_delta(global_feat).reshape(batch_size, self.query_count, self.num_points, 2)
        anchors[..., 0] += delta[..., 0] * self.dynamic_anchor_delta_scale
        anchors[..., 2] += delta[..., 1] * self.dynamic_anchor_delta_scale
        return anchors

    def _tokenize(self, sampled: Tensor, valid: Tensor, lane_xyz: Tensor, lane_vis: Tensor) -> Tensor:
        pooled = self.depth_agg(sampled, valid)
        geo = self._geo_tokens(lane_xyz)
        if self.use_gca:
            pooled = pooled + self.gca(pooled, geo)
        flat = sampled.flatten(start_dim=2)
        token = self.token_proj(torch.cat([flat, pooled], dim=-1))
        return self.geo_refiner(token, lane_xyz, lane_vis)

    def _decode_once(self, token: Tensor, prev_proposal: Tensor, stage_idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        stage_out = self.decode_stages[stage_idx](token)
        proposal = prev_proposal.clone()
        proposal[:, :, 5:5 + self.num_points] += stage_out["delta_x"]
        proposal[:, :, 5 + self.num_points:5 + self.num_points * 2] += stage_out["delta_z"]
        proposal[:, :, 5 + self.num_points * 2:5 + self.num_points * 3] = stage_out["vis"]
        ddp_aux = (stage_out["logvar_x"].sum() + stage_out["logvar_z"].sum() + stage_out["cls_logits"].sum()) * 0.0
        return proposal, stage_out["cls_logits"], stage_out["logvar_x"], stage_out["logvar_z"], ddp_aux

    def _pack_output(
        self,
        proposal: Tensor,
        cls_logits: Tensor,
        anchors: Tensor,
        logvar_x: Tensor,
        logvar_z: Tensor,
    ) -> dict[str, Tensor]:
        reg_proposals = torch.cat([proposal, cls_logits], dim=-1)
        lane_xyz, lane_vis = self._lane_from_proposal(proposal)
        cls_prob = 1.0 - torch.softmax(cls_logits, dim=-1)[..., 0]
        return {
            "pred_points": lane_xyz,
            "pred_scores": cls_prob,
            "pred_vis": lane_vis,
            "pred_logits": cls_logits,
            "pred_logvar_x": logvar_x,
            "pred_logvar_z": logvar_z,
            "reg_proposals": reg_proposals,
            "anchors": anchors,
        }

    def forward(self, image: Tensor, project_matrix: Tensor | None = None) -> dict[str, Tensor]:
        feats, global_feat = self.backbone(image)
        feat = self.anchor_projection(feats[self.feature_level])
        batch = image.shape[0]
        anchor_priors = self._build_anchor_priors(global_feat, batch)
        lane_vis = anchor_priors.new_ones(batch, self.query_count, self.num_points)
        anchors = self._proposal_template(batch, anchor_priors, lane_vis)
        proposal = anchors.clone()
        cls_logits = proposal.new_zeros(batch, self.query_count, self.num_category)
        logvar_x = proposal.new_zeros(batch, self.query_count, self.num_points)
        logvar_z = proposal.new_zeros(batch, self.query_count, self.num_points)
        ddp_aux = proposal.new_zeros(())
        lane_xyz = anchor_priors
        for stage_idx in range(self.iter_reg + 1):
            sampled, valid = self.sampler(feat, lane_xyz, project_matrix)
            lane_vis = lane_vis * valid
            token = self._tokenize(sampled, valid, lane_xyz, lane_vis)
            proposal, cls_logits, logvar_x, logvar_z, stage_aux = self._decode_once(token, proposal, stage_idx)
            ddp_aux = ddp_aux + stage_aux
            lane_xyz, lane_vis = self._lane_from_proposal(proposal)
        out = self._pack_output(proposal, cls_logits, anchors, logvar_x, logvar_z)
        out["ddp_aux"] = ddp_aux
        out["anchor_priors"] = anchors[:, :, 5:5 + self.num_points].new_zeros(batch, self.query_count, self.num_points, 3)
        out["anchor_priors"][..., 0] = anchors[:, :, 5:5 + self.num_points]
        out["anchor_priors"][..., 1] = self.y_steps.view(1, 1, -1)
        out["anchor_priors"][..., 2] = anchors[:, :, 5 + self.num_points:5 + self.num_points * 2]
        return out
