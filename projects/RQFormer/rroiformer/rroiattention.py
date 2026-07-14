import torch
from mmengine.model import BaseModule, constant_init, xavier_init
from torch import Tensor, nn

from mmdet.utils import OptConfigType
from mmrotate.registry import MODELS


@MODELS.register_module()
class RRoIAttention(BaseModule):
    """RRoI attention with optional rotated geometry guidance.

    The original RQFormer path is kept: attention weights are generated from
    object queries over the fixed RoI grid. When rotated boxes are provided,
    this module additionally injects rotated geometry embedding into the query
    and adds an orientation-aware spatial bias before softmax.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 roi_pooler_resolution: int = 7,
                 geometry_dims: int = 7,
                 use_geometry: bool = True,
                 init_cfg: OptConfigType = None) -> None:
        super(RRoIAttention, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.roi_pooler_resolution = roi_pooler_resolution
        self.geometry_dims = geometry_dims
        self.use_geometry = use_geometry

        num_grid = roi_pooler_resolution**2
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_grid)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.geometry_embed = nn.Sequential(
            nn.Linear(geometry_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims))
        self.orientation_bias = nn.Sequential(
            nn.Linear(geometry_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_heads * num_grid))

        self.init_weights()

    def init_weights(self) -> None:
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        for module in [self.geometry_embed, self.orientation_bias]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    xavier_init(layer, distribution='uniform', bias=0.)
        constant_init(self.geometry_embed[-1], val=0., bias=0.)
        constant_init(self.orientation_bias[-1], val=0., bias=0.)

    def _box_geometry(self,
                      boxes: Tensor,
                      img_shape: tuple = None) -> Tensor:
        """Encode rotated boxes as normalized geometry vectors.

        Args:
            boxes (Tensor): shape (num_queries, 5), ordered as
                (cx, cy, w, h, theta).
            img_shape (tuple, optional): image shape in (h, w, ...).

        Returns:
            Tensor: shape (num_queries, 7) containing normalized center,
            size, angle and aspect-ratio terms.
        """
        eps = boxes.new_tensor(1e-6)
        cx, cy, w, h, theta = boxes.unbind(dim=-1)
        w = w.clamp_min(eps)
        h = h.clamp_min(eps)

        if img_shape is not None:
            img_h = boxes.new_tensor(float(img_shape[0])).clamp_min(eps)
            img_w = boxes.new_tensor(float(img_shape[1])).clamp_min(eps)
        else:
            img_w = boxes[:, 0].max().clamp_min(eps)
            img_h = boxes[:, 1].max().clamp_min(eps)

        return torch.stack((
            cx / img_w,
            cy / img_h,
            torch.log(w / img_w + eps),
            torch.log(h / img_h + eps),
            torch.sin(theta),
            torch.cos(theta),
            torch.log(w / h + eps)), dim=-1)

    def forward(self,
                query: Tensor,
                roi_feat: Tensor,
                boxes: Tensor = None,
                img_shape: tuple = None) -> Tensor:
        """Forward function for RRoIAttention.

        Args:
            query (Tensor): shape (bs, num_queries, embed_dims).
            roi_feat (Tensor): shape
                (bs * num_queries, embed_dims, pooling_h, pooling_w).
            boxes (Tensor, optional): rotated boxes of the current image,
                shape (num_queries, 5).
            img_shape (tuple, optional): image shape for geometry
                normalization.

        Returns:
            Tensor: shape (bs, num_queries, embed_dims).
        """
        bs, num_queries = query.shape[:2]
        geometry = None
        if self.use_geometry and boxes is not None and boxes.numel() > 0:
            geometry = self._box_geometry(boxes, img_shape)
            query = query + self.geometry_embed(geometry).unsqueeze(0)

        attention_logits = self.attention_weights(query).view(
            bs, num_queries, self.num_heads,
            self.roi_pooler_resolution**2)
        if geometry is not None:
            orientation_bias = self.orientation_bias(geometry).view(
                1, num_queries, self.num_heads,
                self.roi_pooler_resolution**2)
            attention_logits = attention_logits + orientation_bias

        attention_weights = attention_logits.softmax(-1)
        attention_weights = attention_weights.unsqueeze(-2)
        value = self.value_proj(roi_feat.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2).contiguous()
        value = value.view(bs, num_queries, self.num_heads, -1,
                           self.roi_pooler_resolution**2)
        output = (value * attention_weights).sum(-1).view(
            bs, num_queries, self.embed_dims)

        output = self.output_proj(output)
        return output
