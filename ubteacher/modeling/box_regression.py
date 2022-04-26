import math
from typing import List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch.nn import functional as F

from detectron2.structures import Boxes

# Value for clamping large dw and dh predictions. The heuristic is that we clamp
# such that dw and dh are no larger than what would transform a 16px box into a
# 1000px box (based on a small anchor, 16px, and a typical image size, 1000px).
_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


__all__ = ["Box2BoxTransform_XYXY"]


@torch.jit.script
class Box2BoxTransform_XYXY(object):

    def __init__(
        self, weights: Tuple[float, float, float, float], scale_clamp: float = _DEFAULT_SCALE_CLAMP
    ):
        
        self.weights = weights
        self.scale_clamp = scale_clamp

    def get_deltas(self, src_boxes, target_boxes):
        
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_x1 = src_boxes[:, 0]
        src_y1 = src_boxes[:, 1]
        src_x2 = src_boxes[:, 2]
        src_y2 = src_boxes[:, 3]

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_x1 = target_boxes[:, 0]
        target_y1 = target_boxes[:, 1]
        target_x2 = target_boxes[:, 2]
        target_y2 = target_boxes[:, 3]
        
        wx, wy, ww, wh = self.weights
        dx1 = wx * (target_x1 - src_x1) / src_widths
        dy1 = wy * (target_y1 - src_y1) / src_heights
        dx2 = wx * (target_x2 - src_x2) / src_widths
        dy2 = wy * (target_y2 - src_y2) / src_heights

        deltas = torch.stack((dx1, dy1, dx2, dy2), dim=1)
        assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):

        deltas = deltas.float()  # ensure fp32 for decoding precision
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        wx, wy, ww, wh = self.weights
        dx1 = deltas[:, 0::4] / wx
        dy1 = deltas[:, 1::4] / wy
        dx2 = deltas[:, 2::4] / wx
        dy2 = deltas[:, 3::4] / wy

        pred_x1 = dx1 * widths[:, None] + x1[:, None]
        pred_y1 = dy1 * heights[:, None] + y1[:, None]
        pred_x2 = dx2 * widths[:, None] + x2[:, None]
        pred_y2 = dy2 * heights[:, None] + y2[:, None]

        pred_boxes = torch.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim=-1)
        return pred_boxes.reshape(deltas.shape)