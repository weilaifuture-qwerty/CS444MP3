import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from network import RetinaNet
from detection_utils import compute_targets, get_detections, set_seed
from predict import validate, test
from tensorboardX import SummaryWriter
from absl import app, flags
import numpy as np
from dataset import CocoDataset, Resizer, Normalizer, collater
from torchvision import transforms
import losses
import logging
import time

FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-4, 'Learning Rate')
flags.DEFINE_float('momentum', 0.9, 'Momentum for optimizer')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight Deacy for optimizer')
flags.DEFINE_string('output_dir', 'runs/retina-net-basic/', 'Output Directory')
flags.DEFINE_integer('batch_size', 1, 'Batch Size')
flags.DEFINE_integer('seed', 2, 'Random seed')
flags.DEFINE_integer('max_iter', 100000, 'Total Iterations')
flags.DEFINE_integer('val_every', 10000, 'Iterations interval to validate')
flags.DEFINE_integer('save_every', 50000, 'Iterations interval to validate')
flags.DEFINE_integer('preload_images', 1, 
    'Weather to preload train and val images at beginning of training. Preloading takes about 7 minutes on campus cluster but speeds up training by a lot. Set to 0 to disable.')
flags.DEFINE_multi_integer('lr_step', [60000, 80000], 'Iterations to reduce learning rate')

log_every = 20

import torch
import numpy as np
import random

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def get_detections(outs):
    B, BB, _, _ = outs[0][0].shape
    _, A, _, _ = outs[0][2].shape
    A = A // 4
    num_classes = BB // A
    
    pred_bboxes, pred_clss, anchors = [], [], []
    for pred_cls, pred_bbox, anchor in outs:
        # Get all the anchors, pred and bboxes
        H, W = pred_cls.shape[-2:]
        pred_cls = pred_cls.reshape(B, A, -1, H, W)
        pred_bbox = pred_bbox.reshape(B, A, -1, H, W)

        pred_clss.append(pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes))
        pred_bboxes.append(pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))

        anchor = anchor.reshape(B, A, -1, H, W)
        anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
    pred_clss = torch.cat(pred_clss, dim=1)
    pred_bboxes = torch.cat(pred_bboxes, dim=1)
    anchors = torch.cat(anchors, dim=1)
    return pred_clss, pred_bboxes, anchors

def compute_bbox_iou(bbox1, bbox2, dim=1):
    """
    Args:
        bbox1: (N, 4) tensor of (x1, y1, x2, y2)
        bbox2: (M, 4) tensor of (x1, y1, x2, y2)
    Returns
        iou: (N, M) tensor of IoU values
    """
    bbox1 = bbox1.unsqueeze(1)
    bbox2 = bbox2.unsqueeze(0)
    max_min_x = torch.max(bbox1[...,0], bbox2[...,0])
    min_max_x = torch.min(bbox1[...,2], bbox2[...,2])
    max_min_y = torch.max(bbox1[...,1], bbox2[...,1])
    min_max_y = torch.min(bbox1[...,3], bbox2[...,3])
    intersection = torch.clamp(min_max_x - max_min_x, min=0) * torch.clamp(min_max_y - max_min_y, min=0)
    area1 = (bbox1[...,2] - bbox1[...,0]) * (bbox1[...,3] - bbox1[...,1])
    area2 = (bbox2[...,2] - bbox2[...,0]) * (bbox2[...,3] - bbox2[...,1])
    iou = intersection / (area1 + area2 - intersection)
    return iou

def compute_targets(anchor, cls, bbox):
    """
    Args:
        anchor: batch of anchors in the format (x1, y1, x2, y2) or in other words (xmin, ymin, xmax, ymax); shape is (B, A, 4), where B denotes image batch size and A denotes the number of anchors
        cls: groundtruth object classes of shape (B, number of objects in the image, 1)
        bbox: groundtruth bounding boxes of shape (B, number of objects in the image, 4)
    Returns:
        gt_clss: groundtruth class targets of shape (B, A, 1)
        gt_bboxes: groundtruth bbox targets of shape (B, A, 4)
    
    Hint: remember if the max_iou for that bounding box is between [0, 0.4) then the gt_cls should equal 0(because it is being assigned as background) and the
    gt_bbox should be all zero(it can be anything since it will be ignored however our tests set them to zero so you should too).
    Also, if the max iou is between [0.4, 0.5) then the gt_cls should be equal to -1(since it's neither background or assigned to a class. This is basically tells the model to ignore this box) 
    and the gt_bbox should again arbitrarilarly be set to all zeros).
    Otherwise if the max_iou > 0.5, you should assign the anchor to the gt_box with the max iou, and the gt_cls will be the ground truth class of that max_iou box
    Hint: use torch.max to get both the max iou and the index of the max iou.

    Hint: We recommend using the compute_bbox_iou function which efficently computes the ious between two lists of bounding boxes as a helper function.

    Hint: make sure that the returned gt_clss tensor is of type int(since it will be used as an index in the loss function). Also make sure that both the gt_bboxes and gt_clss are on the same device as the anchor. 
    You can do this by calling .to(anchor.device) on the tensor you want to move to the same device as the anchor.

    VECTORIZING CODE: Again, you can use for loops initially to make the tests pass, but in order to make your code efficient 
    during training, you should only have one for loop over the batch dimension and everything else should be vectorized. We recommend using boolean masks to do this. i.e
    you can compute the max_ious for all the anchor boxes and then do gt_cls[max_iou < 0.4] = 0 to access all the anchor boxes that should be set to background and setting their gt_cls to 0. 
    This will remove the need for a for loop over all the anchor boxes. You can then do the same for the other cases. This will make your code much more efficient and faster to train.
    """
    B, A, _ = anchor.shape
    gt_clss = torch.full((B, A, 1), -1).to(anchor.device)
    gt_bboxes = torch.zeros((B, A, 4)).to(anchor.device)
    for b in range(B):
        iou = compute_bbox_iou(anchor[b], bbox[b])
        max_iou, max_iou_index = torch.max(iou, dim = 1)
        gt_clss[b] = cls[b][max_iou_index]
        gt_clss[b, max_iou < 0.5, 0] = -1
        gt_clss[b, max_iou < 0.4, 0] = 0
        mask = max_iou >= 0.5
        gt_bboxes[b][mask] = bbox[b][max_iou_index[mask]]
    return gt_clss.to(torch.int), gt_bboxes

def compute_bbox_targets(anchors, gt_bboxes):
    """
    Args:
        anchors: anchors of shape (A, 4)
        gt_bboxes: groundtruth object classes of shape (A, 4)
    Returns:
        bbox_reg_target: regression offset of shape (A, 4)
    
    Remember that the delta_x and delta_y we compute are with respect to the center of the anchor box. I.E, we're seeing how much that center of the anchor box changes. 
    We also need to normalize delta_x and delta_y which means that we need to divide them by the width or height of the anchor box respectively. This is to make
    our regression targets more invariant to the size of the original anchor box. So, this means that:
    delta_x = (gt_bbox_center_x - anchor_center_x) / anchor_width  and delta_y would be computed in a similar manner.

    When computing delta_w and delta_h, there are a few things to note.
    1. We also want to normalize these with respect to the width and height of the anchor boxes. so delta_w = gt_bbox_width / anchor_width
    2. Logarithm: In order to make our regresssion targets better handle varying sizees of the bounding boxes, we use the logarithmic scale for our delta_w and delta_h
       This is to ensure that if for example the gt_width is twice or 1/2 the size of the anchor_width, the magnitude in the log scale would stay the same but only the sign of
       our regression target would be different. Therefore our formula changes to delta_w = log(gt_bbox_width / anchor_width)
    3. Clamping: Remember that logarithms can't handle negative values and that the log of values very close to zero will have very large magnitudes and have extremly 
       high gradients which might make training unstable. To mitigate this we use clamping to ensure that the value that we log isn't too small. Therefore, our final formula will be
       delta_w = log(max(gt_bbox_width,1) / anchor_width)
       
    """
    # TODO(student): Complete this function
    anchor_center_x = (anchors[:, 2] + anchors[:, 0]) / 2
    anchor_center_y = (anchors[:, 3] + anchors[:, 1]) / 2
    gt_bbox_center_x = (gt_bboxes[:, 2] + gt_bboxes[:, 0]) / 2
    gt_bbox_center_y = (gt_bboxes[:, 3] + gt_bboxes[:, 1]) / 2
    anchor_width = anchors[:, 2] - anchors[:, 0]
    anchor_height = anchors[:, 3] - anchors[:, 1]
    gt_bbox_width = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    gt_bbox_height = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    delta_x = (gt_bbox_center_x - anchor_center_x) / anchor_width
    delta_y = (gt_bbox_center_y - anchor_center_y) / anchor_height
    tmp = torch.ones(gt_bbox_width.shape).to(anchors.device)
    delta_w = torch.log(torch.maximum(gt_bbox_width, tmp) / anchor_width).to(anchors.device)
    delta_h = torch.log(torch.maximum(gt_bbox_height, tmp) / anchor_height).to(anchors.device)
    ans = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)
    ans = ans.to(anchors.device)
    return ans

def apply_bbox_deltas(boxes, deltas):
    """
    Args:
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        deltas: (N, 4) tensor of (dxc, dyc, dlogw, dlogh)
    Returns
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        
    """
    # TODO(student): Complete this function
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    xcenter = (boxes[:, 2] + boxes[:, 0]) / 2 + deltas[:, 0] * w
    ycenter = (boxes[:, 3] + boxes[:, 1]) / 2 + deltas[:, 1] * h
    dw = torch.exp(deltas[:, 2])
    dh = torch.exp(deltas[:, 3])
    w = w * dw
    h = h * dh
    xmin = xcenter - w/2
    xmax = xcenter + w/2
    ymin = ycenter - h/2
    ymax = ycenter + h/2
    new_boxes = torch.stack([xmin, ymin, xmax, ymax], dim = -1)
    new_boxes = new_boxes.to(boxes.device)
    return new_boxes

def nms(bboxes, scores, threshold=0.5):
    """
    Args:
        bboxes: (N, 4) tensor of (x1, y1, x2, y2)
        scores: (N,) tensor of scores
    Returns:
        keep: (K,) tensor of indices to keep
    
    Remember that nms is used to prevent having many boxes that overlap each other. To do this, if multiple boxes overlap each other beyond a
    threshold iou, nms will pick the "best" box(the one with the highest score) and remove the rest. One way to implement this is to
    first compute the ious between all pairs of bboxes. Then loop over the bboxes from highest score to lowest score. Since this is the 
    best bbox(the one with the highest score), It will be choosen over all overlapping boxes. Therefore, you should add this bbox to your final 
    resulting bboxes and remove all the boxes that overlap with it from consideration. Then repeat until you've gone through all of the bboxes.

    make sure that the indices tensor that you return is of type int or long(since it will be used as an index to select the relevant bboxes to output)
    """
    N, _ = bboxes.shape
    iou = compute_bbox_iou(bboxes, bboxes)
    tag = torch.zeros(N)
    scores = -scores
    indexs = torch.argsort(scores)
    keep = []
    for index in indexs:
        if tag[index] == 0:
            tag[index] = 1
            keep.append(index)
            for j in range(N):
                if iou[index][j] > threshold:
                    tag[j] = 1
    return torch.tensor(keep, device = bboxes.device).to(torch.int)

import torch
import math
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = -1 // divisor
    num_groups = 32 // divisor
    eps = 1e-5  # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups), out_channels, eps, affine
    )


class Anchors(nn.Module):
    def __init__(
        self,
        stride,
        sizes=[4, 4 * math.pow(2, 1 / 3), 4 * math.pow(2, 2 / 3)],
        aspect_ratios=[0.5, 1, 2],
    ):
        """
        Args:
            stride: stride of the feature map relative to the original image
            sizes: list of sizes (sqrt of area) of anchors in units of stride
            aspect_ratios: list of aspect ratios (h/w) of anchors
        __init__ function does the necessary precomputations.
        forward function computes anchors for a given feature map

        Ex. if you are trying the compute the anchor boxes for the above image,
        you should store a 9*4 tensor in self.anchor_offsets(because we have 9 types of anchor boxes
        and each anchor box has 4 offsets from the location of the center of the box).
        HINT: Try using a double for loop to loop over every possible
        combination of sizes and aspect ratios to find out the coordinates of the anchor box for that particular size and aspect ratio.

        When calculating the width and height of the anchor box for a particular size and aspect ratio, remember that if the anchor box were a square
        then the length of each side would just be size*stride; however, we want the height and width of our anchor box to have a different
        width and height depending on the aspect ratio, so think about how you would use the aspect ratio to appropriatly scale the width and height such that w*h = (size*stride)^2 and
        aspect_ratio = h/w
        """
        super(Anchors, self).__init__()
        self.stride = stride
        self.anchor_offsets = torch.zeros((len(sizes) * len(aspect_ratios), 4))
        cnt = 0
        for size in sizes:
            for aspect_ratio in aspect_ratios:
                w = size * stride / math.sqrt(aspect_ratio) 
                h = aspect_ratio * w
                self.anchor_offsets[cnt][0] = -w/2
                self.anchor_offsets[cnt][1] = -h/2
                self.anchor_offsets[cnt][2] = w/2
                self.anchor_offsets[cnt][3] = h/2
                cnt = cnt + 1

    def forward(self, x):
        """
        Args:
            x: feature map of shape (B, C, H, W)
        Returns:
            anchors: list of anchors in the format (x1, y1, x2, y2)(in other words (xmin, ymin, xmax, ymax)), giving the shape
            of (B, A*4, H, W) where A is the number of types of anchor boxes we have.
            Hint: We want to loop over every pixel of the input and use that as the center of our bounding boxes. Then we can apply the offsets for each bounding box that we
            found earlier to get all the bounding boxes for that pixel. However, remember that this feature map is scaled down from the original image, so
            when finding the base y, x values(the location of the center of the anchor box with respect to the original image), remember that you need to multiply the current position
            in our feature map (i,j) by the stride to get what the position of the center would be in the base image.
            Hint2: the anchor boxes will be identical for all elements of the batch so try just calculating the anchor boxes for one element of the batch and
                then using torch.repeat to duplicate the tensor B times
            Hint3, remember to transfer your anchors to the same device that the input x is on before returning(you can access this with x.device). This is so we can use a gpu when training.

            MAKING CODE EFFICIENT: We recommend first just using for loops and to make sure that your logic is correct. However, this will be very slow. Therefore,
            we recommend using torch.mesh grid to create the grid of all possible y and x values and then adding them to your anchor offsets tensor that you stored from before
            to get tensors for x1, x2, y1, and y2 for all the anchor boxes. Then you should simply stack those tensors together, reshape them to match the expected output
            size and use torch.repeat to repeat that tensor B times across the batch dimension.
            Your final code should be fully verterized and not have any for loops. 
            Also make sure that when you create a tensor you put it on the same device that x is on.
        """
        B, C, H, W = x.shape
        A, _ = self.anchor_offsets.shape
        anchors_for_one = torch.zeros((A*4, H, W), device = x.device)
        y = torch.arange(H, device = x.device) * self.stride
        x = torch.arange(W, device = x.device) * self.stride
        y, x = torch.meshgrid(y, x, indexing='ij')
        anchors_for_one_pixel = torch.stack((x, y, x, y)).repeat(A, 1, 1)
        anchors_for_one_pixel.to(x.device)
        offsets = torch.unsqueeze(torch.unsqueeze(torch.flatten(self.anchor_offsets), 1), 1).expand(-1, H, W)
        offsets = offsets.to(x.device)
        anchors_for_one = anchors_for_one_pixel + offsets
        anchors_for_one.to(x.device)
        # x_min = torch.Tensor.repeat(x, (A, 1, 1)) + torch.unsqueeze(torch.unsqueeze(self.anchor_offsets[:, 0], 1), 1).expand(-1, H, W)
        # x_max = torch.Tensor.repeat(x, (A, 1, 1)) + torch.unsqueeze(torch.unsqueeze(self.anchor_offsets[:, 2], 1), 1).expand(-1, H, W)
        # y_min = torch.Tensor.repeat(y, (A, 1, 1)) + torch.unsqueeze(torch.unsqueeze(self.anchor_offsets[:, 1], 1), 1).expand(-1, H, W)
        # y_max = torch.Tensor.repeat(y, (A, 1, 1)) + torch.unsqueeze(torch.unsqueeze(self.anchor_offsets[:, 3], 1), 1).expand(-1, H, W)
        # print(anchors_for_one_pixel.shape, offsets.shape)
        # cnt = 0
        # for h in range(H):
        #     y = h*self.stride
        #     for w in range(W):
        #         x = w*self.stride         
        #         cnt = 0       
        #         for offset in self.anchor_offsets:
        #             anchors_for_one[cnt][h][w] = x + offset[0]
        #             cnt = cnt + 1
        #             anchors_for_one[cnt][h][w] = y + offset[1]
        #             cnt = cnt + 1
        #             anchors_for_one[cnt][h][w] = x + offset[2]
        #             cnt = cnt + 1
        #             anchors_for_one[cnt][h][w] = y + offset[3]
        #             cnt = cnt + 1
        anchors = anchors_for_one.repeat(B, 1, 1, 1)
        anchors.to(x.device)
        return anchors


class RetinaNet(nn.Module):
    def __init__(self, p67=False, fpn=False,num_anchors=9):
        super(RetinaNet, self).__init__()
        self.resnet = [
            create_feature_extractor(
                resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
                return_nodes={
                    "layer2.3.relu_2": "conv3",
                    "layer3.5.relu_2": "conv4",
                    "layer4.2.relu_2": "conv5",
                },
            )
        ]
        self.resnet[0].eval()
        self.cls_head, self.bbox_head = self.get_heads(10, num_anchors)

        self.p67 = p67
        self.fpn = fpn

        anchors = nn.ModuleList()

        self.p5 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0),
            group_norm(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
        )
        self._init(self.p5)
        anchors.append(Anchors(stride=32))

        if self.p67:
            self.p6 = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1),
                group_norm(256),
            )
            self._init(self.p6)
            self.p7 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                group_norm(256),
            )
            self._init(self.p7)
            anchors.append(Anchors(stride=64))
            anchors.append(Anchors(stride=128))

        if self.fpn:
            self.p4_lateral = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                group_norm(256),
            )
            self.p4 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
            )
            self._init(self.p4)
            self._init(self.p4_lateral)
            anchors.append(Anchors(stride=16))

            self.p3_lateral = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0), group_norm(256)
            )
            self.p3 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
            )
            self._init(self.p3)
            self._init(self.p3_lateral)
            anchors.append(Anchors(stride=8))

        self.anchors = anchors

    def _init(self, modules):
        for layer in modules.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def to(self, device):
        super(RetinaNet, self).to(device)
        self.anchors.to(device)
        self.resnet[0].to(device)
        return self

    def get_heads(self, num_classes, num_anchors, prior_prob=0.01):
        cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(
                256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
            ),
        )
        bbox_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, num_anchors * 4, kernel_size=3, stride=1, padding=1),
        )

        # Initialization
        for modules in [cls_head, bbox_head]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(cls_head[-1].bias, bias_value)

        return cls_head, bbox_head

    def get_ps(self, feats):
        conv3, conv4, conv5 = feats["conv3"], feats["conv4"], feats["conv5"]
        p5 = self.p5(conv5)
        outs = [p5]

        if self.p67:
            p6 = self.p6(conv5)
            outs.append(p6)

            p7 = self.p7(p6)
            outs.append(p7)

        if self.fpn:
            p4 = self.p4(
                self.p4_lateral(conv4)
                + nn.Upsample(size=conv4.shape[-2:], mode="nearest")(p5)
            )
            outs.append(p4)

            p3 = self.p3(
                self.p3_lateral(conv3)
                + nn.Upsample(size=conv3.shape[-2:], mode="nearest")(p4)
            )
            outs.append(p3)
        # outs = [outs[:]]
        return outs

    def forward(self, x):
        with torch.no_grad():
            feats = self.resnet[0](x)

        feats = self.get_ps(feats)

        # apply the class head and box head on top of layers
        outs = []
        for f, a in zip(feats, self.anchors):
            cls = self.cls_head(f)
            bbox = self.bbox_head(f)
            outs.append((cls, bbox, a(f)))
        return outs
    
import numpy as np
import torch
import torch.nn as nn
from detection_utils import compute_bbox_targets

class LossFunc(nn.Module):

    def forward(self, classifications, regressions, anchors, gt_clss, gt_bboxes, gamma = 2, alpha = 0.25):

        device = classifications.device
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            targets_cls = gt_clss[j, :, :]
            targets_bbox = gt_bboxes[j, :, :]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            positive_indices = (targets_cls > 0).view(-1)
            non_negative_indices = (targets_cls >= 0).view(-1)
            num_positive_anchors = positive_indices.sum()

            if num_positive_anchors == 0:
                bce = -(torch.log(1.0 - classification))
                cls_loss = bce
                classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0).float().to(device))
                continue

            
            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.to(device)
            targets[non_negative_indices, :] = 0
            targets[positive_indices, targets_cls[positive_indices] - 1] = 1

            # bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            bce = -(alpha * targets * ((1.0 - classification) ** alpha) * torch.log(classification) + (1.0 - alpha) * (1.0 - targets) * (classification ** alpha) * torch.log(1.0 - classification))
            cls_loss = bce
            
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(device))
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            targets_bbox = targets_bbox[positive_indices, :]
            bbox_reg_target = compute_bbox_targets(anchor[positive_indices, :].reshape(-1,4), targets_bbox.reshape(-1,4))
            targets = bbox_reg_target.to(device)
            regression_diff = torch.abs(targets - regression[positive_indices, :])
            regression_losses.append(regression_diff.mean())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)


def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s: %(levelname)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.getLogger().handlers = []
    if not len(logging.getLogger().handlers): 
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

def logger(tag, value, global_step):
    if tag == '':
       logging.info('')
    else:
       logging.info(f'  {tag:>15s} [{global_step:07d}]: {value:5f}')

def main(_):
    setup_logging()
    torch.set_num_threads(4)
    torch.manual_seed(FLAGS.seed)
    set_seed(FLAGS.seed)
    
    dataset_train = CocoDataset('train', seed=FLAGS.seed,
        preload_images=FLAGS.preload_images > 0,
        transform=transforms.Compose([Normalizer(), Resizer(), transforms.RandomHorizontalFlip(p=0.5)]))
    dataset_val = CocoDataset('val', seed=0, 
        preload_images=FLAGS.preload_images > 0,
        transform=transforms.Compose([Normalizer(), Resizer(), transforms.RandomHorizontalFlip(p=0.5)]))
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, pin_memory=True) 
    
    model = RetinaNet(p67=True, fpn=True)

    num_classes = dataset_train.num_classes
    # device = torch.device('cuda:0')
    # For Mac users
    device = torch.device("mps") 
    model.to(device)

    writer = SummaryWriter(FLAGS.output_dir, max_queue=1000, flush_secs=120)
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.lr, 
                                momentum=FLAGS.momentum, 
                                weight_decay=FLAGS.weight_decay)
    
    # milestones = [int(x) for x in FLAGS.lr_step]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=milestones, gamma=0.1)

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=2000)
    
    optimizer.zero_grad()
    dataloader_iter = None
    
    times_np, cls_loss_np, bbox_loss_np, total_loss_np = [], [], [], []
    lossFunc = losses.LossFunc()
     
    for i in range(FLAGS.max_iter):
        iter_start_time = time.time()
        
        if dataloader_iter is None or i % len(dataloader_iter) == 0:
            dataloader_iter = iter(dataloader_train)
        
        image, cls, bbox, is_crowd, image_id, _ = next(dataloader_iter)
        
        if len(bbox) == 0:
            continue

        image = image.to(device, non_blocking=True)
        bbox = bbox.to(device, non_blocking=True)
        cls = cls.to(device, non_blocking=True)

        outs = model(image)
        pred_clss, pred_bboxes, anchors = get_detections(outs)
        gt_clss, gt_bboxes = compute_targets(anchors, cls, bbox)
        
        pred_clss = pred_clss.sigmoid()
        classification_loss, regression_loss = lossFunc(pred_clss, pred_bboxes,
                                                        anchors, gt_clss,
                                                        gt_bboxes)
        cls_loss = classification_loss.mean()
        bbox_loss = regression_loss.mean()
        total_loss = cls_loss + bbox_loss
        
        if np.isnan(total_loss.item()):
            logging.error(f'Loss went to NaN at iteration {i+1}')
            break
        
        if np.isinf(total_loss.item()):
            logging.error(f'Loss went to Inf at iteration {i+1}')
            break
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Some logging
        lr = scheduler.get_last_lr()[0]
        total_loss_np.append(total_loss.item())
        cls_loss_np.append(cls_loss.item())
        bbox_loss_np.append(bbox_loss.item())
        times_np.append(time.time() - iter_start_time)
                      
        if (i+1) % log_every == 0:
            print('')
            writer.add_scalar('iteration_rate', len(times_np) / np.sum(times_np), i+1)
            logger('iteration_rate', len(times_np) / np.sum(times_np), i+1)
            writer.add_scalar('loss_box_reg', np.mean(bbox_loss_np), i+1)
            logger('loss_box_reg', np.mean(bbox_loss_np), i+1)
            writer.add_scalar('lr', lr, i+1)
            logger('lr', lr, i+1)
            writer.add_scalar('loss_cls', np.mean(cls_loss_np), i+1)
            logger('loss_cls', np.mean(cls_loss_np), i+1)
            writer.add_scalar('total_loss', np.mean(total_loss_np), i+1)
            logger('total_loss', np.mean(total_loss_np), i+1)
            times_np, cls_loss_np, bbox_loss_np, total_loss_np = [], [], [], []


        if (i+1) % FLAGS.save_every == 0:
            torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_{i+1}.pth')
            
        if (i+1) % FLAGS.val_every == 0 or (i+1) == FLAGS.max_iter:
            logging.info(f'Validating at {i+1} iterations.')
            val_dataloader = DataLoader(dataset_val, num_workers=3, collate_fn=collater)
            result_file_name = f'{FLAGS.output_dir}/results_{i+1}_val.json'
            model.eval()
            validate(dataset_val, val_dataloader, device, model, result_file_name, writer, i+1)
            model.train()

    torch.save(model.state_dict(), f'{FLAGS.output_dir}/model_final.pth')

    # Save prediction result on test set
    dataset_test = CocoDataset('test', preload_images=False,
                               transform=transforms.Compose([Normalizer(), Resizer()])) 
    test_dataloader = DataLoader(dataset_test, num_workers=1, collate_fn=collater)
    result_file_name = f'{FLAGS.output_dir}/results_{FLAGS.max_iter}_test.json'
    model.eval()
    test(dataset_test, test_dataloader, device, model, result_file_name)

if __name__ == '__main__':
    app.run(main)
