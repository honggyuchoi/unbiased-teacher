#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import logging
import argparse
import time
import os.path
import pickle
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.visualizer import Visualizer, VisImage
from detectron2.data.detection_utils import convert_image_to_rgb

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import (
    UBTeacherTrainer, 
    BaselineTrainer
)
from ubteacher.engine.mocov2trainer import MoCov2Trainer
from ubteacher.engine.mocov1trainer import MoCov1Trainer
# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.data.dataset_mapper import VisualizeMapper 
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from typing import Dict, List, Optional, Tuple
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sn
from matplotlib.ticker import NullFormatter


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

voc_names =  ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] 

def visualize_cosine_sim(cfg, model, datasets='coco'):
    # build dataloader(has annotation)
    mapper = VisualizeMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)
    
    # partially execute a model to get embeded features 
    embeded_features = []
    embeded_features_classes = []
    model.eval()
    output_dir = f'./visualize_cosine_mat'
    os.makedirs(output_dir, exist_ok=True)
    model_name =  cfg.MODEL.WEIGHTS
    model_name = model_name.split('/')[-2]

    global coco_names, voc_names

    if datasets == 'coco':
        names = coco_names
        num_classes = len(coco_names)
    elif datasets == 'voc':
        names = voc_names
        num_classes = len(voc_names)
    else:
        raise NotImplementedError

    for idx, inputs in enumerate(data_loader):
        
        if idx % 10 == 0:
            print('idx: ', idx)
        # from rcnn.py
        images = model.preprocess_image(inputs)
        if "instances" in inputs[0]:
            gt_instances = [x["instances"].to('cuda') for x in inputs]
        else:
            gt_instances = None

        try:
            features = model.backbone(images.tensor)
        except:
            features = model.backbone_q(images.tensor)

        del images

        features = [features['p2'], features['p3'], features['p4'], features['p5']]
        
        # extract embeded feature with gt boxes
        box_features = model.roi_heads.box_pooler(features, [x.gt_boxes for x in gt_instances])
        
        try:
            box_features = model.roi_heads.box_head(box_features)
        except:
            box_features = model.roi_heads.box_head_q(box_features)
        del features

        gt_classes = gt_instances[0].gt_classes
        
        embeded_features.append(box_features.detach().cpu())
        embeded_features_classes.append(gt_classes.detach().cpu())
  

    embeded_features = torch.cat(embeded_features) # num, feature_dim
    embeded_features_classes = torch.cat(embeded_features_classes) # num

    mask = torch.ones((num_classes, num_classes))
    mask = torch.tril(mask, diagonal=-1)

    proto_type = torch.zeros((num_classes, embeded_features.shape[1]))    
    for i in range(num_classes):
        classwise_embeded_features = embeded_features[(embeded_features_classes == i)]
        proto_type[i] = torch.mean(classwise_embeded_features, dim=0)

    norm_proto_type = proto_type / torch.clamp(proto_type.norm(dim=1)[:,None], min=1E-6)

    cosine_sim_mat = torch.mm(norm_proto_type, norm_proto_type.T) * mask

    fig = plt.figure(figsize=(12, 9))
    sn.set(font_scale=1.0 if num_classes < 50 else 0.8)  # for label size
    labels = (0 < len(names) < 99) and len(names) == num_classes  # apply names to ticklabels
    sn.heatmap(cosine_sim_mat, vmin=0.0, vmax=1.0, annot=num_classes < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                xticklabels=names,
                yticklabels=names).set_facecolor((1, 1, 1))
    fig.tight_layout()
    fig.savefig(Path(output_dir) /  f'cosine_mat_{model_name}.png', dpi=250)  

def visualize_confusion_mat(cfg, model, datasets='coco', score_threshold=0.1, iou_threshold=0.5, ignore_background=False, normalize="row"):
    '''
        model: evaluated model
        num_classes: number of classes in specific dataset(MS COCO:80, PASCAL VOC: 20)
        conf_threshold: filtering predictions that have low confidence(classification score)
        iou_threshold: matching 
    '''
    global coco_names, voc_names

    if datasets == 'coco':
        names = coco_names
        num_classes = len(coco_names)
    elif datasets == 'voc':
        names = voc_names
        num_classes = len(voc_names)
    else:
        raise NotImplementedError
    
    # build dataloader(has annotation)
    conf_mat = np.zeros((num_classes + 1, num_classes + 1))

    mapper = VisualizeMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)

    model.eval()
    output_dir = f'./visualize_confusion_mat'
    os.makedirs(output_dir, exist_ok=True)
    model_name =  cfg.MODEL.WEIGHTS
    model_name = model_name.split('/')[-2]
    for idx, inputs in enumerate(data_loader):
        # import pdb
        # pdb.set_trace()
        if idx % 10 == 0:
            print('idx: ', idx)
        # from rcnn.py
        images = model.preprocess_image(inputs)
        if "instances" in inputs[0]:
            gt_instances = [x["instances"].to('cuda') for x in inputs]
        else:
            gt_instances = None

        features = model.backbone(images.tensor)
        # select top k proposals 
        proposals_rpn, _ = model.proposal_generator(images, features)

        proposals_roih, ROI_predictions = model.roi_heads(
            images,
            features,
            proposals_rpn,
            targets=None,
            compute_loss=False,
            branch="unsup_data_weak",
        )
        # confidence threshold 
        select = (proposals_roih[0].scores > score_threshold)
        proposals_roih = [proposals_roih[0][select]]   

        # matching predictions - gt boxes 
        for predictions_per_image, targets_per_image in zip(proposals_roih, gt_instances):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, predictions_per_image.pred_boxes)
            
            if has_gt:
                matched_vals, matched_idx = match_quality_matrix.max(dim=0)
            else:
                matched_vals = torch.zeros(match_quality_matrix.shape[1]).to(match_quality_matrix.device)

            # To find false negative 
            if has_gt:
                check = torch.zeros(len(targets_per_image))
                check[matched_idx] = 1
                gt_classes = targets_per_image.gt_classes
                matched_classes = targets_per_image.gt_classes[matched_idx]
                #background class 0~79는 object classes, 80은 background class # 
                # false positive assigned to background 
                matched_classes[matched_vals < 0.5] = num_classes 

            else:
                matched_classes = torch.full((match_quality_matrix.shape[1],), num_classes)

            predicted_classes = predictions_per_image.pred_classes



        for i in range(matched_classes.shape[0]):
            conf_mat[matched_classes[i], predicted_classes[i]] += 1
        for i in range(check.shape[0]):
            if check[i] == 0:
                conf_mat[gt_classes[i], num_classes] += 1 
            
    # normalize 
    if normalize == "row":
        array = conf_mat / (conf_mat.sum(1).reshape(num_classes + 1, 1) + 1E-6)
    elif normalize == "column":
        array = conf_mat / (conf_mat.sum(0).reshape(1, num_classes + 1) + 1E-6)
    else:
        raise NotImplementedError

    # draw plot
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    fig = plt.figure(figsize=(12, 9))
    sn.set(font_scale=1.0 if num_classes < 50 else 0.8)  # for label size
    labels = (0 < len(names) < 99) and len(names) == num_classes  # apply names to ticklabels
    sn.heatmap(array, annot=num_classes < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                xticklabels=names + ['background FN'] if labels else "auto",
                yticklabels=names + ['background FP'] if labels else "auto").set_facecolor((1, 1, 1))
    fig.axes[0].set_xlabel('Predicted')
    fig.axes[0].set_ylabel('True')
    fig.tight_layout()
    fig.savefig(Path(output_dir) /  f'confusion_matrix_{model_name}.png', dpi=250)
    
def visualize_tsne(cfg, model, datasets='coco'):
    # build dataloader(has annotation)
    mapper = VisualizeMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)
    
    # partially execute a model to get embeded features 
    embeded_features = []
    embeded_features_classes = []
    model.eval()
    output_dir = f'./visualize_tsne'
    os.makedirs(output_dir, exist_ok=True)
    model_name =  cfg.MODEL.WEIGHTS
    model_name = model_name.split('/')[-1]

    global coco_names, voc_names

    if datasets == 'coco':
        names = coco_names
        num_classes = len(coco_names)
    elif datasets == 'voc':
        names = voc_names
        num_classes = len(voc_names)
    else:
        raise NotImplementedError

    for idx, inputs in enumerate(data_loader):
        if idx % 10 == 0:
            print('idx: ', idx)
        # from rcnn.py
        images = model.preprocess_image(inputs)
        if "instances" in inputs[0]:
            gt_instances = [x["instances"].to('cuda') for x in inputs]
        else:
            gt_instances = None
        try:
            features = model.backbone(images.tensor)
        except:
            features = model.backbone_q(images.tensor)
        features = [features['p2'], features['p3'], features['p4'], features['p5']]
        # extract embeded feature with gt boxes
        box_features = model.roi_heads.box_pooler(features, [x.gt_boxes for x in gt_instances])
        try:
            box_features = model.roi_heads.box_head(box_features)
        except:
            box_features = model.roi_heads.box_head_q(box_features)
        gt_classes = gt_instances[0].gt_classes

        embeded_features.append(box_features.detach().cpu())
        embeded_features_classes.append(gt_classes.detach().cpu())
        
  
    embeded_features = torch.cat(embeded_features) # num, feature_dim
    embeded_features_classes = torch.cat(embeded_features_classes) # num
    
    # Select 100 instance for each classes
    x = []
    y = []
    for i in range(num_classes):
        temp_embeded_features = embeded_features[embeded_features_classes == i]
        temp_embeded_features_classes = embeded_features_classes[embeded_features_classes == i]
        temp_len = temp_embeded_features.shape[0]

        if temp_len > 100:
            select = np.random.choice(temp_len, 100, replace=False)
            x.append(temp_embeded_features[select])
            y.append(temp_embeded_features_classes[select])
        else:
            x.append(temp_embeded_features)
            y.append(temp_embeded_features_classes)
    features = torch.cat(x, dim=0)
    labels = torch.cat(y, dim=0)
    
    n_components = 2
    perplexities=list(range(10,15,10))
    #perplexities=list(range(10,150,10))
    
    #palette = sn.color_palette(None, num_classes)
    color_m = []
    for i in range(num_classes):
        if (i < 20) or (i >= 40 and i < 60): 
            color_m.append(plt.cm.tab20b(i%20))
        else:
            color_m.append(plt.cm.tab20c(i%20))



    # Y = torch.randn((80,2))
    # labels = torch.arange(80)

    for i, perplexity in enumerate(perplexities):
        _, ax = plt.subplots()
        tsne = TSNE(n_components=n_components, init='pca',
                    random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(features)
        
        sn.scatterplot(x=Y[:,0], y=Y[:,1], hue=labels, ax=ax, legend='full', palette=color_m)

        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')

        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize='xx-small', labels=names)

        plt.savefig(Path(output_dir) / f'tsne_{model_name}_{perplexity}.png', dpi=300)
        plt.pause(0.0001)
        plt.clf()


def visualize_proposals(cfg, model):
    # build dataloader(has annotation)
    mapper = VisualizeMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)

    model.eval()
    max_vis_prop = 20
    objectness_thresh = cfg.VISUALIZATION.RPN_THRESHOLD
    output_dir = f'./visualize_proposals_{objectness_thresh}'
    os.makedirs(output_dir, exist_ok=True)
    for idx, inputs in enumerate(data_loader):
        # import pdb
        # pdb.set_trace()
        if idx % 10 == 0:
            print('idx: ', idx)
        # from rcnn.py
        images = model.preprocess_image(inputs)
        if "instances" in inputs[0]:
            gt_instances = [x["instances"].to('cuda') for x in inputs]
        else:
            gt_instances = None

        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features)
        
        input = inputs[0]
        prop = proposals[0]
        img = input["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), model.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        #box_size = min(len(prop.proposal_boxes), max_vis_prop)

        objectness = torch.sigmoid(prop.objectness_logits)
        foreground_idx = (objectness >= objectness_thresh)
        
        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(
            boxes=prop.proposal_boxes[foreground_idx].tensor.cpu().numpy()
        )
        prop_img = v_pred.get_image()
        vis_img = np.concatenate((anno_img, prop_img), axis=1)
        #vis_img = vis_img.transpose(2, 0, 1)
        v=VisImage(vis_img, 0.5)
        v.save( output_dir + f'/testing_image_{idx}.png')
        if idx == 100:
            break
@torch.no_grad()
def visualize_predictions(cfg, model):
    # build dataloader(has annotation)
    mapper = VisualizeMapper(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)

    model.eval()
    max_vis_prop = 20
    objectness_thresh = cfg.VISUALIZATION.RPN_THRESHOLD
    score_thresh = cfg.VISUALIZATION.ROI_THRESHOLD
    output_dir = f'./visualize_predictions_{objectness_thresh}_{score_thresh}'
    os.makedirs(output_dir, exist_ok=True)
    for idx, inputs in enumerate(data_loader):
        # import pdb
        # pdb.set_trace()
        if idx % 10 == 0:
            print('idx: ', idx)
        # from rcnn.py
        images = model.preprocess_image(inputs)
        if "instances" in inputs[0]:
            gt_instances = [x["instances"].to('cuda') for x in inputs]
        else:
            gt_instances = None

        features = model.backbone(images.tensor)
        proposals_rpn, _ = model.proposal_generator(images, features)

        objectness = torch.sigmoid(proposals_rpn[0].objectness_logits)
        foreground_idx = (objectness >= objectness_thresh)
        
        proposals_rpn = proposals_rpn[0][foreground_idx]

        proposals_roih, ROI_predictions = model.roi_heads(
            images,
            features,
            [proposals_rpn],
            targets=None,
            compute_loss=False,
            branch="unsup_data_weak",
        )
        
        select = (proposals_roih[0].scores >= score_thresh)
        proposals_roih = proposals_roih[0][select]

        input = inputs[0]
        img = input["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), model.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        #box_size = min(len(prop.proposal_boxes), max_vis_prop)
        
        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(
            boxes=proposals_roih.pred_boxes.tensor.cpu().numpy(),
            labels=proposals_roih.pred_classes.cpu().numpy()
        )
        prop_img = v_pred.get_image()
        vis_img = np.concatenate((anno_img, prop_img), axis=1)
        #vis_img = vis_img.transpose(2, 0, 1)
        v=VisImage(vis_img, 0.5)
        v.save(output_dir + f'/testing_image_{idx}.png')
        if idx == 100:
            break

def label_and_proposals(proposals, targets, box_features, roi_heads):
    if box_features.dim() > 2:
        box_features = torch.flatten(box_features, start_dim=1)
    
    for proposals_per_image, targets_per_image in zip(proposals, targets):
        has_gt = len(targets_per_image) > 0
        
        match_quality_matrix = pairwise_iou(
            targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
        )
        matched_idxs, matched_labels = roi_heads.proposal_matcher(match_quality_matrix) # which gt, foreground
        
        matched_classes = targets_per_image.gt_classes[matched_idxs]

        # select foreground proposals
        foreground_box = proposals_per_image[matched_labels == 1]
        foreground_box_classes = matched_classes[matched_labels == 1]
        foreground_box_features = box_features[matched_labels == 1]
        # sampled_idxs, gt_classes = self._sample_proposals(
        #     matched_idxs, matched_labels, targets_per_image.gt_classes
        # )

        return foreground_box_features, foreground_box_classes

def plot_embedding(plot, data, label, dataset, plot_num, show=None):
    # param data:data
    # param label:label
    # param title:title of output
    # param show:(int) if you have too much proposals to draw, you can draw part of them
    # return: tsne-image
    # 클래스 별로 갯수 제한을 해서 plot하는게 필요할수도??
    if show is not None:
        temp = [i for i in range(len(data))]
        random.shuffle(temp)
        data = data[temp]
        data = data[:show]
        label = torch.tensor(label)[temp]
        label = label[:show]
        label.numpy().tolist()

    x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
    data = (data - x_min) / (x_max - x_min) # norm data

    # go through all the samples
    # data = data.tolist()
    # label = label.squeeze().tolist()
    
    # for i in range(len(data)):
    #     plt.text(data[i][0], data[i][1], ".", fontsize=18, color=plt.cm.tab20(label[i] / 20))
    # import pdb
    # pdb.set_trace()
    label_list = list(set(label))
    
    for i in label_list:
        
        idx = (np.array(label) == i)
        plot.scatter(data[idx,0], data[idx,1], s=10, alpha=0.6, color=plt.cm.tab20(i / 20))

    if 'voc' in dataset and plot_num == 1:
        object_list = np.array(["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"])
        plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize='xx-small', labels=object_list[label_list])

    return plot

# def preprocess_image(size_divisibility, pixel_mean, pixel_std, batched_inputs: List[Dict[str, torch.Tensor]]):
#     """
#     Normalize, pad and batch the input images.
#     """
#     if torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'

#     images = [x["image"].to(device) for x in batched_inputs]
#     images = [(x - pixel_mean) / pixel_std for x in images]
#     images = ImageList.from_tensors(images, size_divisibility)
#     return images



def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    # teacher - student only roi_heads
    elif cfg.SEMISUPNET.Trainer == "mocov1":
        Trainer = MoCov1Trainer
    # teacher - student backbone to roi_heads 
    elif cfg.SEMISUPNET.Trainer == "mocov2":
        Trainer = MoCov2Trainer
    else:
        raise ValueError("Trainer Name is not found.")

    if cfg.SEMISUPNET.Trainer == "ubteacher" \
        or cfg.SEMISUPNET.Trainer == "mocov2":

        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        DetectionCheckpointer(
            ensem_ts_model, save_dir=cfg.OUTPUT_DIR
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

    else:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
    test_model = model_teacher
    if cfg.VISUALIZATION.TYPE == "rpn":
        visualize_proposals(cfg, test_model)
    elif cfg.VISUALIZATION.TYPE == "roi":
        visualize_predictions(cfg, test_model)
    elif cfg.VISUALIZATION.TYPE == "tsne":
        visualize_tsne(cfg, test_model)
    elif cfg.VISUALIZATION.TYPE == "conf_mat":
        visualize_confusion_mat(cfg, test_model, 'coco', )
    elif cfg.VISUALIZATION.TYPE == "cosine_sim":
        visualize_cosine_sim(cfg, test_model)
    elif cfg.VISUALIZATION.TYPE == "umap":
        visualize_umap(cfg, test_model)
    else:
        raise NotImplementedError
    

if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
