import torch
from detectron2.structures.boxes import Boxes

class FeatureQueue:
    def __init__(self, cfg, store_score=True):
        self.queue_size = cfg.MOCO.QUEUE_SIZE
        self.feature_dim = cfg.MOCO.CONTRASTIVE_FEATURE_DIM
        self.labeled_contrastive_iou_thres = cfg.MOCO.LABELED_CONTRASTIVE_IOU_THRES
        self.unlabeled_contrastive_iou_thres = cfg.MOCO.UNLABELED_CONTRASTIVE_IOU_THRES
       
        self.queue = torch.randn(self.queue_size, self.feature_dim).detach()
        self.queue_label = torch.empty(self.queue_size).fill_(-1).long().detach()
        self.queue_ptr = torch.zeros(1, dtype=torch.long).detach()
        self.cycles = torch.zeros(1).detach()

        if store_score == True:
            self.queue_score = torch.empty(self.queue_size).fill_(-1.0).float().detach()
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, key, proposals, iou_threshold, background=False):
        label = torch.cat([p.gt_classes for p in proposals], dim=0)
        iou = torch.cat([p.iou for p in proposals], dim=0)
    
        if background == True:
            select_foreground = torch.nonzero((iou > iou_threshold)).view(-1)
            num_foreground = select_foreground.shape[0]

            select_background = torch.nonzero(((0.3 < iou) & (iou < 0.4))).view(-1)
            num_background = select_background.shape[0]

            if num_foreground < num_background:
                indices = torch.randperm(num_background)[:num_foreground] # without replacement
                select_background = select_background[indices]
                num_background = select_background.shape[0]

            num_select = num_foreground + num_background
            select = torch.cat([select_foreground, select_background], dim = 0) 
            
        else:
            select = torch.nonzero((iou > iou_threshold)).view(-1)
            num_select = select.shape[0]# select iou over 0.8 (No background)
        
        if num_select == 0:
            return 0

        print(f'selected projected feature: {num_select}')
        keys = key[select]
        labels = label[select]
        ious = iou[select]

        batch_size = keys.shape[0]
        if batch_size == 0:
            return 0
        
        ptr = int(self.queue_ptr)
        cycles = int(self.cycles)
        if ptr + batch_size <= self.queue.shape[0]:
            self.queue[ptr:ptr + batch_size, :] = keys
            self.queue_label[ptr:ptr + batch_size] = labels
        else:
            rem = self.queue.shape[0] - ptr
            self.queue[ptr:ptr + rem, :] = keys[:rem, :]
            self.queue_label[ptr:ptr + rem] = labels[:rem]

        ptr += batch_size
        if ptr >= self.queue.shape[0]:
            ptr = 0
            cycles += 1
        self.cycles[0] = cycles
        self.queue_ptr[0] = ptr
        return cycles
    
    @torch.no_grad()
    def _dequeue_and_enqueue_label(self, key, proposals, background=False):
        label = torch.cat([p.gt_classes for p in proposals], dim=0)
        iou = torch.cat([p.iou for p in proposals], dim=0)
    
        if background == True:
            select_foreground = torch.nonzero((iou > self.labeled_contrastive_iou_thres)).view(-1)
            num_foreground = select_foreground.shape[0]

            select_background = torch.nonzero(((0.3 < iou) & (iou < 0.4))).view(-1)
            num_background = select_background.shape[0]

            if num_foreground < num_background:
                indices = torch.randperm(num_background)[:num_foreground] # without replacement
                select_background = select_background[indices]
                num_background = select_background.shape[0]

            num_select = num_foreground + num_background
            select = torch.cat([select_foreground, select_background], dim = 0) 
            
        else:
            select = torch.nonzero((iou > self.labeled_contrastive_iou_thres)).view(-1)
            num_select = select.shape[0]# select iou over 0.8 (No background)

        print(f'selected projected feature(label data): {num_select}')
        keys = key[select]
        labels = label[select]
        ious = iou[select]

        batch_size = keys.shape[0]
        if batch_size == 0:
            return 0
        
        ptr = int(self.queue_ptr)
        cycles = int(self.cycles)
        if ptr + batch_size <= self.queue.shape[0]:
            self.queue[ptr:ptr + batch_size, :] = keys
            self.queue_label[ptr:ptr + batch_size] = labels
        else:
            rem = self.queue.shape[0] - ptr
            self.queue[ptr:ptr + rem, :] = keys[:rem, :]
            self.queue_label[ptr:ptr + rem] = labels[:rem]

        ptr += batch_size
        if ptr >= self.queue.shape[0]:
            ptr = 0
            cycles += 1
        self.cycles[0] = cycles
        self.queue_ptr[0] = ptr
        return cycles

    @torch.no_grad()
    def _dequeue_and_enqueue_unlabel(self, key, classes):
        label = torch.cat([c for c in classes], dim=0)

        batch_size = keys.shape[0]
        print(f'selected projected feature(unlabel data): {batch_size}')
        if batch_size == 0:
            return 0 

        if self.queue_size % batch_size != 0:
            print()
            print('update by unlabeled_k')
            print(self.queue_ptr, self.cycles, batch_size, self.queue.shape)
            print()

        ptr = int(self.queue_ptr)
        cycles = int(self.cycles)
        if ptr + batch_size <= self.queue.shape[0]:
            self.queue[ptr:ptr + batch_size, :] = keys
            self.queue_label[ptr:ptr + batch_size] = labels
        else:
            rem = self.queue.shape[0] - ptr
            self.queue[ptr:ptr + rem, :] = keys[:rem, :]
            self.queue_label[ptr:ptr + rem] = labels[:rem]

        ptr += batch_size
        if ptr >= self.queue.shape[0]:
            ptr = 0
            cycles += 1
        self.cycles[0] = cycles
        self.queue_ptr[0] = ptr
        return cycles

    def get_queue_info(self):
        print(self.queue_ptr)
        print(self.cycles)
    
        return self.queue_ptr, self.cycles
    
    @torch.no_grad()
    def get_queue(self):
        return self.queue
    @torch.no_grad()
    def get_queue_label(self):
        return self.queue_label

    @torch.no_grad()
    def _dequeue_and_enqueue_score(self, features, classes, scores):
        batch_size = features.shape[0]
        if batch_size == 0:
            return 0
        
        ptr = int(self.queue_ptr)
        cycles = int(self.cycles)
        if ptr + batch_size <= self.queue.shape[0]:
            self.queue[ptr:ptr + batch_size, :] = features
            self.queue_label[ptr:ptr + batch_size] = classes
            self.queue_score[ptr:ptr + batch_size] = scores
        else:
            rem = self.queue.shape[0] - ptr
            self.queue[ptr:ptr + rem, :] = features[:rem, :]
            self.queue_label[ptr:ptr + rem] = classes[:rem]
            self.queue_score[ptr:ptr + rem] = scores[:rem]

        ptr += batch_size
        if ptr >= self.queue.shape[0]:
            ptr = 0
            cycles += 1
        self.cycles[0] = cycles
        self.queue_ptr[0] = ptr
        return cycles

class ClasswiseFeatureQueue:
    def __init__(self, cfg):
        self.feature_dim = cfg.MOCO.CONTRASTIVE_FEATURE_DIM
        self.labeled_contrastive_iou_thres = cfg.MOCO.LABELED_CONTRASTIVE_IOU_THRES
        self.unlabeled_contrastive_iou_thres = cfg.MOCO.UNLABELED_CONTRASTIVE_IOU_THRES

        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.items_per_class = cfg.MOCO.CLASSWISE_QUEUE_ITEMS_PER_CLASS
        
        self.queue = torch.randn(self.num_classes, self.items_per_class, self.feature_dim).detach()
        self.queue_ptr = torch.zeros(self.num_classes, dtype=torch.long).detach()
        self.cycles = torch.zeros(self.num_classes, dtype=torch.long).detach()
        self.queue_label = torch.empty((self.num_classes, self.items_per_class)).fill_(-1).long().detach()
        self.queue_score = torch.empty((self.num_classes, self.items_per_class)).fill_(-1).detach()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key, proposals, iou_threshold, background=False):
        label = torch.cat([p.gt_classes for p in proposals], dim=0)
        iou = torch.cat([p.iou for p in proposals], dim=0)
    
        select = torch.nonzero((iou > iou_threshold)).view(-1)

        print(f'selected projected feature(label data): {select.numel()}')

        keys = key[select]
        labels = label[select]
        ious = iou[select]

        for i in torch.unique(labels):
            idx = (labels == i)
            select_keys = keys[idx]
            select_ious = ious[idx]

            batch_size = select_keys.shape[0]
            if batch_size == 0:
                continue
            ptr = int(self.queue_ptr[i])
            cycles = int(self.cycles[i])

            if ptr + batch_size <= self.queue[i].shape[0]:
                self.queue[i, ptr:ptr + batch_size, :] = select_keys
                self.queue_label[i, ptr:ptr + batch_size] = i
            else:
                rem = self.queue[i].shape[0] - ptr
                self.queue[i, ptr:ptr + rem, :] = select_keys[:rem, :]
                self.queue_label[i, ptr:ptr + rem] = i
            
            ptr += batch_size
            if ptr >= self.queue[i].shape[0]:
                ptr = 0
                cycles += 1
            self.queue_ptr[i] = ptr
            self.cycles[i] = cycles

        return 0

    @torch.no_grad()
    def _dequeue_and_enqueue_label(self, key, proposals, background=False):
        label = torch.cat([p.gt_classes for p in proposals], dim=0)
        iou = torch.cat([p.iou for p in proposals], dim=0)
    
        select = torch.nonzero((iou > self.labeled_contrastive_iou_thres)).view(-1)

        print(f'selected projected feature(label data): {select.numel()}')
        keys = key[select]
        labels = label[select]
        ious = iou[select]

        for i in torch.unique(labels):
            idx = (labels == i)
            select_keys = keys[idx]
            select_ious = ious[idx]

            batch_size = select_keys.shape[0]
            if batch_size == 0:
                continue
            ptr = int(self.queue_ptr[i])
            cycles = int(self.cycles[i])

            if ptr + batch_size <= self.queue[i].shape[0]:
                self.queue[i, ptr:ptr + batch_size, :] = select_keys
                self.queue_label[i, ptr:ptr + batch_size] = i
            else:
                rem = self.queue[i].shape[0] - ptr
                self.queue[i, ptr:ptr + rem, :] = select_keys[:rem, :]
                self.queue_label[i, ptr:ptr + rem] = i
            
            ptr += batch_size
            if ptr >= self.queue[i].shape[0]:
                ptr = 0
                cycles += 1
            self.queue_ptr[i] = ptr
            self.cycles[i] = cycles

        return 0

    @torch.no_grad()
    def _dequeue_and_enqueue_unlabel(self, keys, classes):
        labels = torch.cat([c for c in classes], dim=0)

        for i in torch.unique(labels):
            idx = (labels == i)
            select_keys = keys[idx]
            batch_size = select_keys.shape[0]

            ptr = int(self.queue_ptr[i])
            cycles = int(self.cycles[i])

            if ptr + batch_size <= self.queue[i].shape[0]:
                self.queue[i, ptr:ptr + batch_size, :] = select_keys
                self.queue_label[i, ptr:ptr + batch_size] = i
            else:
                rem = self.queue[i].shape[0] - ptr
                self.queue[i, ptr:ptr + rem, :] = select_keys[:rem, :]
                self.queue_label[i, ptr:ptr + batch_size] = i

            ptr += batch_size
            if ptr >= self.queue[i].shape[0]:
                ptr = 0
                cycles += 1
            self.cycles[i] = cycles
            self.queue_ptr[i] = ptr
        return 0
    
    def get_queue_info(self):
        print(self.queue_ptr)
        print(self.cycles)
    
        return self.queue_ptr, self.cycles
    
    @torch.no_grad()
    def get_queue(self):
        return self.queue.view(-1, self.queue.shape[-1])

    @torch.no_grad()
    def get_queue_label(self):
        return self.queue_label.view(-1)

    @torch.no_grad()
    def get_queue_score(self):
        return self.queue_score.view(-1)

    @torch.no_grad()
    def _dequeue_and_enqueue_score(self, features, classes, scores):
        assert (features.dim() == 2) or (features.dim() == 0), 'features dimension is not 2'
        batch_size = features.shape[0]
        if batch_size == 0:
            return 0
        
        for i in torch.unique(classes):
            idx = (classes == i)
            select_features = features[idx]
            select_scores = scores[idx]

            batch_size = select_features.shape[0]
            if batch_size == 0:
                continue

            ptr = int(self.queue_ptr[i])
            cycles = int(self.cycles[i])

            if ptr + batch_size <= self.queue[i].shape[0]:
                self.queue[i, ptr:ptr + batch_size, :] = select_features
                self.queue_label[i, ptr:ptr + batch_size] = i
                self.queue_score[i, ptr:ptr + batch_size] = select_scores
            else:
                rem = self.queue[i].shape[0] - ptr
                self.queue[i, ptr:ptr + rem, :] = select_features[:rem, :]
                self.queue_label[i, ptr:ptr + rem] = i
                self.queue_score[i, ptr:ptr + rem] = select_scores[:rem]

            ptr += batch_size
            if ptr >= self.queue[i].shape[0]:
                ptr = 0
                cycles += 1
            self.cycles[i] = cycles
            self.queue_ptr[i] = ptr
        
        return cycles

# From soft teacher 
# https://github.com/microsoft/SoftTeacher/blob/main/configs/soft_teacher/base.py
# translation and rescaling
def box_jittering(boxes, boxes_class, boxes_score, images_size, times=4, frac=0.01):
    def _aug_single(box, image_size): # image_size (height, width)
        # random translate and resizing

        box_scale = box.tensor[:, 2:4] - box.tensor[:, :2]
        box_scale = (
            box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
        )
        aug_scale = box_scale * frac  # [n,4]

        offset = (
            torch.randn(times, box.tensor.shape[0], 4, device=box.tensor.device) # normal distribution N(0,1)
            * aug_scale[None, ...]
        )
        new_box = box.tensor.clone()[None, ...].expand(times, box.tensor.shape[0], -1)
        new_box = new_box[:,:,:4] + offset

        # (x1,y1,x2,y2)
        new_box[:,:,0] = new_box[:,:,0].clamp(min=0.0)
        new_box[:,:,1] = new_box[:,:,1].clamp(min=0.0)

        new_box[:,:,2] = new_box[:,:,2].clamp(max=image_size[1]) # image width
        new_box[:,:,3] = new_box[:,:,3].clamp(max=image_size[0]) # image height

        return Boxes(new_box.reshape(-1,4))

    def _aug_single_class(box_class):
        new_class = box_class.clone()[None, ...].expand(times, box_class.shape[0]).reshape(-1)
        return new_class 

    def _aug_single_score(box_score):
        new_score = box_score.clone()[None, ...].expand(times, box_score.shape[0]).reshape(-1)
        return new_score

    jittered_boxes = [_aug_single(box, image_size) for box, image_size in zip(boxes,images_size)]
    jittered_classes = [_aug_single_class(box_class) for box_class in boxes_class]
    jittered_scores = [_aug_single_score(box_score) for box_score in boxes_score]

    return jittered_boxes, jittered_classes, jittered_scores



# compute loss with unlabel_data_q
    # Generate predicted features with unlabel_data_q_2
    # def unsup_forward(self, model, unlabel_data_q, unlabel_data_q_2, proposals_roih_unsup_k, branch='supervised'):
        
    #     record_all_unlabel_data = {}
    #     len_unlabel_data_q = len(unlabel_data_q)
        
    #     images_q = model.preprocess_image(unlabel_data_q)
    #     images_q_2 = model.preprocess_image(unlabel_data_q_2)
        
    #     gt_instances = [x["instances"].to(self.device) for x in unlabel_data_q]
    #     # generate image features
    #     # unlabel_data_q -> proposal_losses and detector_losses
    #     # unlabel_data_q, unlabel_data_q_2 -> cont loss 
    #     features = model.backbone(torch.cat([images_q.tensor, images_q_2.tensor], dim=0))
    #     features_q = {}
        
    #     for key in features.keys():
    #         features_q[key] = features[key][:len_unlabel_data_q]
    #     # generate proposals_rpn of unlabel_data_q
    #     proposals_rpn, proposal_losses = model.proposal_generator(
    #         images_q, features_q, gt_instances
    #     )
    #     del images_q_2.tensor, images_q_2

    #     proposals_rpn = model.roi_heads.label_and_sample_proposals(
    #         proposals_rpn, gt_instances
    #     )
    #     del gt_instances
    #     features_q = [features_q[f] for f in self.box_in_features]
    #     # detection loss of unlabel_data_q
    #     box_features_q = model.roi_heads.box_pooler(features_q, [x.proposal_boxes for x in proposals_rpn])
    #     box_features_q = model.roi_heads.box_head(box_features_q)
    #     predictions = model.roi_heads.box_predictor(box_features_q)
    #     detector_losses = model.roi_heads.box_predictor.losses(predictions, proposals_rpn)

    #     # _, detector_losses = model.roi_heads(images_q, features_q, proposals_rpn, gt_instances, branch=branch)
    #     record_all_unlabel_data.update(proposal_losses)
    #     record_all_unlabel_data.update(detector_losses)
    #     del images_q, features_q, box_features_q, proposals_rpn

    #     #compute cont loss
    #     features = [features[f] for f in self.box_in_features]
    #     sources = self.extract_projection_features(model, features, proposals_roih_unsup_k, prediction=self.cont_prediction_head, box_jitter=self.box_jitter)
        
    #     return record_all_unlabel_data, sources