import torch


class FeatureQueue:
    def __init__(self, cfg):
        self.queue_size = cfg.MOCO.QUEUE_SIZE
        self.feature_dim = cfg.MOCO.CONTRASTIVE_FEATURE_DIM
        self.labeled_contrastive_iou_thres = cfg.MOCO.LABELED_CONTRASTIVE_IOU_THRES
        self.unlabeled_contrastive_iou_thres = cfg.MOCO.UNLABELED_CONTRASTIVE_IOU_THRES
       
        self.queue = torch.randn(self.queue_size, self.feature_dim).detach()
        self.queue_label = torch.empty(self.queue_size).fill_(-1).long().detach()
        self.queue_ptr = torch.zeros(1, dtype=torch.long).detach()
        self.cycles = torch.zeros(1).detach()

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
