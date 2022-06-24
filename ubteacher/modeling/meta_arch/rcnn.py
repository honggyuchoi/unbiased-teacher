# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from ubteacher.utils import box_jittering
import detectron2.utils.comm as comm
@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, 
        batched_inputs, 
        branch="supervised",
        given_proposals=None, 
        val_mode=False,
        uncertainty_threshold=None,
        cur_threshold=None,
        jittering_times=None,
        jittering_frac=None
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
        
        #cls label and reg label are different
        elif branch == "supervised_with_uncertainty":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, 
                features, 
                gt_instances, 
                pseudo_label_reg=True,
                uncertainty_threshold=uncertainty_threshold
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, 
                features, 
                proposals_rpn, 
                gt_instances, 
                branch=branch,
                pseudo_label_reg=True,
                uncertainty_threshold=uncertainty_threshold
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        # generate pseudo label candidate
        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        # generate pseudo label candidate
        elif branch == "unsup_data_weak_with_uncertainty":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
                cur_threshold=cur_threshold,
                jittering_times=jittering_times,
                jittering_frac=jittering_frac
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

# For mocov2 
@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN_MoCo(GeneralizedRCNN):
    def forward(
        self, batched_inputs, 
        branch="supervised", 
        given_proposals=None, 
        val_mode=False, 
        queue_obj=None,
        temperature=None,
        noise=False,
        split=None # split for unlabel_q / unlabel_k
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        # generate pseudo label candidate
        # No match with gt label
        # just prediction 
        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            # ROI_predictions: ??
            return {}, proposals_rpn, proposals_roih, ROI_predictions, features
        
        elif branch == "unsup_data_weak_box_features":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions, box_features = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            # ROI_predictions: ??
            return {}, proposals_rpn, proposals_roih, ROI_predictions, box_features

        elif branch == "unsup_data_weak_with_projection":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions, projection_features = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            # ROI_predictions: ??
            return proposals_rpn, proposals_roih, projection_features

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
        
        elif branch == "contrastive_label" :
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, 
                features, 
                proposals_rpn, 
                gt_instances, 
                branch=branch, 
                queue_obj=queue_obj, 
                queue_label=queue_label,
                temperature=temperature,
                noise=noise
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch is "contrastive_unlabel":
            # compute loss -> unlabel_q
            # compute proposals_rpn -> unlabel_q, unlabel_k
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, split=split 
            )
            # compute regular loss(cls loss, reg loss) -> unlabel_q
            # compute contrastive loss -> unlabel_q, unlabel_k
            _, detector_losses = self.roi_heads(
                images, 
                features, 
                proposals_rpn,
                gt_instances, 
                branch=branch, 
                queue_obj=queue_obj, 
                temperature=temperature,
                noise=noise,
                split=split
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "generate_rpn_proposals":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, gt_instances
            )

            return _, proposals_rpn, [], None, features
        else:
            raise NotImplementedError


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN_MoCov1(GeneralizedRCNN):
    def forward(self, 
        batched_inputs, 
        branch="supervised", 
        given_proposals=None, 
        val_mode=False, 
        split=None,
        noise=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        # generate pseudo label candidate
        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )
            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            # ROI_predictions: ??
            return {}, proposals_rpn, proposals_roih, ROI_predictions, features

        elif branch == "val_loss":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )
            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
        
        # label_data_k -> cont features, noise cont features, cls loss, reg loss
        # label_data_q -> cont features, noise cont features, cls loss, reg loss
        elif branch == "contrastive_label" or branch == "contrastive_unlabel":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, split=split
            )
            # cont_features -> for queue update
            # noise_cont_features -> for contrastive loss
            # detector_losses: cls loss + reg loss
            box_projected_features,target_box_features,sampled_proposals,detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                split=split,
                noise=noise
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # cls loss + reg_loss / source_features / target_features /sampled_proposals_rpn
            return losses, box_projected_features, target_box_features, sampled_proposals

        elif branch == "generate_rpn_proposals":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, gt_instances
            )
            return _, proposals_rpn, [], None, features

        else:
            raise NotImplementedError           


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN_Uncertainty(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        # generate pseudo label candidate
        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "val_loss_with_uncertainty":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            pred_instances, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )
            
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, pred_instances, [], None

        # thresholding + smoothl1 loss
        elif (branch == "uncertainty_threshold") \
            or (branch == "weighted_smoothl1_loss") \
                or (branch == 'uncertainty_threshold_with_NLL') \
                    or (branch == 'uncertainty_threshold_with_bhattacharyya_loss'):
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        else:
            raise NotImplementedError

# IS this valid for uncertainty reg and smoothL1 reg 
@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN_cont(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False,
        pseudo_label_reg=False, # If true, compute reg loss using uncertainty threhold 
        uncertainty_threshold=None,
        proposals_roih_unsup_k=None, 
        prediction=False, 
        box_jitter=False,  
        jitter_times=1, 
        jitter_frac=0.06,
        class_aware_cont=True,
        cont_score_threshold=None,
        cont_loss_type=None,
        temperature=None,
        targets=None,
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)
        
        # memory to gpu and backbone operation
        if (branch == 'unsup_forward') or (branch == 'predictions_with_cont'):
            pass
        else:
            images = self.preprocess_image(batched_inputs)
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            features = self.backbone(images.tensor)

        ##################################################################################

        if (branch == "supervised") or (branch=="supervised_smoothL1"):
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances,
                pseudo_label_reg=pseudo_label_reg,
                uncertainty_threshold=uncertainty_threshold
            )
            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch,
                pseudo_label_reg=pseudo_label_reg,
                uncertainty_threshold=uncertainty_threshold
            )

            # To set unused_parameter = False
            temp = torch.rand((0,1024),device='cuda:0', requires_grad=True)
            temp_loss = self.roi_heads.feat_predictor(self.roi_heads.box_projector(temp)).sum()

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses['loss_temp'] = temp_loss
            return losses, [], [], None

        # generate pseudo label candidate
        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )
            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )
            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "val_loss_with_uncertainty":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )
            # roi_head lower branch
            pred_instances, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, pred_instances, [], None

        elif branch == "predictions_with_cont":
            unlabeled_data_k, unlabeled_data_q, unlabeled_data_q2 = batched_inputs
            batch_size = len(unlabeled_data_k)
            images_k = self.preprocess_image(unlabeled_data_k)
            images_q = self.preprocess_image(unlabeled_data_q)
            images_q2 = self.preprocess_image(unlabeled_data_q2)

            features = self.backbone(torch.cat([images_k.tensor, images_q.tensor, images_q2.tensor], dim=0))
            del images_q, images_q2
            features_k = {}
            features_q_all = {}

            for key in features.keys():
                features_k[key], features_q_all[key] = features[key].split([batch_size, batch_size * 2])

            proposals_rpn, _ = self.proposal_generator(
                images_k, features_k, None, compute_loss = False
            )
            proposals_roih, _ = self.roi_heads(
                images_k, features, proposals_rpn, targets=None, compute_loss=False, branch='unsup_data_weak'
            )
            del proposals_rpn

            features_q_all = [features_q_all[f] for f in self.roi_heads.box_in_features]
            targets = self.extract_projection_features(
                features,
                proposals_roih,
                prediction=False,
                box_jitter=box_jitter,
                jitter_time=jitter_times,
                jitter_frac=jitter_frac                
            )

            return {}, _, proposals_roih, _, targets 
                

        
        elif branch == "unsup_forward":

            assert isinstance(batched_inputs, list), 'Need two dataset unlabel_q and unlabel_q_2'
            unlabel_data_q, unlabel_data_q_2 = batched_inputs
            record_all_unlabel_data = {}
            len_unlabel_data_q = len(unlabel_data_q)
            
            images_q = self.preprocess_image(unlabel_data_q)
            images_q_2 = self.preprocess_image(unlabel_data_q_2)
            
            gt_instances = [x["instances"].to(self.device) for x in unlabel_data_q]
            # generate image features
            # unlabel_data_q -> proposal_losses and detector_losses
            # unlabel_data_q, unlabel_data_q_2 -> cont loss 
            features = self.backbone(torch.cat([images_q.tensor, images_q_2.tensor], dim=0))
            features_q = {}
            
            for key in features.keys():
                features_q[key] = features[key][:len_unlabel_data_q]
            # generate proposals_rpn of unlabel_data_q
            
            proposals_rpn, proposal_losses = self.proposal_generator(
                images_q, features_q, gt_instances, 
                pseudo_label_reg=pseudo_label_reg,
                uncertainty_threshold=uncertainty_threshold
            )
            
            del images_q_2.tensor, images_q_2

            _, detector_losses = self.roi_heads(
                images_q, features_q, proposals_rpn, gt_instances, branch=branch, 
                pseudo_label_reg=pseudo_label_reg,
                uncertainty_threshold=uncertainty_threshold
            )
            del images_q, features_q, proposals_rpn, gt_instances

            # _, detector_losses = model.roi_heads(images_q, features_q, proposals_rpn, gt_instances, branch=branch)
            record_all_unlabel_data.update(proposal_losses)
            record_all_unlabel_data.update(detector_losses)

            #compute cont sources
            features = [features[f] for f in self.roi_heads.box_in_features]
            sources = self.extract_projection_features(features, \
                                                        proposals_roih_unsup_k, \
                                                        prediction=prediction, \
                                                        box_jitter=box_jitter, \
                                                        jitter_times=jitter_times, \
                                                        jitter_frac=jitter_frac
                                                    )
            sym_cont_loss = self.get_sym_cont_loss(sources, targets, len_unlabel_data_q, cont_score_threshold, cont_loss_type, temperature, class_aware_cont)
            record_all_unlabel_data.update({
                'loss_cont': sym_cont_loss 
            })
            
            return record_all_unlabel_data, [], [], None

    def extract_projection_features(self, image_features, rois, prediction=True, box_jitter=False, jitter_times=1, jitter_frac=0.06, feature_noise=False, temp_threshold=0.1):
        total_rois = rois + rois

        predicted_classes = [x.pred_classes[x.scores > temp_threshold] for x in total_rois]
        predicted_boxes = [x.pred_boxes[x.scores > temp_threshold] for x in total_rois]
        predicted_scores = [x.scores[x.scores > temp_threshold] for x in total_rois]
        
        image_size = [x.image_size for x in total_rois]

        assert len(image_size) == len(predicted_classes)
        assert len(predicted_classes) == len(predicted_scores)
        assert len(predicted_scores) == len(predicted_boxes)

        if box_jitter == True:
            predicted_boxes, predicted_classes, predicted_scores = box_jittering(predicted_boxes, predicted_classes, predicted_scores, image_size, times=jitter_times, frac=jitter_frac)

        predicted_scores = torch.cat(predicted_scores, dim=0)
        predicted_classes = torch.cat(predicted_classes, dim=0)
                
        box_features = self.roi_heads.box_pooler(image_features, predicted_boxes) # 7x7xC feature map
        if feature_noise:
            raise NotImplementedError
        box_features = self.roi_heads.box_head(box_features)
        box_features = self.roi_heads.box_projector(box_features)
        if prediction:
            box_features = self.roi_heads.feat_predictor(box_features)
        
        box_features = F.normalize(box_features, dim=1)
        assert predicted_classes.shape[0] == predicted_scores.shape[0]
        assert predicted_classes.shape[0] == box_features.shape[0]

        targets = {
            'features': box_features,
            'gt_classes': predicted_classes.detach(),
            'gt_scores': predicted_scores.detach() 
        }
        return targets

    def get_sym_cont_loss(self, source, target, batch_size, cont_score_threshold, cont_loss_type, temperature, class_aware_cont):
        # from student 
        source_features_q, source_features_q_2 = source['features'].chunk(2)
        source_gt_classes_q, source_gt_classes_q_2 = source['gt_classes'].chunk(2)
        source_gt_scores = source['gt_scores']
        source_gt_scores[(source_gt_scores < cont_score_threshold)] = 0.0
        source_gt_scores_q, source_gt_scores_q_2 = source_gt_scores.chunk(2)

        target_gt_scores = target['gt_scores']
        target_gt_scores[(target_gt_scores < cont_score_threshold)] = 0.0
        # from teacher 
        
        target_features_q_2 = torch.zeros((100 * batch_size, 128), device=self.device, requires_grad=False)
        target_gt_classes_q_2 = torch.zeros(100 * batch_size, device=self.device, requires_grad=False).fill_(-1.0)
        target_gt_scores_q_2 = torch.zeros(100 * batch_size, device=self.device, requires_grad=False)

        num_of_targets = int(target['features'].shape[0] / 2)
        _, target_features_q_2[:num_of_targets] = target['features'].chunk(2) # (N,128)
        _, target_gt_classes_q_2[:num_of_targets] = target['gt_classes'].chunk(2) # (N)
        _, target_gt_scores_q_2[:num_of_targets] = target_gt_scores.chunk(2) #(N)

        if comm.get_world_size() > 1:
            target_features_q_2 = concat_all_gather(target_features_q_2)
            target_gt_classes_q_2 = concat_all_gather(target_gt_classes_q_2)
            target_gt_scores_q_2 = concat_all_gather(target_gt_scores_q_2)

        valid = (target_gt_classes_q_2 != -1.0)
        target_features_q_2 = target_features_q_2[valid]
        target_gt_classes_q_2 = target_gt_classes_q_2[valid]
        target_gt_scores_q_2 = target_gt_scores_q_2[valid]
        
        if cont_loss_type == 'infoNCE':
            loss_first = self.cont_loss(source_features_q, source_gt_classes_q, source_gt_scores_q, target_features_q_2, target_gt_classes_q_2, target_gt_scores_q_2, temperature, class_aware_cont)
        elif cont_loss_type == 'byol':
            loss_first = self.byol_loss(source_features_q, source_gt_classes_q, source_gt_scores_q, target_features_q_2, target_gt_classes_q_2, target_gt_scores_q_2)
        else:
            raise NotImplementedError

        del source_features_q, source_gt_classes_q, source_gt_scores_q, target_features_q_2, target_gt_classes_q_2, target_gt_scores_q_2

        target_features_q = torch.zeros((100 * batch_size, 128), device=self.device, requires_grad=False)
        target_gt_classes_q = torch.zeros(100 * batch_size, device=self.device, requires_grad=False).fill_(-1.0)
        target_gt_scores_q = torch.zeros(100 * batch_size, device=self.device, requires_grad=False)

        num_of_targets = int(target['features'].shape[0] / 2)
        target_features_q[:num_of_targets], _ = target['features'].chunk(2) # (N,128)
        target_gt_classes_q[:num_of_targets], _ = target['gt_classes'].chunk(2) # (N)
        target_gt_scores_q[:num_of_targets], _ = target_gt_scores.chunk(2) #(N)

        if comm.get_world_size() > 1:
            target_features_q = concat_all_gather(target_features_q)
            target_gt_classes_q = concat_all_gather(target_gt_classes_q)
            target_gt_scores_q = concat_all_gather(target_gt_scores_q)

        valid = (target_gt_classes_q != -1.0)
        target_features_q = target_features_q[valid]
        target_gt_classes_q = target_gt_classes_q[valid]
        target_gt_scores_q = target_gt_scores_q[valid]

        if cont_loss_type == 'infoNCE':
            loss_second = self.cont_loss(source_features_q_2, source_gt_classes_q_2, source_gt_scores_q_2, target_features_q, target_gt_classes_q, target_gt_scores_q, temperature, class_aware_cont)
        elif cont_loss_type == 'byol':
            loss_second = self.byol_loss(source_features_q_2, source_gt_classes_q_2, source_gt_scores_q_2, target_features_q, target_gt_classes_q, target_gt_scores_q)
        else:
            raise NotImplementedError

        del source_features_q_2, source_gt_classes_q_2, source_gt_scores_q_2, target_features_q, target_gt_classes_q, target_gt_scores_q

        loss = 0.5 * (loss_first + loss_second) # symmetric loss function

        return loss.mean()

    # if train model with multi gpus, matrix is not NxN
    def cont_loss(self, source_features, source_gt_classes, source_gt_scores, target_features, target_gt_classes, target_gt_scores, temperature, class_aware_cont):
        self_mask = torch.zeros((source_features.shape[0], target_features.shape[0]), device=self.device).float().fill_diagonal_(1.0)
        class_match_mask = torch.eq(source_gt_classes.view(-1,1), target_gt_classes.view(1,-1)).float()
        score_weight = torch.mm(source_gt_scores.view(-1,1), target_gt_scores.view(1,-1))

        weighted_matched_mask = (score_weight * class_match_mask).fill_diagonal_(1.0)
        #weighted_matched_mask = (score_weight * class_match_mask).fill_diagonal_(1.0)

        cos_sim = torch.mm(source_features, target_features.T.detach())
        del target_features, target_gt_classes, target_gt_scores 
        cos_sim /= temperature

        logits_row_max, _ = torch.max(cos_sim, dim=1, keepdim=True)
        cos_sim = cos_sim - logits_row_max.detach()

        log_prob = cos_sim - torch.log((torch.exp(cos_sim)*(1.0-self_mask)).sum(dim=1, keepdim=True))

        if class_aware_cont:
             loss = -(log_prob * weighted_matched_mask).sum(dim=1) / torch.count_nonzero(weighted_matched_mask,dim=1)
        else:
            loss = -(log_prob * self_mask).sum(dim=1)
        return loss.mean()

    # L = 2 - 2 * || z_i * z_j || 
    # cont_loss에서 같은 클래스 이미지인데 negative로 학습될수있는 문제가 있다(score가 낮아서 classwise positive pair가 되지 않은 녀석들)
    # class aware byol loss를 적용하면 negative pair가 적용되지 않기때문에 이런 negative한 경우가 발생하지 않을 것이다. 
    def byol_loss(self, source_features, source_gt_classes, source_gt_scores, target_features, target_gt_classes, target_gt_scores):
        self_mask = torch.zeros(source_features.shape[0], source_features.shape[0]).float().fill_diagonal_(1.0).cuda()
        
        class_match_mask = torch.eq(source_gt_classes.view(-1,1), target_gt_classes.view(1,-1)).float()

        score_weight = torch.mm(source_gt_scores.view(-1,1), target_gt_scores.view(1,-1))

        weighted_matched_mask = (score_weight * class_match_mask).fill_diagonal_(1.0)

        cos_sim = torch.mm(source_features, target_features.T.detach())
        
        loss =  - 2 * (cos_sim * weighted_matched_mask).sum(dim=1) / torch.count_nonzero(weighted_matched_mask, dim=1)

        return loss.mean()

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN_IoU(GeneralizedRCNN):
    def forward(
        self, 
        batched_inputs, 
        branch="supervised", 
        given_proposals=None, 
        val_mode=False,
        uncertainty_threshold=None,
        training_with_jittering=False,
        jittering_times=None,
        jittering_frac=None
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # for labeled data 
        # compute iou loss for training iou branch
        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, 
                features, 
                proposals_rpn, 
                gt_instances, 
                branch=branch,
                training_with_jittering=training_with_jittering,
                jittering_times=jittering_times,
                jittering_frac=jittering_frac
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        # cls label and reg label are different
        elif branch == "supervised_pseudo":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, 
                features, 
                gt_instances, 
                pseudo_label_reg=True,
                uncertainty_threshold=uncertainty_threshold
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, 
                features, 
                proposals_rpn, 
                gt_instances, 
                branch=branch,
                pseudo_label_reg=True,
                uncertainty_threshold=uncertainty_threshold
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        # generate pseudo label candidate
        elif (branch == "unsup_data_weak") or \
            (branch == "unsup_data_weak_with_iou"):
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

# Utils for multi process training
@torch.no_grad()
def concat_all_gather(tensor):
    # batch_size * 100
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    rank = torch.distributed.get_rank()
    # Switch first block and corrent rank block
    tensors_gather[0], tensors_gather[rank] = tensors_gather[rank], tensors_gather[0]

    output = torch.cat(tensors_gather, dim=0)
    return output
