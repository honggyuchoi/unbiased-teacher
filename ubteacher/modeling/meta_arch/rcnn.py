# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
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

    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results

# For mocov2 
@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN_MoCo(GeneralizedRCNN):
    def forward(
        self, batched_inputs, 
        branch="supervised", 
        given_proposals=None, 
        val_mode=False, 
        queue=None, 
        queue_label=None,
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
                queue=queue, 
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
                queue=queue, 
                queue_label=queue_label,
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