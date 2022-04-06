def generate_projected_box_features(model, boxes, features):
    """
        features: List(dim 0: image index)
        boxes: List(dim 0: image index)
    """
    box_features = model.roi_heads.box_pooler(features, boxes)
    box_features = model.roi_heads.box_head(features)
    box_projected_features = model.roi_heads.box_projector(features)

    return box_projected_features



@torch.no_grad()
def label_and_sample_proposals(
    self, proposals: List[Instances], targets: List[Instances]
) -> List[Instances]:
    """
    Prepare some proposals to be used to train the ROI heads.
    It performs box matching between `proposals` and `targets`, and assigns
    training labels to the proposals.
    It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
    boxes, with a fraction of positives that is no larger than
    ``self.positive_fraction``.

    Args:
        See :meth:`ROIHeads.forward`

    Returns:
        list[Instances]:
            length `N` list of `Instances`s containing the proposals
            sampled for training. Each `Instances` has the following fields:

            - proposal_boxes: the proposal boxes
            - gt_boxes: the ground-truth box that the proposal is assigned to
                (this is only meaningful if the proposal has a label > 0; if label = 0
                then the ground-truth box is random)

            Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
    """
    # Augment proposals with ground-truth boxes.
    # In the case of learned proposals (e.g., RPN), when training starts
    # the proposals will be low quality due to random initialization.
    # It's possible that none of these initial
    # proposals have high enough overlap with the gt objects to be used
    # as positive examples for the second stage components (box head,
    # cls head, mask head). Adding the gt boxes to the set of proposals
    # ensures that the second stage components will have some positive
    # examples from the start of training. For RPN, this augmentation improves
    # convergence and empirically improves box AP on COCO by about 0.5
    # points (under one tested configuration).
    if self.proposal_append_gt:
        proposals = add_ground_truth_to_proposals(targets, proposals)

    proposals_with_gt = []

    num_fg_samples = []
    num_bg_samples = []
    for proposals_per_image, targets_per_image in zip(proposals, targets):
        has_gt = len(targets_per_image) > 0
        match_quality_matrix = pairwise_iou(
            targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
        )
        # iou값 저장하자
        matched_vals, _ = match_quality_matrix.max(dim=0) # max iou, _
        
        matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix) # matching된 gt idx와 foreground/background
        sampled_idxs, gt_classes = self._sample_proposals(
            matched_idxs, matched_labels, targets_per_image.gt_classes
        )

        # Set target attributes of the sampled proposals:
        proposals_per_image = proposals_per_image[sampled_idxs]
        proposals_per_image.gt_classes = gt_classes # matching된 gt class(background면 self.num_class)
        # IoU 값 저장 
        if not proposals_per_image.has('iou'):
            proposals_per_image.set('iou', matched_vals[sampled_idxs])

        if has_gt:
            sampled_targets = matched_idxs[sampled_idxs]
            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            # NOTE: here the indexing waste some compute, because heads
            # like masks, keypoints, etc, will filter the proposals again,
            # (by foreground/background, or number of keypoints in the image, etc)
            # so we essentially index the data twice.
            for (trg_name, trg_value) in targets_per_image.get_fields().items():
                if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                    proposals_per_image.set(trg_name, trg_value[sampled_targets])
        # If no GT is given in the image, we don't know what a dummy gt value can be.
        # Therefore the returned proposals won't have any gt_* fields, except for a
        # gt_classes full of background label.

        num_bg_samples.append((gt_classes == self.num_classes).sum().item())
        num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
        proposals_with_gt.append(proposals_per_image)

    # Log the number of fg/bg samples that are selected for training ROI heads
    storage = get_event_storage()
    storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
    storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

    return proposals_with_gt