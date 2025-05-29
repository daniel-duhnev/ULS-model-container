from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from typing import Union, List, Tuple
import numpy as np

class CustomSpatialOnlyTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 100

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int, ...]],
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,
        do_dummy_2d_data_aug,
        use_mask_for_norm=None,
        is_cascaded=False,
        foreground_labels=None,
        regions=None,
        ignore_label=None
    ) -> BasicTransform:
        # Get default full pipeline from nnU-Net
        transforms = super().get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label
        )

        # Remove all intensity augmentations (wrapped in RandomTransform)
        transforms.transforms = [
            t for t in transforms.transforms if not isinstance(t, RandomTransform)
        ]

        return transforms

    def get_validation_transforms(
        self,
        deep_supervision_scales,
        is_cascaded=False,
        foreground_labels=None,
        regions=None,
        ignore_label=None
    ) -> BasicTransform:
        return super().get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label
        )
