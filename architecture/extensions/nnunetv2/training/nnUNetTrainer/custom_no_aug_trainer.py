from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
import numpy as np
from typing import Union, List, Tuple

class CustomNoAugTrainer(nnUNetTrainer):
    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        device
    ):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = 100

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int, ...]],
        rotation_for_DA,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        """
        Disable all random augmentations but keep mandatory crop/pad.
        """
        # Start with the default training transforms (includes crop/pad)
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

        # Zero out random spatial deform parameters
        for t in transforms.transforms:
            if isinstance(t, SpatialTransform):
                t.p_elastic_deform = 0.0
                t.p_rotation = 0.0
                t.p_scaling = 0.0
                # leave crop/pad settings intact

        # Remove any intensity RandomTransforms
        transforms.transforms = [
            t for t in transforms.transforms
            if not isinstance(t, RandomTransform)
        ]

        return transforms

    def get_validation_transforms(
        self,
        deep_supervision_scales: Union[List, Tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        return super().get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label
        )
