from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
import numpy as np
from typing import Union, List, Tuple

class CustomShallowSpatialTrainer(nnUNetTrainer):
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
        # Start with nnU-Net's default transforms
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

        # Shallow spatial-only augmentation parameters
        elastic_prob = 0.0  # Disabled completely
        rotation_prob = 0.2
        scaling_prob = 0.2
        scaling_range = (0.9, 1.1)

        patch_size_spatial = patch_size[1:] if do_dummy_2d_data_aug else patch_size

        # Convert rotation degrees to radians
        if len(rotation_for_DA) == 0:
            rotation_rad = (0, 0)
        else:
            rotation_rad = tuple(np.deg2rad(x) for x in rotation_for_DA[:2])
            if len(rotation_rad) == 1:
                rotation_rad = (rotation_rad[0], rotation_rad[0])

        # Replace default SpatialTransform with shallow-only version
        for idx, t in enumerate(transforms.transforms):
            if isinstance(t, SpatialTransform):
                transforms.transforms[idx] = SpatialTransform(
                    patch_size=patch_size_spatial,
                    patch_center_dist_from_border=0,
                    random_crop=False,
                    p_elastic_deform=elastic_prob,
                    elastic_deform_scale=(0, 0),
                    elastic_deform_magnitude=(0, 0),
                    p_synchronize_def_scale_across_axes=0,
                    p_rotation=rotation_prob,
                    rotation=rotation_rad,
                    p_scaling=scaling_prob,
                    scaling=scaling_range,
                    p_synchronize_scaling_across_axes=0,
                    bg_style_seg_sampling=True,
                    mode_seg='bilinear',
                    border_mode_seg='zeros',
                    center_deformation=True,
                    padding_mode_image='zeros'
                )

        # Remove any intensity transforms
        transforms.transforms = [
            t for t in transforms.transforms if not isinstance(t, RandomTransform)
        ]
        
        print("Final transform pipeline:")
        for t in transforms.transforms:
            print(t)

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
