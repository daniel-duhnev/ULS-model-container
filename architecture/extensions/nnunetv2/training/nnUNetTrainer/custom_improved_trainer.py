from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast

import numpy as np
from typing import Union, List, Tuple


class CustomImprovedTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device):
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
        # Get base transforms from parent first
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
            ignore_label,
        )

        # Parameters for shallow spatial transform
        elastic_prob = 0.0  # no elastic deformation
        rotation_prob = 0.2
        scaling_prob = 0.2
        scaling_range = (0.9, 1.1)

        patch_size_spatial = patch_size[1:] if do_dummy_2d_data_aug else patch_size

        # Convert degrees to radians for rotation
        if len(rotation_for_DA) == 0:
            rotation_rad = (0, 0)
        else:
            rotation_rad = tuple(np.deg2rad(x) for x in rotation_for_DA[:2])
            if len(rotation_rad) == 1:
                rotation_rad = (rotation_rad[0], rotation_rad[0])

        # Replace any SpatialTransform with shallow spatial version
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

        # Override intensity transforms only
        for idx, t in enumerate(transforms.transforms):
            if isinstance(t, RandomTransform):
                inner = t.transform

                if isinstance(inner, MultiplicativeBrightnessTransform):
                    transforms.transforms[idx] = RandomTransform(
                        MultiplicativeBrightnessTransform(
                            multiplier_range=BGContrast((0.95, 1.05)),
                            synchronize_channels=inner.synchronize_channels,
                            p_per_channel=inner.p_per_channel,
                        ),
                        apply_probability=0.05,
                    )
                elif isinstance(inner, GammaTransform):
                    transforms.transforms[idx] = RandomTransform(
                        GammaTransform(
                            gamma=BGContrast((0.95, 1.05)),
                            p_invert_image=0,
                            synchronize_channels=inner.synchronize_channels,
                            p_per_channel=inner.p_per_channel,
                            p_retain_stats=inner.p_retain_stats,
                        ),
                        apply_probability=0.05,
                    )
                elif isinstance(inner, GaussianNoiseTransform):
                    transforms.transforms[idx] = RandomTransform(
                        GaussianNoiseTransform(
                            noise_variance=(0, 0.002),
                            p_per_channel=inner.p_per_channel,
                            synchronize_channels=inner.synchronize_channels,
                        ),
                        apply_probability=0.02,
                    )
                elif isinstance(inner, GaussianBlurTransform):
                    transforms.transforms[idx] = RandomTransform(
                        GaussianBlurTransform(
                            blur_sigma=(0.1, 0.5),
                            benchmark=inner.benchmark,
                            synchronize_channels=inner.synchronize_channels,
                            synchronize_axes=inner.synchronize_axes,
                            p_per_channel=inner.p_per_channel,
                        ),
                        apply_probability=0.05,
                    )
                elif isinstance(inner, ContrastTransform):
                    transforms.transforms[idx] = RandomTransform(
                        ContrastTransform(
                            contrast_range=BGContrast((0.95, 1.05)),
                            preserve_range=inner.preserve_range,
                            synchronize_channels=inner.synchronize_channels,
                            p_per_channel=inner.p_per_channel,
                        ),
                        apply_probability=0.05,
                    )
                elif isinstance(inner, SimulateLowResolutionTransform):
                    transforms.transforms[idx] = RandomTransform(
                        SimulateLowResolutionTransform(
                            scale=(0.8, 1),
                            synchronize_channels=inner.synchronize_channels,
                            synchronize_axes=inner.synchronize_axes,
                            ignore_axes=inner.ignore_axes,
                            allowed_channels=inner.allowed_channels,
                            p_per_channel=inner.p_per_channel,
                        ),
                        apply_probability=0.05,
                    )

        self.print_to_log_file("Transform pipeline:\n" + "\n".join([str(t) for t in transforms.transforms]),
                               also_print_to_console=True,
                               add_timestamp=False)

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
            ignore_label=ignore_label,
        )
