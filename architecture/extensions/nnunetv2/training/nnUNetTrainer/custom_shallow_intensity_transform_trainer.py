from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from typing import Union, List, Tuple
import numpy as np

class ShallowIntensityTrainer(nnUNetTrainer):
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

        # Disable spatial randomness
        for t in transforms.transforms:
            if isinstance(t, SpatialTransform):
                t.p_elastic_deform = 0.0
                t.p_rotation = 0.0
                t.p_scaling = 0.0

        # Add conservative intensity transforms
        brightness_prob = 0.15
        gamma_prob = 0.15
        noise_prob = 0.05

        new_transforms = []
        for t in transforms.transforms:
            new_transforms.append(t)
            if isinstance(t, SpatialTransform):
                new_transforms.extend([
                    RandomTransform(
                        MultiplicativeBrightnessTransform(
                            multiplier_range=(0.85, 1.15),
                            synchronize_channels=False,
                            p_per_channel=1
                        ),
                        apply_probability=brightness_prob
                    ),
                    RandomTransform(
                        GammaTransform(
                            gamma=(0.8, 1.2),
                            p_invert_image=0,
                            synchronize_channels=False,
                            p_per_channel=1,
                            p_retain_stats=1
                        ),
                        apply_probability=gamma_prob
                    ),
                    RandomTransform(
                        GaussianNoiseTransform(
                            noise_variance=(0, 0.01),
                            p_per_channel=1,
                            synchronize_channels=True
                        ),
                        apply_probability=noise_prob
                    )
                ])

        transforms.transforms = new_transforms
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
