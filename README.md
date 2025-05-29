# ULS23 Model Container With Custom Trainers

## Overview

This repository contains the code to build a GrandChallenge algorithm usign the [nnUnetv2](https://github.com/MIC-DKFZ/nnUNet/tree/master) framework. It is based on the [oncology-ULS-fast-for-challenge](https://github.com/DIAGNijmegen/oncology-ULS-fast-for-challenge/tree/main) repository with additional custom trainers which explore different data augmentation strategies. All models have been trained using [nnUNet v2.6.0](https://github.com/MIC-DKFZ/nnUNet/releases/tag/v2.6.0) and Python 3.10.12.

## The Data

The fully annotated data used to train a model using the custom trainers can be found on Zenodo under [AIMI course ULS project: cropped fully annotated data and baseline weights](https://zenodo.org/records/15355959). The fully annotated data contained a total of 7216 image-label pairs obtained from the following datasets:
| Datasets                                                 | Data Type       | Image-Label Pairs |
|----------------------------------------------------------|-----------------|:-----------------:|
| KiTS21 [Heller et al., 2023]                             | Kidney          | 333               |
| LiTS [Bilic et al., 2023]                                | Liver           | 888               |
| NIH-LN ABD [Roth et al., 2014]                           | Lymph nodes     | 558               |
| NIH-LN MED [Roth et al., 2014]                           | Lymph nodes     | 379               |
| LIDC-IDRI [Armato III et al., 2011; Jacobs et al., 2016] | Lung            | 2246              |
| LNDb [Pedrosa et al., 2021]                              | Lung            | 692               |
| MDSC-Lung [Antonelli et al., 2022]                       | Lung            | 76                |
| MDSC-Colon [Antonelli et al., 2022]                      | Colon           | 133               |
| MDSC-Pancreas [Antonelli et al., 2022]                   | Pancreas        | 283               |
| DeepLesion3D [Yan et al., 2018a]                         | Bone lesions    | 760               |
| Radboudumc-Bone [de Grauw et al., 2023a]                 | Pancreas        | 744               |
| Radboudumc-Pancreas [de Grauw et al., 2023a]             | Various lesions | 124               |

The data was preprocessed and the model was trained on the ICIS computing cluster. More information on the usage of the cluster can be found on the [iCIS Intra Wiki](https://wiki.icis-intra.cs.ru.nl/Cluster) as well as the [GitLab Wiki](https://gitlab.science.ru.nl/das-dl/gpu-cluster-wiki). Due to cluster limitations and restrictions, experiments were ran using 1/7th of each dataset totalling 1030 image-label pairs. Each experiment averaged around 6 hours of training time.

## Repository Structure

This repo has the following structure:
- `/architecture/extensions/nnunetv2` contains the extensions to the nnUNetv2 framework that should be merged with your local nnunet install.
- `/architecture/input/` contains an example of a stacked VOI image and the accompanying spacings file. Uncommenting line 63 and 66 in the Dockerfile will allow you to run your algorithm locally with this data and check whether it runs inference correctly. 
- `/process.py` is where the model is loaded, predictions are made and postprocessing is applied. If you're testing this model locally and want to use a CPU instead of a GPU, you can do this by changing 'cuda' to 'cpu' in line 23 of process.py.
- `/preprocess.sh` is an example of how to preprocess the raw data using a batch job on the cluster.
- `/train.sh` is an example of how to run a job to train the model on the cluster using a custom trainer with your desired settings.

## Custom Trainers
The table below summarises the custom trainers used for the various experiments exploring different data augmentation strategies.

| File Name                                       | Trainer Class Name                    | Functionality                                                                            |
|-------------------------------------------------|---------------------------------------|------------------------------------------------------------------------------------------|
| `custom_no_aug_trainer.py`                      | CustomNoAugTrainer                    | Training without any data augmentation, except for padding and cropping                  |
| `custom_trainer_spatial_only.py`                | CustomSpatialOnlyTrainer              | Removed all intensity transforms from default nnUNet trainer                             |
| `custom_trainer_intensity_only.py`              | CustomIntensityOnlyTrainer            | Removed all spatial transforms (except padding and cropping) from default nnUNet trainer |
| `shallow_spatial_transform_trainer.py`          | CustomShallowSpatialTrainer           | Modified spatial transforms and removed intensity ones completely                        |
| `shallow_spatial_default_intensity_trainer.py`  | ShallowSpatialDefaultIntensityTrainer | Modified spatial transforms and kept intensity ones from the default nnUNet trainer      |
| `custom_shallow_intensity_transform_trainer.py` | ShallowIntensityTrainer               |                                                                                          |
| `custom_shallow_trainer.py`                     | CustomShallowTrainer                  |                                                                                          |
| `custom_improved_trainer.py`                    | CustomImprovedTrainer                 | Custom spatial transforms and modified intensity pipeline with lesion-suited values      |


## References

- Antonelli, M., Reinke, A., Bakas, S., Farahani, K., Kopp-Schneider, A., Landman, B. A., … & Cardoso, M. J. (2022). The medical segmentation decathlon. *Nature Communications*, 13(1), 4128.
- Heller, N., Isensee, F., Trofimova, D., Tejpaul, R., Zhao, Z., Chen, H., … & Weight, C. (2023). The KiTS21 Challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase CT. *arXiv preprint* arXiv:2307.01984.
- Bilic, P., Christ, P., Li, H. B., Vorontsov, E., Ben-Cohen, A., Kaissis, G., … & Menze, B. (2023). The liver tumor segmentation benchmark (LiTS). *Medical Image Analysis*, 84, 102680.
- Roth, H., Lu, L., Seff, A., Cherry, K. M., Hoffman, J., Wang, S., Liu, J., Turkbey, E., & Summers, R. M. (2015). A new 2.5 D representation for lymph node detection in CT [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.AQIIDCNM
- Armato III, S. G., McLennan, G., Bidaut, L., et al. (2011). The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans. *Medical Physics*, 38(2), 915–931.
- Pedrosa, J., Aresta, G., Ferreira, C., Atwal, G., Phoulady, H. A., Chen, X., … & Campilho, A. (2021). LNDb challenge on automatic lung cancer patient management. *Medical Image Analysis*, 70, 102027.
- Yan, K., Wang, X., Lu, L., & Summers, R. M. (2018). DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning. *Journal of Medical Imaging*, 5(3), 036501.
- de Grauw, M. J. J., Scholten, E. Th., Smit, E. J., Rutten, M. J. C. M., Prokop, M., van Ginneken, B., & Hering, A. (2025). The ULS23 challenge: A baseline model and benchmark dataset for 3D universal lesion segmentation in computed tomography. *Medical Image Analysis*, 102, 103525. https://doi.org/10.1016/j.media.2025.103525

This project has been a collaboration between Jakob Jerše (s1156064), Dariga Shokayeva (s1158777), and Daniel Duhnev (s1158773) at Radboud University.
