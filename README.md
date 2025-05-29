## ULS23 Model Container With Custom Trainers
This repository contains the code to build a GrandChallenge algorithm usign the [nnUnetv2](https://github.com/MIC-DKFZ/nnUNet/tree/master) framework. It is based on the [oncology-ULS-fast-for-challenge](https://github.com/DIAGNijmegen/oncology-ULS-fast-for-challenge/tree/main) repository with additional custom trainers which explore different data augmentation strategies.

The fully annotated data used to train a model using the custom trainers can be found on Zenodo under ["AIMI course ULS project: cropped fully annotated data and baseline weights"](https://zenodo.org/records/15355959). The data was preprocessed and the model was trained on the ICIS computing cluster. More information on the usage of the cluster can be found on the [iCIS Intra Wiki](https://wiki.icis-intra.cs.ru.nl/Cluster) as well as the [GitLab Wiki](https://gitlab.science.ru.nl/das-dl/gpu-cluster-wiki).

This repo has the following structure:
- `/architecture/extensions/nnunetv2` contains the extensions to the nnUNetv2 framework that should be merged with your local nnunet install.
- `/architecture/input/` contains an example of a stacked VOI image and the accompanying spacings file. Uncommenting line 63 and 66 in the Dockerfile will allow you to run your algorithm locally with this data and check whether it runs inference correctly. 
- `/process.py` is where the model is loaded, predictions are made and postprocessing is applied. If you're testing this model locally and want to use a CPU instead of a GPU, you can do this by changing 'cuda' to 'cpu' in line 23 of process.py.
- `/preprocess.sh` is an example of how to preprocess the raw data using a batch job on the cluster.
- `/train.sh` is an example of how to run a job to train the model on the cluster using a custom trainer with your desired settings.

This project has been a collaboration between Jakob Jerše (s1156064), Dariga Shokayeva (s1158777), and Daniel Duhnev (s1158773) at Radboud University.
