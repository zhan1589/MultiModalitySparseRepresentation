# Overview

This repository contains the source codes for the "A multi-modality approach for enhancing 4D flow MRI via sparse representation" project.

The whole project dataset with both source code and data can be downloaded from Purdue University Research Repository (PURR) via the link:
https://purr.purdue.edu/publications/3872/1
The project is aim to evaluate and apply a multi-modality approach to enhance the blood flow measurement with 4D flow MRI and improve the hemodynamic analysis in cerebral aneurysms.

The multi-modality data employed in this project has been published in the following paper:
Brindise MC, Rothenberger S, Dickerhoff B, Schnell S, Markl M, Saloner D, Rayz VL, Vlachos PP. 2019 Multi-modality cerebral aneurysm haemodynamic analysis: in vivo 4D flow MRI, in vitro volumetric particle velocimetry and in silico computational fluid dynamics. J. R. Soc. Interface 16: 20190465. http://dx.doi.org/10.1098/rsif.2019.0465

Approval of all ethical and experimental procedures and protocols was granted by the institutional review boards at Purdue University, Northwestern Memorial Hospital, and San Francisco VA Medical Center.

The following section describe the folder structure and codes for the dataset published on PURR.

# Folder structure

## Data (stored in folder "data_depository")

- BT: the data folder for the basilar tip aneurysm.

- ICA: the data folder for the internal carotid artery aneurysm.

- File structure of each data folder: within the "BT" and "ICA" folders, the following file and subfolders exist:
	- basic_info.mat: a MATLAB data file containing basic information of the aneurysmal flow, including the fluid properties and pressure reference point locations.
	- geometry: The folder contains the Cartesian grids of different datasets and the coordinates of the wall points. 
	- MRI_frames: The folder contains the velocity fileds acquired with 4D flow MRI during one cardiac cycle.
	- CFD_frames: The folder contains the velocity fields obtained from patient-specific CFD simulations.
	- PTV_frames: The folder contains the velocity fields obtained from in vitro PTV measurement in cerebral aneurysmal models.
    - MRI_reconstructed: The folder contains the reconstructed velocity fields with the multi-modality sparse representation approach. The reconstruction was carried using main.py in the code package.
    - Hemodynamic_results: The folder contains the reconstructed pressure fields and the calculated WSS from the in vivo 4D flow MRI data and from the reconstructed flow fields. The calculations were carried using main_hemodynamic_evaluation.py in the code package.

## Codes (stored in folder "code_package")

- main.py: a sample python script to perform the flow-library based sparse representation flow reconstruction of the in vivo 4D flow MRI data. The script loads the data from "MRI_frames", constructs the flow-library based on the data from "CFD_frames" and "PTV_frames", performs the flow reconstruction, and stores the results in "MRI_reconstructed".

- main_hemodynamic_evaluation.py: a sample python script to calculate the pressure fields and wall shear stress (WSS). The script loads the data from "MRI_frames" and the results from "MRI_reconstructed", reconstructs pressure reconstruction, calcualtes WSS, and stores the results in "Hemodynamic_results".

- utility_functions.py: codes to reconstruct the flow fields using sparese-representation, reconstruct pressure field using the measurement-error based weighted least-squares [1], and calculates the WSS.

- ErrorEstimationFunctions.py, EvaluatePressureGradient.py, NumericalDifference.py, NumericalLinearOperators.py, UncertaintyAnalysisPressure.py: the in-house codes for post-processing the flow data.

# Funding

This work was supported by the National Institutes of Health under grants R21 NS106696 and R01 HL115267. 

# License

Creative Commons CC0 1.0 Universal license: https://creativecommons.org/publicdomain/zero/1.0/legalcode

# References

1. Zhang J, Brindise MC, Rothenberger S, Schnell S, Markl M, Saloner D, Rayz VL, Vlachos PP. 2020 4D Flow MRI Pressure Estimation Using Velocity Measurement-Error based Weighted Least-Squares. IEEE Trans. Med. Imaging 39, 1668–1680. (doi:10.1109/tmi.2019.2954697)







