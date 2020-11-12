# WP6_HE_modelling
For decades the histopathological examination has been the ‘gold-standard’ method to diagnose diseases. A histopathological examination is performed by staining a tissue section with Hematoxylin and Eosin (H\&E) stain and visually identifying the structural changes in the stained tissue section. Although the histopathological examination is a routine procedure, it requires long sample preparation and staining times which restricts its application for ‘real-time’ disease diagnosis. Due to this reason, a label-free alternative technique which highlights tissue structures as efficiently as histopathological staining is required. In this contribution such an alternative technique is proposed, which is the combination of three non-linear optical modalities including coherent anti-Stokes Raman scattering (CARS), two-photon excitation fluorescence (TPEF) and second-harmonic generation (SHG). These three non-linear imaging modalities are label-free, non-invasive, and provide structural and molecular information of the sample under investigation. However, utilizing the molecular imaging modalities for disease assessment requires pathologists to interpret its image contrast and biological significance. To facilitate this interpretation task, a model that can automatically transform a non-linear multimodal image into an H\&E stained image without the need to stain the tissue pathologically, is desired. This work presents the transformation of the non-linear multimodal images into computationally stained H\&E images using generative adversarial networks (GANs) in a supervised and unsupervised approach. The computationally stained H\&E images obtained from both methods show similar tissue structures and can be utilized for diagnostic applications without altering the tissue. To the author’s best knowledge, it is the first time that non-linear multimodal images are computationally stained to histopathological images using GANs in an unsupervised manner.

[Workflow Image](https://hemospectrum.ipht-jena.de:8081/owncloud/index.php/s/5kZ7ofkNeQmUu36)

## Require packages
Keras

Tensorflow

Matplotlib

Pandas

os

Skimage

Warnings

Time

## Pre-requisite
Install [imageProcessing](https://github.com/pranitapradhan91/imageProcessing.git), [Utils](https://github.com/pranitapradhan91/WP6_HE_modelling.git) and [postProcess](https://github.com/pranitapradhan91/WP6_HE_modelling.git) packages

## Execution
1. Use make_csv.py can be used to make csv data file and adding all metadata.

2. First use preExecute_cycleGAN.py and preExecute_pix2pix.py to generate patches afor training cycleCGAN and Pix2Pix model. This script will be used seperately for train.csv and test.csv. After executing this script a MM_train or MM_test, HE_train or HE_test folders with all patches is created. Using the patches from this folder train.npz matrix is created.

3. Next is training of cycleCGAN and Pix2Pix models. This can be done using execute_cycleGAN.py and execute_Pix2Pix.py. These scripts initiate models from Utils package. 

4. Scripts execute_cycleGAN.py and execute_Pix2Pix.py can be used to load pre-trained models to predict test dataset.

5. Predictions can be post-processed also using scripts execute_cycleGAN.py and execute_Pix2Pix.py. The post-processing steps like removing patch-effect and contrast adjust of cycleCGAN generated images can be done by functions in postProcess.py.

6. Qualitiative and quantitative evaluation is done using qualitative_evaluation.py and quantitative_evaluation.py. Quantitative evaluation calculates MSE, CSS and SSIM metrics. Qualitative evaluation is done by selecting some patches and plotting in subplots.

## Dataset
[Preprocessed images](https://hemospectrum.ipht-jena.de:8081/owncloud/index.php/s/M1DuCdxWe5HsXA2)

[Pix2Pix numpy data](https://hemospectrum.ipht-jena.de:8081/owncloud/index.php/s/eZemDAoK6zRKv0j)

[CycleCGAN numpy data](https://hemospectrum.ipht-jena.de:8081/owncloud/index.php/s/Rln1PT426yWtjyS)

## Pre-trained models
[Pix2pix models](https://hemospectrum.ipht-jena.de:8081/owncloud/index.php/s/NOklxWMtg24j9Xr)

[Cyclegan models](https://hemospectrum.ipht-jena.de:8081/owncloud/index.php/s/ZpfO0ioJuuLsCMM)

[Results of Pix2Pix and cycleCGAN](https://hemospectrum.ipht-jena.de:8081/owncloud/index.php/s/VEc2AYF2xHD1GfX)
