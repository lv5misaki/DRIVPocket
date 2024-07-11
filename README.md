# DRIVPocket

This repository contains the source code, trained models and the test sets for DRIVPokect.

# Introduction

The prediction of protein binding sites is a critical step in drug design. However, the task is challenging due to the small size of the binding sites and the significant variation in size between different proteins. To address these issues, we propose a novel protein binding site prediction model called DRIVPocket, based on dual-stream rotational invariance and voxel feature fusion. Remarkably, DRIVPocket can simultaneously predict the entire pocket region and the atoms near the cavity.

Specifically, we first represent the protein in two modalities, voxel and point cloud, and extract the relevant features via Dual Rotational Invariance Attention (DRIA) feature extraction and Rotational Invariance Down-Up (RID/RIU) sampling modules, respectively. We also fuse the point cloud features into voxel features through DRIA, which is based on shared channel attention and spatial attention. By design, DRIVPocket can better understand the chemical properties and structural features of the protein according to the fusion features.Additionally, DRIVPocket predicts the binding regions and binding atoms from the voxel features and the point cloud features, respectively. Finally, a more accurate segmentation prediction is obtained by integrating the two predictions.

Experiments show that DRIVPokcet improves by 5% on DVO and 1%-4% DCA Top-n prediction compared with previous state-of-the-art methods on four benchmark data sets.

[overview]()
# Datasets


# Train


# Test
