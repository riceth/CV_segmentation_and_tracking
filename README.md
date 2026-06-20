# Image Segmentation & Feature-Based Object Tracking

Two classical-computer-vision pipelines built from first principles in MATLAB on a
descending-parachute image sequence.

## Task 1 — Automated segmentation
A parameter-free pipeline: HSV conversion, Gaussian smoothing, Otsu thresholding,
median-based blob selection, a hue-variance fallback for backlit scenes, and region-growing
refinement for distant captures. All parameters scale with image size.
**Result:** mean Dice coefficient **0.924** across 51 images (all > 0.80).

## Task 2 — Feature extraction & tracking
- Shape descriptors (solidity, circularity, eccentricity) and Histogram-of-Oriented-Gradients features, with a discriminative analysis (CoV, Spearman) of which best track canopy rotation.
- Two **from-scratch** Kalman filters (constant-velocity translation; constant-angular-velocity rotation) with data-derived noise parameters.
**Result:** test RMS of **3.14 px** (translation) and **1.07°** (rotation).

## Stack
MATLAB · classical CV · Kalman filtering · PCA

## Getting started
Open in MATLAB and run the main script for each task (`segmentation/main.m`, `tracking/main.m` — adjust to your filenames). Requires the Image Processing Toolbox.
