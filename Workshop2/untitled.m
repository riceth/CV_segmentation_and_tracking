% To clear image output after each run
clear all; close all;

% Edge detection finds INTENSITY changes, an edge is a sudden change in-
% brightness, and this is easier and computes fasterbecause grayscale is used for this so you operate
% on just 1 channel instead of 3

% Load the image
img = imread('library.jpg');

% Convert to grayscale
BW = rgb2gray(img);

% Display
figure; % to show picture in new tab
imshow(BW);
title('Grayscale Library Image');

% Sobel Filtering: Apply two filters to detect edges in different directions (horizontal and vertical).

% Convert grayscale to double for precision
BW = im2double(BW);

% Sobel kernel for X-direction (vertical edges)
Sx = [-1, 0, 1;
      -2, 0, 2;
      -1, 0, 1];

% Sobel kernel for Y-direction (horizontal edges)
Sy = [-1, -2, -1;
       0,  0,  0;
       1,  2,  1];

% Apply filters: Creates a filtered version showing edges
Gx = conv2(BW, Sx);
Gy = conv2(BW, Sy);

figure;
imshowpair(Gx, Gy, 'montage');
title('Sobel X (vertical edges) | Sobel Y (horizontal edges)');

% Gradient Magnitude: Combine both edge directions (Gx and Gy) into a single edge strength image.
% Calculate gradient magnitude
G = sqrt(Gx.^2 + Gy.^2);

figure;
imshow(G, []);
title('Gradient Magnitude (Combined Edges)');

% Gaussian Smoothing: Smooth the image to remove noise BEFORE edge detection, so you get cleaner edges.
% Create a simple Gaussian kernel
sigma = 1;

% Apply smoothing to original grayscale image
BW_smooth = imgaussfilt(BW, sigma);

% Display
figure;
imshowpair(BW, BW_smooth, 'montage');
title(sprintf('Original | Smoothed (sigma=%.1f)', sigma));

% Apply Sobel to smoothed image
Gx_smooth = conv2(BW_smooth, Sx);
Gy_smooth = conv2(BW_smooth, Sy);
G_smooth = sqrt(Gx_smooth.^2 + Gy_smooth.^2);

% Compare: original edges vs smoothed edges
figure;
imshowpair(G, G_smooth, 'montage');
title('Edges: No smoothing | With smoothing');

%% Laplacian of Gaussian (LoG): Use Laplacian (2nd derivative) for edge detection
% method 1
% Create kernels
gaussian = fspecial('gaussian', [5 5], 2);
laplacian = fspecial('laplacian', 0);

% Apply in sequence
step1 = conv2(BW, gaussian, 'same');
result1 = conv2(step1, laplacian, 'same');

% Display
figure;
imshow(abs(result1), []);
title('Method 1: Gaussian then Laplacian');

% method 2
% Combine kernels first
combined_kernel = conv2(gaussian, laplacian, 'same');

% Apply to image
result2 = conv2(BW, combined_kernel, 'same');

% Display
figure;
imshow(abs(result2), []);
title('Method 2: Combined kernel');
%%