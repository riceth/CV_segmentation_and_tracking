close all;
clear;



% load the image
i = imread('library.jpg');
imshow(i);
title('my image');
bw = rgb2gray(i);

figure; % opens a new image window

% Convert to grayscale
imshow(bw);
title('black and white');
bw = im2double(bw);
figure;
% Convert to double precision
Sx = [-1 0 1;
      -2 0 2;
      -1 0 1];

Sy = [-1 -2 -1;
       0  0  0;
       1  2  1];

% Apply convolution: Sx detects vertical edges, Sy detects horizontal edges

Gx = conv2(bw, Sx, 'same');
Gy = conv2(bw, Sy, 'same');

% Display results
imshowpair(abs(Gx), abs(Gy), 'montage');
title('Gx (left) and Gy (right)');
figure;

% Combine Gx and Gy into a full edge map
G = sqrt(Gx.^2 + Gy.^2);
imshow(G);
title('Gradient Magnitude');
figure;

% Gaussian Smoothing: Reduce noise before edge detection
sigma = 1;   % try values like 1, 2, 3
bw_smooth = imgaussfilt(bw, sigma);

imshow(bw_smooth);
title(['Gaussian Smoothed, sigma = ', num2str(sigma)]);

% % Different sigma values to test
% sigmas = [0.5 1 2 4];
% 
% figure;
% 
% for k = 1:length(sigmas)
% 
%     sigma = sigmas(k);
% 
%     % Apply Gaussian smoothing
%     BW_smooth = imgaussfilt(bw, sigma);
% 
%     % Display result
%     subplot(2,2,k);
%     imshow(BW_smooth);
%     title(['\sigma = ', num2str(sigma)]);
% 
% end
% sgtitle('Gaussian Smoothing Comparison');

% Run Sobel again to improve smoothing
Gx = conv2(bw_smooth, Sx, 'same');
Gy = conv2(bw_smooth, Sy, 'same');
G = sqrt(Gx.^2 + Gy.^2);

imshow(G);
title('Gradient After Gaussian Smoothing');
figure;
% Laplacian of Gaussian (LoG): Use second derivative edge detection

% Method 1: Gaussian → Laplacian
% Create kernels
G = fspecial('gaussian', [5 5], 2);
L = fspecial('laplacian', 0.2);

% Apply separately
bw_smooth = conv2(bw, G, 'same');
LoG1 = conv2(bw_smooth, L, 'same');

imshow(abs(LoG1), []);
title('LoG - Method 1');
figure;

% Method 2: Combine Kernels First
% Combine kernels
LoG_kernel = conv2(G, L, 'same');
% Apply once
LoG2 = conv2(bw, LoG_kernel, 'same');

imshow(abs(LoG2), []);
title('LoG - Method 2');

figure;
imshowpair(abs(LoG2), abs(LoG1), 'montage');
