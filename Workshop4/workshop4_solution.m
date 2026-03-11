%% CMP9135M Workshop 4 — Interest Point Detection (Harris) — Solution
% Tasks covered:
% Task 1: Load + rotate and display side-by-side
% Task 2: Derivatives + structure tensor components (A,B,C)
% Task 3: Harris response R + threshold + non-max suppression + overlay
%         Compare with detectHarrisFeatures
% Task 4: Parameter impact (k, sigma, threshold) with quick sweep
%
% Notes:
% - Requires Image Processing Toolbox for imgaussfilt, imregionalmax
% - detectHarrisFeatures requires Computer Vision Toolbox

clear; close all; clc;

%% -------------------------
% Task 1: Get the data ready
% --------------------------
I0 = imread('cameraman.tif');       % built-in demo image
if size(I0,3) == 3
    I0 = rgb2gray(I0);
end

% create a second view using rotation
angleDeg = 20;
I1 = imrotate(I0, angleDeg, 'bilinear', 'crop');

figure('Name','Task 1: Original vs Rotated');
subplot(1,2,1); imshow(I0); title('Original');


%% ------------------------------------------------
% Task 2: Compute derivatives and second-moment M
% ------------------------------------------------
% Convert to double for computations
I = im2double(I0);
% Image derivatives (Sobel-like). imgradientxy returns x/y gradients.
[Ix, Iy] = imgradientxy(I);

% Gaussian smoothing parameter (sigma) for structure tensor aggregation
sigma = 1.5;

% Structure tensor components:
% A = Gσ * (Ix^2),  B = Gσ * (Iy^2),  C = Gσ * (Ix*Iy)
A = imgaussfilt(Ix.^2, sigma);
B = imgaussfilt(Iy.^2, sigma);
C = imgaussfilt(Ix.*Iy, sigma);

figure('Name','Task 2: Derivatives and Tensor Components');
subplot(2,3,1); imshow(I,[]);  title('I (double)');
subplot(2,3,2); imshow(Ix,[]); title('Ix');
subplot(2,3,3); imshow(Iy,[]); title('Iy');
subplot(2,3,4); imshow(A,[]);  title('A = G*(Ix^2)');
subplot(2,3,5); imshow(B,[]);  title('B = G*(Iy^2)');
subplot(2,3,6); imshow(C,[]);  title('C = G*(IxIy)');

%% ------------------------------------------------------
% Task 3: Harris response + threshold + non-max suppression
% -------------------------------------------------------
% Harris parameter
k = 0.04;

% Harris response:
% R = det(M) - k * trace(M)^2
% det(M) = A*B - C^2, trace(M) = A + B
R = (A .* B - C.^2) - k * (A + B).^2;

% Visualise Harris response as a heatmap
figure('Name','Task 3: Harris Response Heatmap');
imshow(R,[]); colormap(gca, jet); colorbar;
title('Harris Response R (heatmap)');

% Threshold: keep strong responses
tFrac = 0.01;                    % fraction of max(R)
t = tFrac * max(R(:));
Rth = (R > t);

% Non-maximum suppression: keep local maxima only
% imregionalmax identifies local peaks in R
Rmax = imregionalmax(R);

% Final corner mask: thresholded AND local maxima
cornerMask = Rth & Rmax;

% Get corner coordinates (row, col)
[rows, cols] = find(cornerMask);

% Plot corners overlay
figure('Name','Task 3: Corners Overlay (Custom Harris)');
imshow(I0); hold on;
plot(cols, rows, 'r.', 'MarkerSize', 10);
title(sprintf('Custom Harris corners (k=%.2f, sigma=%.1f, tFrac=%.3f)', k, sigma, tFrac));
hold off;

% Compare with MATLAB built-in Harris detector
try
    ptsBuiltIn = detectHarrisFeatures(I0, 'MinQuality', tFrac, 'FilterSize', 5);

    figure('Name','Task 3: Built-in detectHarrisFeatures');
    imshow(I0); hold on;
    plot(ptsBuiltIn.selectStrongest(200));
    title('MATLAB detectHarrisFeatures (top 200 strongest)');
    hold off;

    fprintf('Custom corners found: %d\n', numel(rows));
    fprintf('Built-in points found: %d\n', ptsBuiltIn.Count);

catch ME
    warning("Built-in compare skipped (toolbox missing?): %s", ME.message);
end

%% ------------------------------------------
% Task 4: Modify parameters and analyse impact
% -------------------------------------------
% We will sweep:
% - k in {0.04, 0.06}
% - sigma in {1.0, 2.0}
% - threshold fraction in {0.005, 0.01, 0.02}
kList = [0.04, 0.06];
sigmaList = [1.0, 2.0];
tFracList = [0.005, 0.01, 0.02];

figure('Name','Task 4: Parameter Sweep (Custom Harris)');
tile = tiledlayout(numel(kList), numel(sigmaList)*numel(tFracList), ...
                   'Padding','compact','TileSpacing','compact');

idx = 1;
for kk = 1:numel(kList)
    for ss = 1:numel(sigmaList)
        % recompute tensor components for this sigma
        sig = sigmaList(ss);
        A2 = imgaussfilt(Ix.^2, sig);
        B2 = imgaussfilt(Iy.^2, sig);
        C2 = imgaussfilt(Ix.*Iy, sig);

        for tt = 1:numel(tFracList)
            tf = tFracList(tt);

            % recompute R
            kVal = kList(kk);
            R2 = (A2 .* B2 - C2.^2) - kVal * (A2 + B2).^2;

            % threshold + NMS
            thr = tf * max(R2(:));
            mask = (R2 > thr) & imregionalmax(R2);

            [r2, c2] = find(mask);

            nexttile(idx);
            imshow(I0); hold on;
            plot(c2, r2, 'r.', 'MarkerSize', 6);
            title(sprintf('k=%.2f,\\sigma=%.1f,t=%.3f\\nN=%d', kVal, sig, tf, numel(r2)), ...
                  'FontSize', 8);
            hold off;

            idx = idx + 1;
        end
    end
end
title(tile, 'Custom Harris: Effect of k, sigma, and threshold fraction');

%% Optional: quick notes printed to command window
disp('Parameter impact hints:');
disp('- Increasing threshold fraction -> fewer corners (stronger only).');
disp('- Increasing sigma -> more smoothing -> fewer/noiseless corners, but may miss fine detail.');
disp('- Increasing k generally penalises edges more strongly; corner selection may change.');