%% Task 1: Gaussian Scale-Space Pyramid
% CMP9135 Computer Vision
% Loads a standard MATLAB demo image, constructs a Gaussian scale-space
% pyramid across multiple octaves, and visualises how image details
% progressively disappear as scale increases.

clear all; close all;

%% -----------------------------------------------------------------------
%  1. LOAD IMAGE
% -----------------------------------------------------------------------
% Using 'cameraman.tif' — a built-in MATLAB demo image so results are
% fully reproducible on any machine without extra files.

img = imread('cameraman.tif');         % Load (256x256 uint8 greyscale)
img = im2double(img);                  % Convert to [0,1] double for maths

fprintf('Image loaded: %d x %d pixels\n', size(img,1), size(img,2));

%% -----------------------------------------------------------------------
%  2. PYRAMID PARAMETERS
% -----------------------------------------------------------------------
num_octaves   = 4;   % Number of octaves (resolution halves each octave)
scales_per_octave = 4;   % Number of blurred images inside each octave

% Base sigma for the very first level of each octave.
% Doubling sigma each octave keeps the smoothing consistent across scales.
base_sigma = 1.0;

% Scale multiplier between consecutive levels within an octave.
% k = 2^(1/scales_per_octave) ensures the sigma doubles across the octave.
k = 2^(1 / scales_per_octave);

fprintf('Octaves: %d  |  Scales per octave: %d  |  k = %.4f\n', ...
        num_octaves, scales_per_octave, k);

%% -----------------------------------------------------------------------
%  3. BUILD THE PYRAMID
% -----------------------------------------------------------------------
% pyramid{o}{s} holds the blurred image at octave o, scale level s.
% Between octaves the image is downsampled by 2 (halving each dimension).

pyramid  = cell(num_octaves, 1);
sigma_at = zeros(num_octaves, scales_per_octave);   % log of all sigmas used

current_image = img;  % Start from the original full-resolution image

for o = 1 : num_octaves

    pyramid{o} = cell(scales_per_octave, 1);

    for s = 1 : scales_per_octave

        % Effective sigma relative to the ORIGINAL image resolution
        % (doubles every octave, multiplies by k within an octave)
        sigma = base_sigma * (2^(o-1)) * (k^(s-1));
        sigma_at(o, s) = sigma;

        % Apply Gaussian smoothing to the current-octave image.
        % imgaussfilt handles padding automatically.
        pyramid{o}{s} = imgaussfilt(current_image, base_sigma * (k^(s-1)));

        fprintf('  Octave %d | Scale %d | sigma_effective = %.3f\n', o, s, sigma);
    end

    % Downsample for the next octave:
    % Take the most-blurred level of this octave and halve resolution.
    if o < num_octaves
        current_image = imresize(pyramid{o}{scales_per_octave}, 0.5);
    end
end

%% -----------------------------------------------------------------------
%  4. VISUALISE — one figure per octave (tiled subplots)
% -----------------------------------------------------------------------

for o = 1 : num_octaves
    figure('Name', sprintf('Octave %d', o), ...
           'NumberTitle', 'off', ...
           'Color', [0.15 0.15 0.15]);

    for s = 1 : scales_per_octave
        subplot(1, scales_per_octave, s);
        imshow(pyramid{o}{s}, []);
        title(sprintf('\\sigma_{eff} = %.2f', sigma_at(o,s)), ...
              'Color', 'w', 'FontSize', 9);
        axis off;
    end

    sgtitle(sprintf('Octave %d  —  image size: %d × %d px', ...
            o, size(pyramid{o}{1},1), size(pyramid{o}{1},2)), ...
            'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');
end

%% -----------------------------------------------------------------------
%  5. VISUALISE — summary montage (one representative level per octave)
% -----------------------------------------------------------------------
% Pick the middle scale of each octave as the representative image,
% then resize them all to the same dimensions for an easy side-by-side.

ref_size = size(pyramid{1}{1});   % Use octave-1 size as reference

figure('Name', 'Scale-Space Pyramid Overview', ...
       'NumberTitle', 'off', ...
       'Color', [0.15 0.15 0.15]);

for o = 1 : num_octaves
    mid_scale = round(scales_per_octave / 2);
    rep_img   = pyramid{o}{mid_scale};

    % Upscale smaller octave images back to reference size for montage only
    rep_img_display = imresize(rep_img, ref_size);

    subplot(1, num_octaves, o);
    imshow(rep_img_display, []);
    title(sprintf('Octave %d\n\\sigma_{eff}=%.2f\n(%d×%d)', ...
          o, sigma_at(o, mid_scale), ...
          size(rep_img,1), size(rep_img,2)), ...
          'Color', 'w', 'FontSize', 9);
    axis off;
end

sgtitle('Scale-Space Pyramid — One Representative Level per Octave', ...
        'Color', 'w', 'FontSize', 12, 'FontWeight', 'bold');

%% -----------------------------------------------------------------------
%  6. VISUALISE — Difference of Gaussian (DoG) preview
%     Subtracting consecutive scale levels reveals "blob" structures and
%     is the foundation of SIFT interest-point detection (Task 2+).
% -----------------------------------------------------------------------

figure('Name', 'Difference of Gaussians (DoG) — Octave 1', ...
       'NumberTitle', 'off', ...
       'Color', [0.15 0.15 0.15]);

num_dogs = scales_per_octave - 1;

for s = 1 : num_dogs
    dog = pyramid{1}{s+1} - pyramid{1}{s};
    subplot(1, num_dogs, s);
    imshow(dog, []);
    title(sprintf('DoG: s%d - s%d\n(\\sigma_{eff} %.2f - %.2f)', ...
          s+1, s, sigma_at(1,s+1), sigma_at(1,s)), ...
          'Color', 'w', 'FontSize', 9);
    axis off;
end

sgtitle('Difference of Gaussians (DoG) — Octave 1  [preview for Task 2]', ...
        'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold');

%% -----------------------------------------------------------------------
%  7. PRINT SUMMARY TABLE
% -----------------------------------------------------------------------
fprintf('\n=== Scale-Space Pyramid Summary ===\n');
fprintf('%-8s %-8s %-15s %-20s\n', 'Octave', 'Scale', 'Sigma (eff)', 'Image Size');
fprintf('%s\n', repmat('-',1,55));
for o = 1 : num_octaves
    [h, w] = size(pyramid{o}{1});
    for s = 1 : scales_per_octave
        fprintf('%-8d %-8d %-15.4f %d x %d\n', o, s, sigma_at(o,s), h, w);
    end
    fprintf('%s\n', repmat('-',1,55));
end