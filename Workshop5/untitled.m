%% Task 1: Gaussian Scale-Space Pyramid

clear all; close all;

% Load image
img = im2double(imread('cameraman.tif'));

% Parameters
num_octaves       = 4;
scales_per_octave = 5;
base_sigma        = 1.6;
k                 = 2^(1/scales_per_octave);

% Build pyramid
pyramid  = cell(num_octaves, 1);
sigma_at = zeros(num_octaves, scales_per_octave);
current  = img;

for o = 1:num_octaves
    pyramid{o} = cell(scales_per_octave, 1);
    for s = 1:scales_per_octave
        sigma_at(o,s)  = base_sigma * (k^(s-1));
        pyramid{o}{s}  = imgaussfilt(current, sigma_at(o,s));
    end
    current = imresize(pyramid{o}{end}, 0.5);
end

% Display one row per octave
figure;
for o = 1:num_octaves
    for s = 1:scales_per_octave
        subplot(num_octaves, scales_per_octave, (o-1)*scales_per_octave + s);
        imshow(pyramid{o}{s}, []);
        title(sprintf('O%d S%d\n\\sigma=%.2f', o, s, sigma_at(o,s)), 'FontSize', 7);
    end
end
sgtitle('Gaussian Scale-Space Pyramid');

%% Task 2: DoG and Extrema Detection
% Run task1 first to get pyramid and sigma_at in the workspace

% Build DoG pyramid
num_dogs = scales_per_octave - 1;
dog      = cell(num_octaves, 1);

for o = 1:num_octaves
    dog{o} = cell(num_dogs, 1);
    for d = 1:num_dogs
        dog{o}{d} = pyramid{o}{d+1} - pyramid{o}{d};
    end
end

% Display DoG images for octave 1
figure;
for d = 1:num_dogs
    subplot(1, num_dogs, d);
    imshow(dog{1}{d}, []);
    title(sprintf('DoG %d', d), 'FontSize', 8);
end
sgtitle('DoG Images - Octave 1');

% Detect extrema
threshold = 0.005;
keypoints = [];

for o = 1:num_octaves
    [rows, cols] = size(dog{o}{1});
    for d = 2:num_dogs-1
        for r = 2:rows-1
            for c = 2:cols-1
                val = dog{o}{d}(r,c);
                if abs(val) < threshold, continue; end

                % 26-neighbour cube
                prev = dog{o}{d-1}(r-1:r+1, c-1:c+1);
                curr = dog{o}{d  }(r-1:r+1, c-1:c+1);
                next = dog{o}{d+1}(r-1:r+1, c-1:c+1);
                nb   = curr(:); nb(5) = [];
                nb   = [prev(:); nb; next(:)];

                if val > max(nb) || val < min(nb)
                    kp.x     = c;
                    kp.y     = r;
                    kp.oct   = o;
                    kp.sigma = sigma_at(o,d);
                    keypoints = [keypoints, kp]; %#ok<AGROW>
                end
            end
        end
    end
end

fprintf('Keypoints found: %d\n', numel(keypoints));

%% Task 3: Visualise Keypoints
% Run task1.m and task2.m first

figure;
imshow(img, []); hold on;
title(sprintf('DoG Keypoints (%d detected)', numel(keypoints)));

theta = linspace(0, 2*pi, 50);

for i = 1:numel(keypoints)
    kp    = keypoints(i);         
    scale = 2^(kp.oct - 1);       % map back to original resolution
    cx    = kp.x * scale;
    cy    = kp.y * scale;
    r     = kp.sigma * scale * 3; % circle radius proportional to scale

    plot(cx + r*cos(theta), cy + r*sin(theta), 'g-', 'LineWidth', 0.8);
    plot(cx, cy, 'g+', 'MarkerSize', 3);
end

hold off;

% A small circle means a fine detail detected at low sigma.
% A large circle means a coarse structure detected only at high sigma.
% This is the information needed to match features between two photos taken at different distances.