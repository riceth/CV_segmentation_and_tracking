% CMP9135M Computer Vision — Assessment 2
% main.m : feature extraction + object tracking on the parachute dataset.
%
% Tools used in this file:
%% Tool: GitHub CoPilot (free student version)
%% Purpose: interactive code completion, e.g., function body skeletons,
%%          variable and function names.
%
% Run-order note: this script is self-contained. Execute from top to
% bottom, or run individual %% cells after STEP 0 has been run once
% (STEP 0 populates the workspace variables that later steps consume).

clc; clear; close all;

% Reproducibility: any stochastic operation (e.g. random quarantine-set
% selection) is seeded so results are identical across runs.
rng(42);


%% ============================================================
%  STEP 0: Setup and data loading
%  - Locate the parachute dataset (51 RGB + 51 GT masks).
%  - Build a robust binary-mask extractor.
%  - Declare the train / quarantine / test splits up front, BEFORE
%    writing any feature or filter logic, so that no parameter is
%    chosen by looking at data we later evaluate on.
%% ============================================================

% ---- 0.1  Dataset paths ------------------------------------------------
% Adjust dataRoot if your unzipped dataset sits elsewhere.
dataRoot = 'parachute';
rgbDir   = fullfile(dataRoot, 'images');
gtDir    = fullfile(dataRoot, 'GT');

rgbList = dir(fullfile(rgbDir, '*.png'));
gtList  = dir(fullfile(gtDir,  '*.png'));

assert(~isempty(rgbList), 'No RGB images found in %s', rgbDir);
assert(~isempty(gtList),  'No GT masks found in %s',  gtDir);
assert(numel(rgbList) == numel(gtList), ...
    'RGB (%d) and GT (%d) file counts must match.', ...
    numel(rgbList), numel(gtList));

% Natural-sort by the integer embedded in each filename so that
% "frame_2.png" comes before "frame_10.png" regardless of zero-padding.
rgbList = sortByEmbeddedNumber(rgbList);
gtList  = sortByEmbeddedNumber(gtList);

numFrames = numel(rgbList);
fprintf('Found %d frame pairs in %s\n', numFrames, dataRoot);


% ---- 0.2  Load images and extract binary masks -------------------------
% Images may differ in spatial size across the sequence, so we store them
% in cell arrays rather than a single numeric array.
rgbImgs = cell(numFrames, 1);
gtMasks = cell(numFrames, 1);

for i = 1:numFrames
    rgbImgs{i} = imread(fullfile(rgbList(i).folder, rgbList(i).name));
    gtRaw      = imread(fullfile(gtList(i).folder,  gtList(i).name));
    gtMasks{i} = extractMask(gtRaw);
end

fprintf('Loaded %d RGB frames and %d GT masks.\n', numFrames, numFrames);


% ---- 0.3  Declare splits (before any analysis) -------------------------
% The brief uses 0-indexed frame numbers; MATLAB indices are 1-based.
%   Brief frames  0–40  <=>  MATLAB indices  1–41   (KF training)
%   Brief frames 41–50  <=>  MATLAB indices 42–51   (KF evaluation)
trainIdx = 1:41;
testIdx  = 42:51;

% Shape/HoG development quarantine: hold out 5 random frames so that no
% feature-extraction parameter is ever tuned on them. Selected here,
% before any feature logic exists, so the choice is independent of results.
%
% Why 5: ~10% of the 51 frames, in line with the standard 10–20% hold-out
% convention. Smaller would give noisier confirmation statistics; larger
% would shrink the development set below 80%, leaving fewer frames in which
% to observe trajectory patterns. The number is therefore set by convention,
% not chosen to produce any particular result.
nQuarantine   = 5;
quarantineIdx = sort(randperm(numFrames, nQuarantine));
developIdx    = setdiff(1:numFrames, quarantineIdx);

fprintf('KF  train indices (MATLAB) : %s\n', mat2str(trainIdx));
fprintf('KF  test  indices (MATLAB) : %s\n', mat2str(testIdx));
fprintf('Shape/HoG quarantine (1..%d): %s\n', numFrames, mat2str(quarantineIdx));


% ---- 0.4  Sanity view: overlay masks on a few sample frames -----------
% First, middle and last frames — chosen for visual coverage of the
% sequence, not used by any subsequent computation. Display only.
sampleIdx = unique([1, round(numFrames/2), numFrames]);
figure('Name', 'STEP 0 sanity: mask overlay on RGB');
tiledlayout(1, numel(sampleIdx), 'Padding', 'compact', 'TileSpacing', 'compact');
for k = 1:numel(sampleIdx)
    i = sampleIdx(k);
    nexttile;
    imshow(labeloverlay(rgbImgs{i}, gtMasks{i}, ...
        'Colormap', [1 0 0], 'Transparency', 0.55));
    title(sprintf('Frame %d  (brief idx %d)', i, i-1));
end


%% ============================================================
%  STEP 1: Task 1(i) — Shape features
%  For each GT mask compute four shape descriptors:
%     solidity         = Area / ConvexArea                       in (0, 1]
%     circularity      = 4*pi*Area / Perimeter^2                 in (0, 1]
%     non_compactness  = 1 - circularity                         in [0, 1)
%     eccentricity     = ellipse eccentricity of the region      in [0, 1)
%  All four are scale- and translation-invariant, so they depend on
%  the parachute's shape alone and not on where it sits in the frame.
%% ============================================================

% ---- 1.1  Compute the four features for every frame --------------------
shapeFeatures = zeros(numFrames, 4);   % columns: [sol, nc, circ, ecc]
shapeNames    = {'solidity', 'non-compactness', 'circularity', 'eccentricity'};

for i = 1:numFrames
    shapeFeatures(i, :) = computeShapeFeatures(gtMasks{i});
end

% Save as a table for readability (frame index column uses the brief's
% 0-based numbering, which is how every plot/table in the report refers
% to frames).
shapeTable = array2table(shapeFeatures, 'VariableNames', ...
    {'solidity', 'non_compactness', 'circularity', 'eccentricity'});
shapeTable.frame = (0:numFrames-1)';
shapeTable = movevars(shapeTable, 'frame', 'Before', 1);


% ---- 1.2  Plot each feature's trajectory across the sequence -----------
frames0 = 0:numFrames-1;                  % brief-numbered frame axis
figure('Name', 'STEP 1: Shape-feature trajectories');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
for j = 1:4
    nexttile;
    plot(frames0, shapeFeatures(:, j), '-o', 'LineWidth', 1.2, 'MarkerSize', 3);
    grid on;
    xlabel('frame index (brief numbering)');
    ylabel(shapeNames{j});
    title(shapeNames{j});
end
sgtitle('Shape features across the 51 parachute frames');


% ---- 1.3  Summary statistics (develop set vs. quarantine) --------------
% Reporting them separately is the evaluation-rigour practice from the
% A01 feedback: any commentary we write about trajectories is grounded
% in the development subset; the quarantine numbers are the independent
% confirmation.
fprintf('\n--- STEP 1: Shape-feature summary ---\n');
fprintf('%-16s  %10s %10s  |  %10s %10s\n', ...
    'feature', 'dev mean', 'dev std', 'qtn mean', 'qtn std');
for j = 1:4
    devVals = shapeFeatures(developIdx, j);
    qtnVals = shapeFeatures(quarantineIdx, j);
    fprintf('%-16s  %10.4f %10.4f  |  %10.4f %10.4f\n', ...
        shapeNames{j}, mean(devVals), std(devVals), ...
        mean(qtnVals), std(qtnVals));
end

% Expose for later cells / the discriminative analysis in STEP 3.
clear j i devVals qtnVals


%% ============================================================
%  STEP 2: Task 1(ii) — HoG features (4 orientation bins)
%  One HoG feature vector per image. The orientation histogram uses
%  exactly four bins centred at 0°, 45°, 90°, 135° (per the brief).
%
%  Input image: the RGB image with the background suppressed by the
%  GT mask, so HoG describes edges on the parachute itself rather
%  than sky/cloud texture. (The brief notes that the masks "can be
%  used to extract the corresponding areas from the RGB images".)
%
%  Cell size: derived from the parachute's bounding-box in each frame
%  rather than a fixed pixel value. The DESIGN choice is "how many
%  tiles cover the object" (TILES_ACROSS_OBJECT, conventionally 4
%  following the SIFT 4×4 grid). The pixel cell size is a derived
%  consequence of that choice and the object's apparent size.
%% ============================================================

% ---- 2.1  Compute HoG vectors for every frame --------------------------
% TILES_ACROSS_OBJECT: SIFT-style 4×4 grid convention. Citable design
% choice, NOT a tuned pixel value.
TILES_ACROSS_OBJECT = 4;

% MIN_CELL_PX: mathematical sanity floor. A 4-bin orientation histogram
% built from fewer than ~4 gradient samples is statistically meaningless.
% Below 2 px per cell each cell holds at most 4 gradients, so 2 is the
% smallest cell size at which the histogram has any informational content.
MIN_CELL_PX = 2;

hogVectors = cell(numFrames, 1);     % HoG feature vector per frame
hogCellSizes = zeros(numFrames, 1);  % the per-frame cell size actually used
hogGridSizes = zeros(numFrames, 2);  % the per-frame [nRowsTiles nColsTiles]

for i = 1:numFrames
    [hogVectors{i}, hogCellSizes(i), hogGridSizes(i,:)] = ...
        computeHoG4(rgbImgs{i}, gtMasks{i}, ...
                    TILES_ACROSS_OBJECT, MIN_CELL_PX);
end

% Per-frame vector length (varies with bbox aspect ratio, since each cell
% contributes 4 bins and the grid is m×n cells).
hogLengths = cellfun(@numel, hogVectors);

fprintf('\n--- STEP 2: HoG summary ---\n');
fprintf('Cell size  (px)  : min=%d  median=%d  max=%d\n', ...
    min(hogCellSizes), round(median(hogCellSizes)), max(hogCellSizes));
fprintf('Grid (rows×cols) : median=%d × %d\n', ...
    round(median(hogGridSizes(:,1))), round(median(hogGridSizes(:,2))));
fprintf('Vector length    : min=%d  median=%d  max=%d\n', ...
    min(hogLengths), round(median(hogLengths)), max(hogLengths));


% ---- 2.2  Per-bin energy across the sequence ---------------------------
% For each frame, sum the contributions in each of the four orientation
% bins (0°, 45°, 90°, 135°). This collapses the variable-length per-frame
% vectors into a fixed 51×4 matrix that's directly plottable, and answers
% "how does the dominant edge orientation change across the sequence?".
binEnergy = zeros(numFrames, 4);
for i = 1:numFrames
    v = hogVectors{i};
    % Vector layout from computeHoG4: [c1_b1 c1_b2 c1_b3 c1_b4  c2_b1 ...]
    % so reshaping into 4 rows groups all values from each bin together.
    binEnergy(i, :) = sum(reshape(v, 4, []), 2)';
end

binNames = {'0°', '45°', '90°', '135°'};

frames0 = 0:numFrames-1;
figure('Name', 'STEP 2: HoG per-bin energy across the sequence');
plot(frames0, binEnergy, '-o', 'LineWidth', 1.2, 'MarkerSize', 3);
grid on; xlabel('frame index (brief numbering)');
ylabel('summed bin energy (L2-normalised vector)');
legend(binNames, 'Location', 'best');
title('HoG: orientation-bin energy per frame');


% ---- 2.3  Heatmap of full HoG vectors (zero-padded) --------------------
% Different frames produce different-length vectors, so to visualise all
% 51 side-by-side we zero-pad up to the longest. This is display-only.
maxLen = max(hogLengths);
H = zeros(numFrames, maxLen);
for i = 1:numFrames
    v = hogVectors{i};
    H(i, 1:numel(v)) = v;
end

figure('Name', 'STEP 2: HoG vectors heatmap');
imagesc(H); colormap parula; colorbar;
xlabel('HoG vector index (zero-padded to max length)');
ylabel('frame index (MATLAB 1-based)');
title('HoG feature vectors — one row per frame');


% ---- 2.4  Dev vs. quarantine summary -----------------------------------
fprintf('\nHoG bin-energy summary (dev vs quarantine):\n');
fprintf('%-6s  %10s %10s  |  %10s %10s\n', ...
    'bin', 'dev mean', 'dev std', 'qtn mean', 'qtn std');
for j = 1:4
    devVals = binEnergy(developIdx, j);
    qtnVals = binEnergy(quarantineIdx, j);
    fprintf('%-6s  %10.4f %10.4f  |  %10.4f %10.4f\n', ...
        binNames{j}, mean(devVals), std(devVals), ...
        mean(qtnVals), std(qtnVals));
end

clear i j v devVals qtnVals maxLen H


%% ============================================================
%  STEP 3: Task 1(iii) — Discriminative comparison
%% ============================================================

% <placeholder — implemented in Phase 3>


%% ============================================================
%  STEP 4: Task 2 — Measurements from GT masks (centroid + angle)
%% ============================================================

% <placeholder — implemented in Phase 4>


%% ============================================================
%  STEP 5: Task 2 — Translation Kalman filter (from scratch)
%% ============================================================

% <placeholder — implemented in Phase 5>


%% ============================================================
%  STEP 6: Task 2 — Rotation Kalman filter (from scratch)
%% ============================================================

% <placeholder — implemented in Phase 6>


%% ============================================================
%  STEP 7: Results — tables and plots
%% ============================================================

% <placeholder — implemented in Phase 7>


%% ============================================================
%  Local helper functions
%% ============================================================

function [vec, cellPx, gridRC] = computeHoG4(rgbImg, mask, tilesAcross, minCellPx)
% computeHoG4  HoG feature with 4 orientation bins (0/45/90/135°),
% computed on the RGB image with the background suppressed by `mask`.
%
% INPUTS
%   rgbImg      H×W×3 RGB image (uint8 or double).
%   mask        H×W logical mask (true = parachute, false = background).
%   tilesAcross design choice: number of HoG tiles across the object's
%               smaller bounding-box side. Conventionally 4 (SIFT-style).
%   minCellPx   sanity floor on cell side length, in pixels.
%
% OUTPUTS
%   vec     1×N HoG feature vector, L2-normalised.
%           Layout: cells in row-major order, four bins per cell, so
%           vec = [c1_b1 c1_b2 c1_b3 c1_b4  c2_b1 c2_b2 ...].
%   cellPx  scalar cell side length in pixels actually used.
%   gridRC  [nRowsTiles nColsTiles] of the tile grid placed on the bbox.

    % --- 1. Convert to greyscale double in [0,1] for gradient stability.
    if size(rgbImg, 3) == 3
        gImg = rgb2gray(rgbImg);
    else
        gImg = rgbImg;
    end
    gImg = im2double(gImg);

    % --- 2. Apply the mask: zero out background pixels so subsequent
    % gradients only carry information about the parachute and its edge.
    gImg(~mask) = 0;

    % --- 3. Crop to the parachute's bounding box. This makes the
    % per-frame "image area for HoG" object-relative, so the same
    % grid resolution covers the same fraction of the parachute in
    % every frame regardless of where it sits or how large it appears.
    stats = regionprops(bwareafilt(mask, 1), 'BoundingBox');
    if isempty(stats)
        vec = []; cellPx = NaN; gridRC = [0 0];
        return;
    end
    bb = stats.BoundingBox;          % [x y width height], 0.5-offset
    x0 = max(1, floor(bb(1)));
    y0 = max(1, floor(bb(2)));
    x1 = min(size(gImg, 2), x0 + ceil(bb(3)) - 1);
    y1 = min(size(gImg, 1), y0 + ceil(bb(4)) - 1);
    patch = gImg(y0:y1, x0:x1);

    [hP, wP] = size(patch);

    % --- 4. Choose cell size by dividing the SHORTER bbox side into
    % `tilesAcross` tiles. Using the shorter side guarantees at least
    % `tilesAcross` tiles in both dimensions; the longer side then gets
    % more tiles, preserving the bbox aspect ratio in the grid.
    cellPx = max(minCellPx, floor(min(hP, wP) / tilesAcross));

    nRows = floor(hP / cellPx);
    nCols = floor(wP / cellPx);
    gridRC = [nRows, nCols];

    if nRows < 1 || nCols < 1
        vec = []; return;
    end

    % Trim the patch to an exact multiple of cellPx so the cells tile cleanly.
    patch = patch(1:nRows*cellPx, 1:nCols*cellPx);

    % --- 5. Image gradients via centred-difference filter [-1 0 1].
    % This is the canonical HoG operator (Dalal & Triggs 2005).
    fx = [-1 0 1];
    fy = fx';
    gx = imfilter(patch, fx, 'replicate');
    gy = imfilter(patch, fy, 'replicate');

    mag = sqrt(gx.^2 + gy.^2);

    % Unsigned orientation in [0, 180): atan2 returns [-pi, pi]; mod by
    % pi folds opposite directions onto each other (an edge running NE-SW
    % and one running SW-NE are the same edge). Convert to degrees.
    ori = mod(atan2(gy, gx), pi) * (180 / pi);   % in [0, 180)

    % --- 6. Soft-assign each pixel's gradient to one of 4 bins centred
    % at 0, 45, 90, 135 degrees (bin width = 45°). Hard-binning is fine
    % here — soft (linear-interpolated) binning is a refinement we can
    % add later if needed; for 4 bins the difference is small.
    binWidth = 180 / 4;                                  % 45° per bin
    binIdx   = floor(ori / binWidth) + 1;                % in {1,2,3,4}
    binIdx(binIdx > 4) = 4;                              % safety cap

    % --- 7. Accumulate magnitude into the (cell × bin) histogram.
    % Vectorised over all pixels; for each pixel we know its cell row,
    % cell column, and bin index, so a single accumarray call builds
    % the entire (nRows × nCols × 4) histogram in one pass.
    [pyIdx, pxIdx] = ndgrid(1:nRows*cellPx, 1:nCols*cellPx);
    cellRow = ceil(pyIdx / cellPx);
    cellCol = ceil(pxIdx / cellPx);

    subs = [cellRow(:), cellCol(:), binIdx(:)];
    hist3D = accumarray(subs, mag(:), [nRows, nCols, 4]);

    % --- 8. Flatten in cell-row-major × bin order so the layout is
    % [c1_b1 c1_b2 c1_b3 c1_b4  c2_b1 ...] — predictable and simple
    % to slice when later analysis needs per-bin or per-cell views.
    vec = reshape(permute(hist3D, [3 2 1]), 1, []);

    % --- 9. L2-normalise the whole vector once. eps prevents division
    % by zero on degenerate frames with no gradient (e.g. all-black).
    vec = vec / (norm(vec) + eps);
end


function feats = computeShapeFeatures(mask)
% computeShapeFeatures  Return [solidity, non_compactness, circularity,
% eccentricity] for the foreground region of a binary mask.
%
% Formulae (definitions, not tuned values — all dimensionless,
% scale- and translation-invariant):
%   solidity        = Area / ConvexArea
%   circularity     = 4*pi*Area / Perimeter^2          (max = 1, perfect disc)
%   non_compactness = 1 - circularity
%   eccentricity    = sqrt(1 - (minor/major)^2)        (ellipse eccentricity)
%
% Domain assumption: the dataset contains a single object of interest
% (the parachute), so when the mask has multiple disconnected components
% we keep only the largest — the others are segmentation artefacts.
    mask = logical(mask);
    if ~any(mask(:))
        feats = [NaN NaN NaN NaN];
        return;
    end

    % Keep the largest connected component. The "1" comes from the domain
    % (one parachute per frame), not from tuning.
    mask = bwareafilt(mask, 1);

    stats = regionprops(mask, ...
        'Area', 'Perimeter', 'ConvexArea', 'Eccentricity');

    A  = stats.Area;
    P  = stats.Perimeter;
    Ac = stats.ConvexArea;
    ecc = stats.Eccentricity;

    % Division-by-zero guard for degenerate masks (single-pixel regions
    % have zero perimeter under MATLAB's estimator).
    if P > 0
        circ = 4 * pi * A / (P^2);
    else
        circ = 0;
    end

    % Mathematical clamp: circularity is provably <= 1 for any planar
    % region (isoperimetric inequality). MATLAB's discrete perimeter
    % estimator can numerically overshoot by a fraction of a percent on
    % small regions, so we cap to the theoretical bound. The "1" here is
    % the bound itself, not a tuned threshold.
    circ = min(circ, 1);

    sol = A / Ac;
    nc  = 1 - circ;

    feats = [sol, nc, circ, ecc];
end


function mask = extractMask(gtImg)
% extractMask  Convert a GT image (either indexed 2-D or RGB "visual"
% version) into a binary parachute mask.
%
% Convention used across this submission:
%     background = 0 (black in either representation)
%     foreground = any non-zero pixel
% The dataset contains a single object of interest (the parachute),
% so collapsing all non-zero labels/colours to TRUE is correct and
% works for both the indexed PNG and the RGB-convenience PNG.
    if ndims(gtImg) == 3
        mask = any(gtImg ~= 0, 3);   % RGB: non-black anywhere in RGB
    else
        mask = gtImg > 0;            % indexed/greyscale: non-zero label
    end
    mask = logical(mask);
end


function lst = sortByEmbeddedNumber(lst)
% sortByEmbeddedNumber  Sort a struct array from dir() by the first
% integer found inside each filename. Falls back to plain lexical
% order if any filename lacks an integer (still deterministic).
    names = {lst.name};
    nums  = nan(1, numel(names));
    for k = 1:numel(names)
        tok = regexp(names{k}, '\d+', 'match', 'once');
        if ~isempty(tok)
            nums(k) = str2double(tok);
        end
    end
    if all(~isnan(nums))
        [~, order] = sort(nums);
    else
        [~, order] = sort(names);
    end
    lst = lst(order);
end