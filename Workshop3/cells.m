%% Load images
clear; close; clc;

if ~exist("cells/", "dir")
    unzip("cells.zip");
end

image_list = dir('cells/images/*.png');
mask_list = dir('cells/masks/*.png');

X = {}; Y = {};

for i = 1:length(image_list)
    ifname = image_list(i).name;
    gfname = mask_list(i).name;
    X{i} = imread(fullfile('cells/images', ifname));
    Y{i} = imread(fullfile('cells/masks', gfname));
end

%% Task 2-5: Segmentation, clean-up, evaluation

numImages = length(X);      % number of images
dice_scores = zeros(1, numImages);  % store DSC for each image

for idx = 1:numImages
    
    % Load image and ground truth
    I = X{idx};
    Yt = logical(Y{idx}(:,:,1));  % ground truth mask
    
    %% Step 1: Segmentation
    grayImg = rgb2gray(I);
    invImg = imcomplement(grayImg); 
    level = graythresh(invImg);
    myown = imbinarize(invImg, level);
    
    %% Step 2: Remove small noise
    myown = bwareaopen(myown, 2500); % remove objects smaller than 50 pixels
    
    %% Step 3: Remove mostly-cut border objects
    BW = myown;
    CC = bwconncomp(BW);
    stats = regionprops(CC, 'Area', 'PixelIdxList');
    
    margin = 10;  % safe inner margin
    innerMask = false(size(BW));
    innerMask(1+margin:end-margin, 1+margin:end-margin) = true;
    
    BW_clean = false(size(BW));
    for k = 1:length(stats)
        objectPixels = stats(k).PixelIdxList;
        totalArea = stats(k).Area;
        innerArea = sum(innerMask(objectPixels));
        ratio = innerArea / totalArea;
        if ratio > 0.8  % keep objects mostly inside
            BW_clean(objectPixels) = true;
        end
    end
    
    myown = BW_clean;
    
    %% Step 4: Fill holes inside objects
    myown = imfill(myown, 'holes');
    
    % Optional: Visualize one image
    
    figure;
    imshowpair(myown, Yt, 'montage');
    title(['Image ', num2str(idx), ' - Segmentation vs Ground Truth']);
    figure;
    imshowpair(myown, Yt);
    title(['Image ', num2str(idx), ' - Segmentation vs Ground Truth']);
    
    
    %% Step 5: Compute Dice manually
    dice_scores(idx) = 2*sum(myown(:) & Yt(:)) / (sum(myown(:)) + sum(Yt(:)));
    
end



%% Step 6: Compute statistics
mean_dice = mean(dice_scores);
std_dice = std(dice_scores);

disp('Dice scores for each image:')
disp(dice_scores)

disp(['Mean Dice: ', num2str(mean_dice)])
disp(['Standard Deviation: ', num2str(std_dice)])
