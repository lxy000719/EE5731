%% part 6
clc;
clear;

% load images
im01 = imread('im01.jpg');
im02 = imread('im02.jpg');
im03 = imread('im03.jpg');
im04 = imread('im04.jpg');
im05 = imread('im05.jpg');

% integrate all images into one cell array
total_img = {im01, im02, im03, im04, im05};
for i = 1:length(total_img)
    img_size(i, :) = size(total_img{i});
end
img_size = img_size(1:5, 1:2);

%% get transform matrix
transform_matrix(1) = projective2d(eye(3));

% 1-2, 2-3, 3-4, 4-5
for i = 2:length(total_img)
    transform_matrix(i) = tf_matrix(total_img{i-1}, total_img{i});
    transform_matrix(i).T = transform_matrix(i).T * transform_matrix(i-1).T;
end

%% get the initial output limits for each transform
for i = 1:length(transform_matrix)
    [xOutputLimit(i, :), yOutputLimit(i, :)] = outputLimits(transform_matrix(i), [1, img_size(i, 2)], [1, img_size(i, 1)]);
end

% find the centre image and index
[B, I] = sort(mean(xOutputLimit, 2));
center = I(floor((numel(transform_matrix) + 1) / 2));

% transform all H matrix based on certer image
invert_center_tformsmatrix = invert(transform_matrix(center));
for i = 1:length(transform_matrix)
    transform_matrix(i).T = transform_matrix(i).T * invert_center_tformsmatrix.T;
end

%% get the size of the stitched image and the final stitched image

% get the final output limits for each transform
for i = 1:length(transform_matrix)
    [xOutputLimit(i,:), yOutputLimit(i,:)] = outputLimits(transform_matrix(i), [1 img_size(i, 2)], [1 img_size(i, 1)]);    
end

max_img_size = max(img_size);

% get the final image weight and height
xWorldLimits = [min([1; xOutputLimit(:)]) max([max_img_size(2); xOutputLimit(:)])];
yWorldLimits = [min([1; yOutputLimit(:)]) max([max_img_size(1); yOutputLimit(:)])];
new_img_hgt = round(max([max_img_size(1); yOutputLimit(:)]) - min([1; yOutputLimit(:)]));
new_img_wdh  = round(max([max_img_size(2); xOutputLimit(:)]) - min([1; xOutputLimit(:)]));

% define the result view
R = imref2d([new_img_hgt, new_img_wdh], xWorldLimits, yWorldLimits);

% initialization final image
stitch_img = zeros([new_img_hgt, new_img_wdh, 3], 'like', im01);

% create blender for combining and overlay images
alphablend = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');

% apply transformation to get the stitched image
for i = 1:length(total_img)
    B = imwarp(total_img{i}, transform_matrix(i), 'OutputView', R);
    mask = imwarp(true(size(total_img{i},1),size(total_img{i},2)), transform_matrix(i), 'OutputView', R);
    stitch_img = step(alphablend, stitch_img, B, mask);
end

% show the stitched image
figure
imshow(stitch_img);

%% functions

function transform = tf_matrix(im01, im02)

% get keypoints for im01 and im02
keypoints1 = SIFT(double(rgb2gray(im01)),3,5,1.3);
keypoints2 = SIFT(double(rgb2gray(im02)),3,5,1.3);

%% find all matches
match_kp1 = keypoints1;
match_kp2 = cell(length(keypoints1));


for i = 1:length(keypoints1)
    min_dist = sqrt(sum((keypoints1{i}.Descriptor - keypoints2{1}.Descriptor).^2));
    match = 1;
    for j = 2:length(keypoints2)
        distance = sqrt(sum((keypoints1{i}.Descriptor - keypoints2{j}.Descriptor).^2));
        if distance < min_dist
            min_dist = distance;
            match = j;
        end
    end
    match_kp2{i} = keypoints2{match};
end

% get the coordinates
coord1 = zeros(length(match_kp1), 2);
coord2 = zeros(length(match_kp2), 2);
for i = 1:length(match_kp1)
    coord1(i, 1) = match_kp1{i}.Coordinates(2);
    coord1(i, 2) = match_kp1{i}.Coordinates(1);
    coord2(i, 1) = match_kp2{i}.Coordinates(2);
    coord2(i, 2) = match_kp2{i}.Coordinates(1);
end


%% use RANSAC to find the best homograpy matrix and matches
number_points = length(match_kp1);
index_total = [1, number_points];
threshold = 5;
cur_error = 1.e1000;
max_inliers = 0;
best_inliers = [];
iter = 0;

while iter < 1000000 && (max_inliers / number_points) < 0.2
    iter = iter + 1;
    num_randpoint = 5;
    % randomly select 5 keypoints
    index = randi(index_total, 1, num_randpoint);
    rand_keypoint1 = match_kp1(index);
    rand_keypoint2 = match_kp2(index);
    rand_keypoint1_coord = zeros(num_randpoint,2);
    rand_keypoint2_coord = zeros(num_randpoint,2);
    
    for i = 1:num_randpoint
        rand_keypoint1_coord(i,:) = fliplr(rand_keypoint1{i}.Coordinates);
        rand_keypoint2_coord(i,:) = fliplr(rand_keypoint2{i}.Coordinates);
    end
    
% get homography matrix H
    A = zeros(8, 9);
    for i = 1:num_randpoint
        from_img = rand_keypoint2_coord;
        to_img = rand_keypoint1_coord;
        x = from_img(i, 1);
        y = from_img(i, 2);
        x_prime = to_img(i, 1);
        y_prime = to_img(i, 2);
        A1 = [x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime, -x_prime];
        A2 = [0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime, -y_prime];
        A(i * 2 - 1, :) = A1;
        A(i * 2, :) = A2;
    end
    % decompose A using SVD
    [U,S,V] = svd(A);
    VT = V';
    % h is the last row of VT
    h = VT(end, :);
    H = reshape(h, [3,3])';
    
    % get the inliers 
    [num_inliers, inliers1, error1] = inliers(rand_keypoint1_coord, rand_keypoint2_coord, threshold, H);
    % check if all 5 points are inliers
    if num_inliers == num_randpoint
        mat_coord1 = zeros(length(match_kp1), 2);
        mat_coord2 = zeros(length(match_kp2), 2);
        for i = 1:length(match_kp1)
            mat_coord1(i, 1) = match_kp1{i}.Coordinates(2);
            mat_coord1(i, 2) = match_kp1{i}.Coordinates(1);
            mat_coord2(i, 1) = match_kp2{i}.Coordinates(2);
            mat_coord2(i, 2) = match_kp2{i}.Coordinates(1);
        end
        [num_inliers, inliers2, error2] = inliers(mat_coord1, mat_coord2, threshold, H);
        if (num_inliers >= max_inliers)
            if (num_inliers > max_inliers || error2 < cur_error)
                max_inliers = num_inliers;
                best_inliers = inliers2;
                cur_error = error2;
            end
        end
    end
end

% get the matching graph
best_match1 = zeros(length(best_inliers), 2);
best_match2 = zeros(length(best_inliers), 2);
for i = 1: length(best_inliers)
    best_match1(i,:) = fliplr(match_kp1{best_inliers(i)}.Coordinates);
    best_match2(i,:) = fliplr(match_kp2{best_inliers(i)}.Coordinates);
end

figure; ax = axes;
showMatchedFeatures(im01, im02, best_match1, best_match2,'montage','Parent', ax);
title(ax, 'Candidate point matches');
legend(ax, 'Matched points 1','Matched points 2');

%% get the best homography matrix and stitch im01 and im02
A = zeros(8, 9);
for i = 1:length(best_match1)
    from_img = best_match2;
    to_img = best_match1;
    x = from_img(i, 1);
    y = from_img(i, 2);
    x_prime = to_img(i, 1);
    y_prime = to_img(i, 2);
    A1 = [x, y, 1, 0, 0, 0, -x*x_prime, -y*x_prime, -x_prime];
    A2 = [0, 0, 0, x, y, 1, -x*y_prime, -y*y_prime, -y_prime];
    A(i * 2 - 1, :) = A1;
    A(i * 2, :) = A2;
end

% decompose A using SVD
[U,S,V] = svd(A);
VT = V';

% h is the last row of VT
best_h = VT(end, :);
best_H = reshape(best_h, [3,3])';
transform = projective2d(best_H.');

end

function [num_inliers, inliers, error] = inliers(rand_keypoint1_coord, rand_keypoint2_coord, threshold, H)
    
    transform_points = zeros(length(rand_keypoint2_coord), 2);
    for i = 1 : length(rand_keypoint2_coord)
        homogeneous_coord = [rand_keypoint2_coord(i,:) 1]';
        res = H * homogeneous_coord;
        res = res / res(3, 1);
        transform_points(i,:) = res(1:2,1)';
    end
    
    inliers = [];    
    num_inliers = 0;
    error = 0;
    for i = 1:length(rand_keypoint1_coord)
        dist = sqrt(sum((transform_points(i,:) - rand_keypoint1_coord(i,:)).^2));
        if dist < threshold
            num_inliers = num_inliers + 1;
            inliers(num_inliers) = i;
            error = error + dist;
        end
    end
end