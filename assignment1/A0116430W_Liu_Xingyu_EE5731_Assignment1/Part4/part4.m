%% Part 4
clc;
clear;

% load im01 and im02
im01 = imread('im01.jpg');
im02 = imread('im02.jpg');

%% select 4 keypoints on both im01 and im02
kp = {0, 0};
imshow(im01, []);
kp(1) = {ginput(4)};
imshow(im02, []);
kp(2) = {ginput(4)};

%% homography matrix
A = zeros(8, 9);
for i = 1:4
    from_img = kp{2};
    to_img = kp{1};
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

%% get the transformation matrix
transform_matrix(1) = projective2d(eye(3));
transform_matrix(2) = projective2d(H.');

% get the size of the final stitched image
image_size_matrix(1, :) = size(im01);
image_size_matrix(2, :) = size(im02);

% get the output limits for transformation 
[xOutputLimit(1, :), yOutputLimit(1, :)] = outputLimits(transform_matrix(1), [1, image_size_matrix(1, 2)], [1, image_size_matrix(1, 1)]);
[xOutputLimit(2, :), yOutputLimit(2, :)] = outputLimits(transform_matrix(2), [1, image_size_matrix(2, 2)], [1, image_size_matrix(2, 1)]);

max_img_size = [480, 640, 3];

%  get the new image width and height according to the output limits
xWorldLimits = [min([1; xOutputLimit(:)]) max([max_img_size(2); xOutputLimit(:)])];
yWorldLimits = [min([1; yOutputLimit(:)]) max([max_img_size(1); yOutputLimit(:)])];
new_img_hgt = round(max([max_img_size(1); yOutputLimit(:)]) - min([1; yOutputLimit(:)]));
new_img_wdh = round(max([max_img_size(2); xOutputLimit(:)]) - min([1; xOutputLimit(:)]));

% define the result view
R = imref2d([new_img_hgt, new_img_wdh], xWorldLimits, yWorldLimits);

% initialization final image
stitch_img = zeros([new_img_hgt, new_img_wdh, 3], 'like', im01);

% create blender for combining and overlay images
alphablend = vision.AlphaBlender;

% apply transformation to get the stitched image
B = imwarp(im01, transform_matrix(1), 'OutputView', R);
stitch_img = step(alphablend, stitch_img, B);
B = imwarp(im02, transform_matrix(2), 'OutputView', R);
stitch_img = step(alphablend, stitch_img, B);

% show the stitched image
imshow(stitch_img);
