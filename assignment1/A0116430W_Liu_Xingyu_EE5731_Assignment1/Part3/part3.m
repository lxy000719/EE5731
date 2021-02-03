%% Part 3
clear;
clc;

% load images h1 and h2
h1 = imread('h1.jpg');
h2 = imread('h2.jpg');

% select 4 keypoints on both h1 and h2
kp = {0, 0};
imshow(h1, []);
kp(1) = {ginput(4)};
imshow(h2, []);
kp(2) = {ginput(4)};

%% from h1 to h2
A = zeros(8, 9);
for i = 1:4
    from_img = kp{1};
    to_img = kp{2};
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
H12 = reshape(h, [3,3])';

% transform h1 to h2
transform_matrix = projective2d(H12.');
figure(1)
imshow(imwarp(h1, transform_matrix), [])

%% from h2 to h1
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
H21 = reshape(h, [3,3])';

% transform h2 to h1
transform_matrix = projective2d(H21.');
figure(2)
imshow(imwarp(h2, transform_matrix), [])