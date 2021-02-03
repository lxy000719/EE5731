%% Part 2
clc;
clear;

% load images
im2 = double(imread('im2.png'));
im6 = double(imread('im6.png'));

% get the image size
[H2 W2 D2] = size(im2);
[H6 W6 D6] = size(im6);

% reshape images
im2 = flip(flip(im2, 1), 2);
im2 = reshape(im2, 1, [], 3);
im6 = flip(flip(im6, 1), 2);
im6 = reshape(im6, 1, [], 3);

% initialization
% set disparity range
disparity = (1:64);
num_pixel = W2 * H2;
segclass = disparity(1) * ones(1, num_pixel);
threshold = 0.02 * (disparity(length(disparity)) - disparity(1));
Labelcost = pdist2(disparity', disparity');
Labelcost(Labelcost > threshold) = threshold;
unary = [];

for k = 1:length(disparity)
    
    t = disparity(k);
    t = H2 * t;
    i6 = im6;
    i6(:, 1:t, :) = [];
    i2 = im2;
    i2(:, (num_pixel - t + 1):num_pixel, :) = [];
    A = sqrt(sum((i6 - i2).^2, 3));
    A = [A, 255 * 3 * ones(1, t)];
    unary = [unary;A];
    
end

% get the sparse matrix
r = im2(:, :, 1);
g = im2(:, :, 2);
b = im2(:, :, 3);

% make use of open source function to find connected edges between pixels
E = edges4connected(H2,W2);

% get the distance of the edges between connected pixels
r_dist = sqrt((r(E(:, 1))-r(E(:, 2))).^2);
g_dist = sqrt((g(E(:, 1))-g(E(:, 2))).^2);
b_dist = sqrt((b(E(:, 1))-b(E(:, 2))).^2);
dist = r_dist + g_dist + b_dist;

pairwise = sparse(E(:, 1),E(:, 2),double(dist), num_pixel, num_pixel, 4*num_pixel);
unary = unary./max(unary(:));

%% find the depth map

addpath('../GCMex')

% create a list of different lambda values
lambda_lst = [0.0003, 0.0006, 0.0009, 0.003, 0.006, 0.009, 0.03, 0.06, 0.09];

for lambda = lambda_lst
    
content = sprintf('Depth map for lambda=%f', lambda);
disp(content);

[DispClass, Einit, Eafter] = GCMex(segclass, single(unary), lambda*pairwise, single(Labelcost));
depth_map = disparity(DispClass+1);

% reshape to display result depth map
result = mat2gray(reshape(depth_map, H2, W2));
result = flip(flip(result ,1), 2);

% display and store the depth map with different lambda value
figure;
imshow(result)
name = sprintf('lambda=%f.jpg', lambda);
imwrite(result, name);

end

%% functions

function E = edges4connected(height,width)

% EDGE4CONNECTED Creates edges where each node
%   is connected to its four adjacent neighbors on a 
%   height x width grid.
%   E - a vector in which each row i represents an edge
%   E(i,1) --> E(i,2). The edges are listed is in the following 
%   neighbor order: down,up,right,left, where nodes 
%   indices are taken column-major.
%
%   (c) 2008 Michael Rubinstein, WDI R&D and IDC
%   $Revision$
%   $Date$
%

N = height*width;
I = []; J = [];
% connect vertically (down, then up)
is = [1:N]'; is([height:height:N])=[];
js = is+1;
I = [I;is;js];
J = [J;js;is];
% connect horizontally (right, then left)
is = [1:N-height]';
js = is+height;
I = [I;is;js];
J = [J;js;is];

E = [I,J];

end