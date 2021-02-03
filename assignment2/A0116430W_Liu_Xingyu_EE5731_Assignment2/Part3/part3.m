%% Part 3
clc;
clear;

% load images
im00 = double(imread('test00.jpg'));
im09 = double(imread('test09.jpg'));

% get the image size
[H00 W00 D00] = size(im00);
[H09 W09 D09] = size(im09);

% rearrange the dimension
im1 = reshape(permute(im00,[2 1 3]),1,[],3);
im2 = reshape(permute(im09,[2 1 3]),1,[],3);

num_pixel = H00 * W00;

% read camera matrix from the file
cam_matrix = readmatrix('cameras.txt');
K1 = cam_matrix(1:3, 1:3);
R1 = cam_matrix(4:6, 1:3);
T1 = cam_matrix(7:7, 1:3)';
K2 = cam_matrix(8:10, 1:3);
R2 = cam_matrix(11:13, 1:3);
T2 = cam_matrix(14:14, 1:3)';

%% parameters setting based on the paper "Consistent Depth Maps Recovery from a Video Sequence"
disparity = 0.0001:0.0002:0.01;
epsilon = 50;
ws = 20 / (disparity(end) - disparity(1));
sigma = 10;
eta = 0.05 * (disparity(end) - disparity(1));
class = size(disparity, 2);

% get the epipolar line
xh = [repmat(1:W00, 1, H00); reshape(repmat(1:H00, W00, 1), 1, num_pixel); ones(1, num_pixel)];
term1 = K2*R2'*(T1-T2)*disparity;
term2 = K2*R2'*R1*inv(K1)*xh;

% project im2 pixel to im1 coordinate
projected = [];
for i=1:length(disparity)
    xprimeh = term2 + repmat(term1(:,i),1,num_pixel);
    xprimeh(1,:) = xprimeh(1,:)./xprimeh(3,:);
    xprimeh(2,:) = xprimeh(2,:)./xprimeh(3,:);
    xprimeh = round(xprimeh);
    xprimeh(2,xprimeh(2,:) <1) = 1;
    xprimeh(2,xprimeh(2,:)>H00) = H00;
    xprimeh(1,xprimeh(1,:) <1) = 1;
    xprimeh(1,find(xprimeh(1,:)>W00)) = W00;
    projected = [projected; xprimeh(1,:)+(xprimeh(2,:)-1)*W00;]; 
end

% get the projected im2
im1 = repmat(im1, class, 1);
im2 = reshape(im2(1, projected, :), class, num_pixel, []);

% get the unary
unary = sqrt(sum((im2 - im1).^2, 3));
% normalize unary
unary = sigma./(sigma + unary);
maximum = max(unary);
unary = unary./repmat(maximum, class, 1);
unary = 1 - unary;

% get lambda
img = permute(im00,[2 1 3]);
[H, W] = size(img(:, :, 1));
E = edges4connected(H, W);
lambda = get_lambda(epsilon, ws, num_pixel, img, E);

addpath('../GCMex')

PAIRWISE = sparse(E(:, 1),E(:, 2), double(lambda), num_pixel, num_pixel, 4 * num_pixel);
labelcost = pdist2(disparity', disparity');
labelcost(labelcost > eta) = eta;
segclass = disparity(end) * ones(1, num_pixel);
[DispClass, Einit, Eafter] = GCMex(segclass, single(unary), PAIRWISE, single(labelcost));

%% get the depth map and save
depth_map = mat2gray(reshape(disparity(DispClass+1), W00, H00)');
figure
imshow(depth_map);
imwrite(depth_map, 'depth_map.png');

%% functions

function lambda = get_lambda(epsilon, ws, num_pixel, img, E)
[H, W] = size(img(:, :, 1));

top = num_pixel - W;
bot =  num_pixel - W;
left = num_pixel - H;
right = num_pixel - H;

% seperate E into 4 neighbors
E_top = E((num_pixel - W + 1):(top+num_pixel - W), :);
E_bot = E(1:bot, :);
E_left = E((top + num_pixel - W + right + 1):(top + num_pixel - W + right + left), :);
E_right = E((top + num_pixel - W + 1):(top + num_pixel - W + right), :);

% seperate into 3 channels rgb
r = img(:, :, 1);
g = img(:, :, 2);
b = img(:, :, 3);
intensity = (r + g + b) / 3;

% find the U matrix for all directions bot, top, right, left
array = zeros(1, num_pixel);
array(unique(E_top(:, 1))) = array(unique(E_top(:, 1))) + 1;
top1 = find(array == 1);
top2 = find(array == 0);
total_top = zeros(num_pixel, 1);
total_top(top1) = 1./(sqrt((intensity(E_top(:, 1)) - intensity(E_top(:, 2))).^2) + epsilon);

array = zeros(1, num_pixel);
array(unique(E_bot(:, 1))) = array(unique(E_bot(:, 1))) + 1;
bot1 = find(array == 1);
bot2 = find(array == 0);
total_bot = zeros(num_pixel, 1);
total_bot(bot1) = 1./(sqrt((intensity(E_bot(:, 1)) - intensity(E_bot(:, 2))).^2) + epsilon);

array = zeros(1, num_pixel);
array(unique(E_left(:, 1))) = array(unique(E_left(:, 1))) + 1;
left1 = find(array == 1);
left2 = find(array == 0);
total_left = zeros(num_pixel, 1);
total_left(left1) = 1./(sqrt((intensity(E_left(:, 1)) - intensity(E_left(:, 2))).^2) + epsilon);

array = zeros(1, num_pixel);
array(unique(E_right(:, 1))) = array(unique(E_right(:, 1))) + 1;
right1 = find(array == 1);
right2 = find(array == 0);
total_right = zeros(num_pixel, 1);
total_right(right1) = 1./(sqrt((intensity(E_right(:, 1)) - intensity(E_right(:, 2))).^2) + epsilon);

% get the number of connections of all the edges
num_connection = zeros(1, num_pixel);
num_connection(unique(E_top(:, 1))) = num_connection(unique(E_top(:, 1))) + 1;
num_connection(unique(E_bot(:, 1))) = num_connection(unique(E_bot(:, 1))) + 1;
num_connection(unique(E_left(:, 1))) = num_connection(unique(E_left(:, 1))) + 1;
num_connection(unique(E_right(:, 1))) = num_connection(unique(E_right(:, 1))) + 1;

% calculate and get the U matrix
U = num_connection'./(total_left + total_right + total_top + total_bot);

% remove index that were not used
lambda_top = ws.* U./(total_top + epsilon);
lambda_top(top2) = [];
lambda_bot = ws.* U./(total_bot + epsilon);
lambda_bot(bot2) = [];
lambda_left = ws.* U./(total_left + epsilon);
lambda_left(left2) = [];
lambda_right = ws.* U./(total_right + epsilon);
lambda_right(right2) = [];


% combine all lambda values
lambda = [lambda_bot;lambda_top;lambda_right;lambda_left];

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