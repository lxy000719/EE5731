%% Part 4
clc;
clear;

%
% total frame = 141
for frame_num = 1:141

% read images
files = dir('Road/src/*.jpg');
images = fullfile('Road', 'src' , {files.name});
[H, W, ~] = size(imread(images{1}));

% rearrange the dimension
for i=1:length(files)
    im{i} = reshape(permute(double(imread(images{i})),[2 1 3]),1,[],3);
end

num_pixel = H * W;

% read camera matrics
camera = textread('Road/cameras.txt','%s');
camera = cellfun(@str2double, camera);
total_frame = camera(1);
camera(1) = [];

for i = 1: total_frame
    index = (21*(i-1) +1):21*i;
    K{i} = reshape(camera(index(1:9)),[3 3])';
    R{i} = reshape(camera(index(10:18)),[3 3])';
    T{i} = reshape(camera(index(19:21)),[1 3])';
end

%%
% set parameters
disparity = 0.0001:0.0002:0.01;
epsilon = 50;
ws = 20/(disparity(end) - disparity(1));
sigma = 10;
eta = 0.05*(disparity(end) - disparity(1));
class = size(disparity,2); 
num_frame = 5;

% get the epipolar line
xh = [repmat(1:W,1,H); reshape(repmat(1:H,W,1),1,num_pixel); ones(1,num_pixel)];

%%
[ind1, ind2] = sort(pdist2((1:total_frame)',(1:total_frame)'));
ind2(1,:) = [];
    
total_unary = zeros(class, num_pixel);
neighbor = ind2(1:num_frame,frame_num); 

for neighbor_index = 1:length(neighbor)
    j = neighbor(neighbor_index);
    term1 = K{j}*R{j}'*(T{frame_num}-T{j})*disparity;
    term2 = K{j}*R{j}'*R{frame_num}*inv(K{frame_num})*xh;
    projected = [];
    for i=1:length(disparity)
        xprimeh = term2 + repmat(term1(:,i),1,num_pixel);
        xprimeh(1,:) = xprimeh(1,:)./xprimeh(3,:);
        xprimeh(2,:) = xprimeh(2,:)./xprimeh(3,:);
        xprimeh = round(xprimeh);
        xprimeh(2,xprimeh(2,:) <1) = 1;
        xprimeh(2,xprimeh(2,:)>H) = H;
        xprimeh(1,xprimeh(1,:) <1) = 1;
        xprimeh(1,find(xprimeh(1,:)>W)) = W;
        projected = [projected; xprimeh(1,:)+(xprimeh(2,:)-1)*W;]; 
    end
    im_org = repmat(im{frame_num}, class, 1);
    projected_img = im{j};
    projected_img = reshape(projected_img(1, projected, :), class, num_pixel, []);
    unary = sqrt(sum((projected_img - im_org).^2, 3)); 
    unary = sigma./(sigma + unary);
    total_unary = unary + total_unary;
end

maximum = max(total_unary);
total_unary = total_unary./repmat(maximum, class, 1);
total_unary = 1 - total_unary;
    
img = permute(double(imread(images{frame_num})), [2 1 3]);
[H, W] = size(img(:, :, 1));
E = edges4connected(H, W);
lambda = get_lambda(epsilon, ws, num_pixel, img, E);

addpath('../GCMex')

% Find Labelcost
PAIRWISE = sparse(E(:, 1),E(:, 2), double(lambda), num_pixel, num_pixel, 4 * num_pixel);
labelcost = pdist2(disparity', disparity');
labelcost(labelcost > eta) = eta;
segclass = disparity(end) * ones(1, num_pixel);
[DispClass, Einit, Eafter] = GCMex(segclass, single(total_unary), PAIRWISE, single(labelcost));

depth_map = mat2gray(reshape(disparity(DispClass+1), H, W)');
imshow(depth_map);
filename = [int2str(frame_num),'.png'];
imwrite(depth_map, filename)

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

%%
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