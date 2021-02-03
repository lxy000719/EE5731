%% Part 1
clc;
clear all;

% get the noise image
img = imread('bayes_in.jpg');
[H, W, D] = size(img);

SOURCE_COLOR = [0, 0, 255]; % blue = foreground
SINK_COLOR = [245, 210, 110]; % yellow = background

% initialization
segclass = zeros(W*H, 1);
%pairwise = sparse(W*H,W*H);
unary = zeros(2, W*H);
[X Y] = meshgrid(1:2, 1:2);
labelcost = min(4, (X - Y).*(X - Y));
edge = 1;

for row = 0:H-1
    for col = 0:W-1
        pixel = 1+ row * W + col;
        % data term
        I = reshape(img(row+1, col+1,:),[1,3]);
        unary(1, pixel) = dist(I, SINK_COLOR);
        unary(2, pixel) = dist(I, SOURCE_COLOR);
        % prior term
        if row+1 < H
            A(edge) = pixel;
            B(edge) = 1+col+(row+1)*W;
            %pairwise(pixel, 1+col+(row+1)*W) = 1;
            edge = edge + 1;
        end
        if col+1 < W
            A(edge) = pixel;
            B(edge) = 1+(col+1)+row*W;
            %pairwise(pixel, 1+(col+1)+row*W) = 1;
            edge = edge + 1;
        end     
        if row-1 >= 0
            A(edge) = pixel;
            B(edge) = 1+col+(row-1)*W;
            %pairwise(pixel, 1+col+(row-1)*W) = 1;
            edge = edge + 1;
        end        
        if col-1 >= 0
            A(edge) = pixel;
            B(edge) = 1+(col-1)+row*W;
            %pairwise(pixel, 1+(col-1)+row*W) = 1;
            edge = edge + 1;
        end
    end
end

%% create the result images with different values of lambda
addpath('../GCMex')

lambda_lst = [1, 5, 10, 50, 100, 500, 1000, 1500];
for lambda = lambda_lst
    
pairwise = sparse(A, B, lambda);
[labels E Eafter] = GCMex(segclass, single(unary), pairwise, single(labelcost),1);
labels = reshape(labels, [W, H])';

for row = 1:H
    for col = 1:W
        if labels(row, col) == 1
            result(row, col, :) = SOURCE_COLOR;
        end
        if labels(row, col) == 0
            result(row, col, :) = SINK_COLOR;          
        end
    end
end

% show and save the denoise images with different values of lambda
figure;
imshow(uint8(result));
name = sprintf('lambda=%d.jpg', lambda);
imwrite(uint8(result), name);

end

%% function
function d = dist(c1, c2)

c1_db = double(c1);
c2_db = double(c2);
d = (abs(c1_db(1) - c2_db(1)) + abs(c1_db(2) - c2_db(2)) + abs(c1_db(3) - c2_db(3))) / 3;

end