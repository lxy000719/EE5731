%% Part 2
clc;
clear;

% reference: https://github.com/aminzabardast/SIFT-on-MATLAB

% get the SIFT keypoints
img = imread('im01.jpg');
img = double(rgb2gray(img));
keyPoints = SIFT(img,3,5,1.6);

%% show SIFT keypoints
img = SIFTKeypointVisualizer(img,keyPoints);
imshow(uint8(img));