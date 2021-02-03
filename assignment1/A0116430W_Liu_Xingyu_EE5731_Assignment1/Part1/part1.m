%% Part 1
clc;
clear;

% kernels

img = double(rgb2gray((imread('cat.jpg'))));
[imghgt, imgwdh] = size(img);


% sobel kernel horizontal and vertical
sobel_Gy = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
sobel_Gx = [-1, -2, -1; 0, 0, 0; 1, 2, 1];

% gaussian kernel
gaussian_kernel1 = [1, 2, 1; 2, 4, 2; 1, 2, 1];
gaussian_kernel2 = [1, 4, 7, 4, 1;...
                    4, 16, 26, 16, 4;...
                    7, 26, 41, 26, 7;...
                    4, 16, 26, 16, 4;...
                    1, 4, 7, 4, 1;...
                    ];
gaussian_kernel3 = [0, 0, 0, 5, 0, 0, 0;...
                    0, 5, 18, 32, 18, 5, 0;...
                    0, 18, 64, 100, 64, 18, 0;...
                    5, 32, 100, 100, 100, 32, 5;...
                    0, 18, 64, 100, 64, 18, 0;...
                    0, 5, 18, 32, 18, 5, 0;...
                    0, 0, 0, 5, 0, 0, 0;...
                    ];

% haar like kernel
haar1 = [-1, 1];
haar2 = [-1; 1];
haar3 = [1, -1, 1];
haar4 = [1; -1; 1];
haar5 = [-1, 1; 1, -1];

haar6 = [-1, -1, -1, 1, 1, 1; -1, -1, -1, 1, 1, 1; -1, -1, -1, 1, 1, 1];

%% 2D Convolution

kernel = haar6; % please change the kernel here to do different convolution
[kernelhgt1, kernelwdh1] = size(kernel);
output1 = [];

for i = 1:imghgt - kernelhgt1 + 1
    for j = 1:imgwdh - kernelwdh1 + 1
        
        sum = 0;
        
        area = img(i:i + kernelhgt1 - 1, j:j + kernelwdh1 - 1);     
        for k = 1:kernelhgt1
            for l = 1:kernelwdh1
                sum = sum + area(k, l) * kernel(k, l);
            end
        end
        output1(i, j) = sum;
    end
end

figure(1)
imshow(output1, []);


%% below is the for combining sobel vertical and horizontal kernel

kernel1 = sobel_Gx;
kernel2 = sobel_Gy;

[kernelhgt2, kernelwdh2] = size(kernel1);
output2 = [];

for i = 1:imghgt - kernelhgt2 + 1
    for j = 1:imgwdh - kernelwdh2 + 1
        
        sum = 0;
        sum1 = 0;
        sum2 = 0;
        
        area = img(i:i + kernelhgt2 - 1, j:j + kernelwdh2 - 1);       
        for k = 1:kernelhgt2
            for l = 1:kernelwdh2
                sum1 = sum1 + area(k, l) * kernel1(k, l);
                sum2 = sum2 + area(k, l) * kernel2(k, l);
                sum = sqrt(sum1^2 + sum2^2);
            end
        end
        output2(i, j) = sum;
    end
end

figure(2)
imshow(output2, []);

