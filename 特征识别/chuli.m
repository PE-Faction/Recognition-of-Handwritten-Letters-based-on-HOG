clear;
load 'net.mat';
load 'tr.mat'
%打开图像
img =imread('img012-001.png');
figure;
imshow(img);
s
%灰度化
grayImg =rgb2gray(img);
figure;
imshow(grayImg);

%阈值二值化
thresh=128;
twoImg=im2bw(grayImg,thresh/255);%对图像二值化
figure;
imshow(twoImg);

%反色
twoImg=~twoImg;
figure;
imshow(twoImg);

%中值滤波
mid=medfilt2(twoImg,[5,5]);
figure;
imshow(mid);


% 找到边界
[m,n]=size(mid);
c=0;
m1=[]
for i=1:m
    for j=1:n
       c=mid(i,j)+c; 
    end
    if(c~=0)
        m1=[m1 i]
    end
    c=0;
end

n1=[]
c=0
for i=1:n
    for j=1:m
       c=mid(j,i)+c; 
    end
    if(c~=0)
        n1=[n1 i]
    end
    c=0;
end
[a b]=size(m1);
[c d]=size(n1);
chuli1 =mid(m1(1):m1(b),n1(1):n1(d));
figure;
imshow(chuli1);
chulihou =imresize(chuli1,[28 28])

chulihou1=zeros(32,32);
[m,n]=size(chulihou);
c=0;
m1=[]
for i=3:m
    for j=3:n
           chulihou1(i,j)= chulihou(i-2,j-2)
    end
end
figure;
imshow(chulihou1)
[m n]=size(chulihou1)
for i=1:n
    for j=1:m
        if (chulihou1(i,j)==1)
            chulihou1(i,j)=255;
        end
    end
end
[featureVector,hogVisualization] = extractHOGFeatures(chulihou1);

% c=zeros(32,32);
% for i=1:32
%     for j=1:32
%         c(i,j)=img1(i,j)-chulihou1(i,j);
%     end
% end



% Test the Network
y = net(featureVector');
yind = vec2ind(y);







% %目标区域的阈值矫正
% [m,n]=size(twoImg);
% m1=m*0.05;
% n1=n*0.05;















% 大津法二值化
%  A=imread(D:\A);
%     thresh=graythresh(A);%确定二值化阈值
%     B=im2bw(A,thresh);%对图像二值化