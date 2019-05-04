clear;
data = csvread('features_zimu.txt'); % read HOG features
%data=data(1:500,:);

labels = data(:,1);%标签

%一共的样本个数
totalInputs=size(labels);
totalInputs=totalInputs(1);

%具体的样本的特征值
data=data(:,2:end);

%分离出单独的标签页
uniq=unique(labels);

binLabels=zeros(totalInputs,size(uniq,1));

% create binary labels
for i=1:totalInputs
    binLabels(i,labels(i))=1;
end


 % randomly shuffle the input rows
rperm=randperm(size(binLabels,1));  %随机打乱
data=data(rperm,:);
binLabels=binLabels(rperm,:);

% split into test and train
split=0.95*size(data,1);
x=data(1:split,:)';
t=binLabels(1:split,:)';


xtest=data(split:end,:)';
ltest=binLabels(split:end,:)';

% Create a Pattern Recognition Network
hiddenLayerSize = 500;
net = patternnet([hiddenLayerSize hiddenLayerSize/2],'trainbr');


% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 5/100;

net.trainFcn = 'traincgp';
% net.trainParam.max_fail=10;
%  net.trainParam.lr=0.005;
%  net.trainParam.mc=0.4;
% 
% Train the Network
[net,tr] = train(net,x,t);


 system('E:\\\练手项目\\手写英文字母单词识别\\Chars74k_Recognition\\code\\tiqu\\Debug\\tiqu.exe img011-001.png')
 ft = load('E:\练手项目\手写英文字母单词识别\Chars74k_Recognition\code\tiqu\tiqu\features.txt');
 ft=ft(1,2:325);

 %Yind   11代表A 12代表B  依次往后类推，先识别大写，再识别小写
 
% Test the Network
y = net(ft');
tind = vec2ind(ltest);
yind = vec2ind(y);

 yind




