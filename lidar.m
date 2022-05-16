clc
clear all
close all

seed = 1001969;
rng(seed);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part A
lenSample = 1000;
%x = linspace(0,1,lenSample);
%y = linspace(0,2,lenSample);

x = rand(1000,1);
y = 2*rand(1000,1);

xyzPoints = zeros(lenSample,3);
for i = 1:lenSample
    xyzPoints(i,1) = x(i);
    xyzPoints(i,2) = y(i);
    xyzPoints(i,3) = x(i)^2 + y(i)^3;
end
ptCloud = pointCloud(xyzPoints);
figure();
pcshow(ptCloud)
xlabel('X')
ylabel('Y')
zlabel('Z')
    

% positive rotation of 60 degrees around the z axis
theta = deg2rad(60);
rot = [cos(theta) -sin(theta) 0; ...
       sin(theta)  cos(theta) 0; ...
               0           0  1];
% translate 2 units in x and 4 units in y           
trans = [2, 4, 0];
tform = rigid3d(rot,trans);

ptCloudOut = pctransform(ptCloud,tform);
figure();
pcshow(ptCloudOut)
xlabel('X')
ylabel('Y')
zlabel('Z')
    

tform_ICP = pcregistericp(ptCloud, ptCloudOut);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part B
load('lidarData.mat')
load('lidarLabel.mat')
intensityIndex = 4;
lenPoints = length(lidarData{1,1});
lenScans = length(lidarData);

% Get var, mean, max for each pointCloud intensity
intensityFeatures = zeros(lenScans,3);
for i = 1:lenScans
    pointCloudIntensity = lidarData{i}(:,4);
    intensityFeatures(i,1) = var(pointCloudIntensity);
    intensityFeatures(i,2) = sum(pointCloudIntensity)/lenPoints;
    intensityFeatures(i,3) = max(pointCloudIntensity);
end


PCAFeatures = zeros(lenScans,1);
for i = 1:lenScans
    pointCloudShape = lidarData{i}(:,1:3);
    
    [coeff,score,latent,tsquared,explained,mu] = pca(pointCloudShape);
    PCAFeatures(i,1) = latent(1);
    %PCAFeatures(i,2) = latent(2);
    %PCAFeatures(i,3) = latent(3);
end

features = [intensityFeatures];

labels = string(lidarLabel).';
G = findgroups(labels);

c = cvpartition(G, 'HoldOut', 0.3, 'Stratify', true);

testIdx = test(c);
trainIdx = training(c);

testFeatures = features(testIdx,:);
trainFeatures = features(trainIdx,:);
testLabel = labels(testIdx);
trainLabel = labels(trainIdx);

uniqueLabels = unique(testLabel);

%for i = 1:length(trainFeatures(1,:))
%    trainFeatures(:,i) = normalize(trainFeatures(:,i));
%    testFeatures(:,i) = normalize(testFeatures(:,i));
%end


t = templateSVM('KernelFunction','linear','Standardize',true);
SVMClassifier = fitcecoc(trainFeatures, trainLabel, 'Learners', t);

predictions = predict(SVMClassifier, testFeatures);


accuracy = sum(string(predictions) == testLabel)/ length(testLabel) * 100