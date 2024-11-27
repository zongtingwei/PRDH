function [train_F, train_L, test_F, test_L] = DIVDATA37(dataName)
file = ['dataset/', dataName, '.mat'];
load(file)

dataMat = ins;
len = size(dataMat, 1);
maxV = max(dataMat);
minV = min(dataMat);
range = maxV - minV;
newdataMat = (dataMat - repmat(minV, [len, 1])) ./ (repmat(range, [len, 1]));
lab = lab;
indices = randperm(length(lab));
splitIdx = round(0.7 * length(lab)); % 70% for training
trainIdx = indices(1:splitIdx);
testIdx = indices(splitIdx+1:end);
train_F = newdataMat(trainIdx, :);
train_L = lab(trainIdx);
test_F = newdataMat(testIdx, :);
test_L = lab(testIdx);
end
