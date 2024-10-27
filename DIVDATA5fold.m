function [train_F, train_L, test_F, test_L] = DIVDATA5fold(dataName, iter)
% the features of a data set is marked with "ins" , the label of a data set
% is marked with "lab"
file = ['dataset/', dataName, '.mat'];
load(file)
global fold
fold = 5; % Change the fold to 5 for 5-fold cross-validation
dataMat = ins;
len = size(dataMat, 1);
maxV = max(dataMat);
minV = min(dataMat);
range = maxV - minV;
newdataMat = (dataMat - repmat(minV, [len, 1])) ./ (repmat(range, [len, 1]));
if mod(iter, 5) == 1
    Indices = crossvalind('Kfold', length(lab), fold);
    save('Indices.mat', 'Indices'); % Save the indices to a file
else
    load('Indices.mat'); % Load the indices from the file
end
if mod(iter, 5) == 0
    site = find(Indices == 5);
    test_F = newdataMat(site, :);
    test_L = lab(site);
    site2 = find(Indices ~= 5);
    train_F = newdataMat(site2, :);
    train_L = lab(site2);
else
    site = find(Indices == mod(iter, 5));
    test_F = newdataMat(site, :);
    test_L = lab(site);
    site2 = find(Indices ~= mod(iter, 5));
    train_F = newdataMat(site2, :);
    train_L = lab(site2);
end
end
