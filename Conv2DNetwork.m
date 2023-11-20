%% Setup
% clearvars; close all; clc;
% trainDIR = 'C:\Users\cjbell\Desktop\ELE594_Data\8Classes\TrainingData\';
% N_tv = 8000; %Number of training data per case
% 
% order = [];
% SNRdB = [];
% peakSNR = [];
% totalSNR = [];
% 
% numClasses = 8;
% filterSize = 8; %function of sample rate?
% numfilters = 8;
% mEpochs = 10;

%% Load Data and Preprocess
% %Load Data
% imds = imageDatastore(trainDIR,'IncludeSubfolders',true,'LabelSource','foldernames');
% labelCount = countEachLabel(imds)
% img = readimage(imds,1);
% image_size = size(img);

% for nn=1:1:numClasses
%     temp_path = [trainDIR,'Case',num2str(nn)]; %Create Path for specific case
%     for ff=1:1:N_tv
%         temp = load([temp_path,'\Count',num2str(ff),'.mat']);
%         order_temp{ff,:} = temp.data.orders; %copy order information
%         SNRdB_temp{ff,:} = temp.data.SNR_dB; %copy SNR information
%         peakSNR_temp(ff,1) = max(temp.data.SNR_dB); %make peak SNR
%         totalSNR_temp(ff,1) = sum(temp.data.SNR_dB); %make total SNR
%     end
% 
%     order = [order;order_temp];
%     SNRdB = [SNRdB;SNRdB_temp];
%     peakSNR = [peakSNR;peakSNR_temp];
%     totalSNR = [totalSNR;totalSNR_temp];
% 
%     clearvars order_temp SNRdB_temp peakSNR_temp toalSNR_temp
% end

%% Split into Train and Validation
%Shuffle Data
% idx = randperm(size(data,1));
% % data_shuffle = data(idx);
% % order_shuffle = order(idx);
% % SNRdB_shuffle = SNRdB(idx);
% % labels_shuffle = labels(idx);
% % peakSNR_shuffle = peakSNR(idx);
% % totalSNR_shuffle = totalSNR(idx);
% folds=5;
% block = length(idx)/folds;
% 
% train_idx(1,:) = idx(1:end-block);
% val_idx(1,:) = idx(end-block+1:end); %last block
% 
% train_idx(2,:) = idx(block+1:end);
% val_idx(2,:) = idx(1:block); %first block
% 
% train_idx(3,:) = idx([1:block,(2*block)+1:end]);
% val_idx(3,:) = idx(block+1:2*block); %second block
% 
% train_idx(4,:) = idx([1:(2*block),(3*block)+1:end]);
% val_idx(4,:) = idx((2*block)+1:3*block); %third block
% 
% train_idx(5,:) = idx([1:(3*block),(4*block)+1:end]);
% val_idx(5,:) = idx((3*block)+1:4*block); %fourth block

%% Build Network
layers = [
    imageInputLayer([image_size 1])
    
    convolution2dLayer(filterSize,numfilters,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([1 2],'Stride',[1 2])
    
    convolution2dLayer(filterSize,2*numfilters,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([1 2],'Stride',[1 2])
    
    convolution2dLayer(filterSize,4*numfilters,'Padding','same')
    batchNormalizationLayer
    reluLayer

    dropoutLayer
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% Train
[imds_train,imds_val] = splitEachLabel(imds,N_tv*0.8,'randomize');

options = trainingOptions('adam',...
    MiniBatchSize=48,...
    MaxEpochs=mEpochs,...
    SequencePaddingDirection='left',...
    ValidationData=imds_val,...
    Plots='training-progress',...
    Shuffle='once',...
    ExecutionEnvironment='gpu',...
    Verbose=0);

net = trainNetwork(imds_train,layers,options);

% for kk=1:1:1  
    % for ii=1:1:size(train_idx,2)
    %     X_Train{ii,1} = data{train_idx(kk,ii),1}';
    % end
    % Y_Train = labels(train_idx(kk,:));
    % for ii=1:1:size(val_idx,2)
    %     X_Val{ii,1} = data{val_idx(kk,ii),1}';
    % end
%     % Y_Val = labels(val_idx(kk,:));
% 
%     options = trainingOptions('adam',...
%         MiniBatchSize=48,...
%         MaxEpochs=mEpochs,...
%         SequencePaddingDirection='left',...
%         ValidationData={X_Val,Y_Val},...
%         Plots='training-progress',...
%         Shuffle='once',...
%         ExecutionEnvironment='gpu',...
%         Verbose=0);
% 
%     net(kk) = trainNetwork(X_Train,Y_Train,layers,options);
% 
%     clearvars X_Train Y_Train X_Val Y_Val
% end

%% Save Network
