function Conv2DNetwork(N_tv,numClasses,filterSize,numfilters,mEpochs,folds,miniBatchSize,trainDIR,outputDIR)
    order = [];
    SNRdB = [];
    peakSNR = [];
    totalSNR = [];
    
    %% Load Data and Preprocess
    %Load Data
    imds = imageDatastore(trainDIR,'IncludeSubfolders',true,'LabelSource','foldernames');
    labelCount = countEachLabel(imds);
    img = readimage(imds,1);
    image_size = size(img);
    % for ii=1:1:size(imds.Labels,1)
    %     if imds.Labels(ii,1)=='Case1'
    %         labels(ii,1) = categorical(1);
    %     elseif imds.Labels(ii,1)=='Case2'
    %         labels(ii,1) = categorical(2);
    %     elseif imds.Labels(ii,1)=='Case3'
    %         labels(ii,1) = categorical(3);
    %     elseif imds.Labels(ii,1)=='Case4'
    %         labels(ii,1) = categorical(4);
    %     elseif imds.Labels(ii,1)=='Case5'
    %         labels(ii,1) = categorical(5);
    %     elseif imds.Labels(ii,1)=='Case6'
    %         labels(ii,1) = categorical(6);
    %     elseif imds.Labels(ii,1)=='Case7'
    %         labels(ii,1) = categorical(7);
    %     elseif imds.Labels(ii,1)=='Case8'
    %         labels(ii,1) = categorical(8);
    %     else
    %         error('Something wrong in Labels')
    %     end
    % end
    
    %Scale Data
    % for ii=1:1:size(imds.Files,1)
    %     img = readimage(imds,ii);
    %         if ii==1
    %             image_size = size(img);
    %         end
    %     data_scaled{ii,1} = double(img)/255;
    % end
    
    
    
    
    
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
    % idx = randperm(size(data_scaled,1));
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
        % sequenceInputLayer([image_size 1])
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
    % for kk=1:1:folds
    %     for ii=1:1:size(train_idx,2)
    %         X_Train{ii,1} = data_scaled{train_idx(kk,ii),1};
    %         X_Train_folds{ii,kk} = X_Train{ii,:};
    %     end
    %     Y_Train(:,kk) = labels(train_idx(kk,:));
    %     % peakSNR_trainfolds(kk,:) = peakSNR(train_idx(kk,:));
    %     % totalSNR_trainfolds(kk,:) = totalSNR(train_idx(kk,:));
    % 
    %     for ii=1:1:size(val_idx,2)
    %         X_Val{ii,1} = data_scaled{val_idx(kk,ii),1};
    %         X_Val_folds{ii,kk} = X_Val{ii,:};
    %     end
    %     Y_Val(:,kk) = labels(val_idx(kk,:));
    %     % peakSNR_valfolds(kk,:) = peakSNR(val_idx(kk,:));
    %     % totalSNR_valfolds(kk,:) = totalSNR(val_idx(kk,:));
    % 
    %     options = trainingOptions('adam',...
    %         MiniBatchSize=miniBatchSize,...
    %         MaxEpochs=mEpochs,...
    %         SequencePaddingDirection='left',...
    %         ValidationData={X_Val,Y_Val(:,kk)},...
    %         Plots='training-progress',...
    %         Shuffle='once',...
    %         ExecutionEnvironment='gpu',...
    %         Verbose=0);
    % 
    %     [net(kk) info(kk)] = trainNetwork(X_Train,Y_Train(:,kk),layers,options);
    %     delete(findall(0));
    %     clearvars X_Train X_Val
    % end
    % 
    % %calc k fold CV statistics
    % for ii=1:1:folds
    %     CV.acc(ii) = info(ii).FinalValidationAccuracy;
    % end
    % CV.acc_mean = mean(temp_cv);
    % CV.acc_std = std(temp_cv);
    % 
    % % Save Network and Info
    % save([outputDIR,'\Fold',num2str(kk)],'net','info','CV')
    
    %% Train
    [imds_temp1,imds_temp2,imds_temp3,imds_temp4,imds_temp5] = splitEachLabel(imds,N_tv/folds,N_tv/folds,N_tv/folds,N_tv/folds,N_tv/folds,'randomize');
    
    %fold 1, T=1-4, V=5
    imds_fold1_train = imageDatastore([imds_temp1.Files;imds_temp2.Files;imds_temp3.Files;imds_temp4.Files],'LabelSource','foldernames');
    imds_fold1_val = imageDatastore(imds_temp5.Files,'LabelSource','foldernames');
    
    %fold 2, T=2-5, V=1
    imds_fold2_train = imageDatastore([imds_temp2.Files;imds_temp3.Files;imds_temp4.Files;imds_temp5.Files],'LabelSource','foldernames');
    imds_fold2_val = imageDatastore(imds_temp1.Files,'LabelSource','foldernames');
     
    %fold 3, T=1,3-5, V=2
    imds_fold3_train = imageDatastore([imds_temp1.Files;imds_temp3.Files;imds_temp4.Files;imds_temp5.Files],'LabelSource','foldernames');
    imds_fold3_val = imageDatastore(imds_temp2.Files,'LabelSource','foldernames');
    
    %fold 4, T=1-2,4-5, V=3
    imds_fold4_train = imageDatastore([imds_temp1.Files;imds_temp2.Files;imds_temp4.Files;imds_temp5.Files],'LabelSource','foldernames');
    imds_fold4_val = imageDatastore(imds_temp3.Files,'LabelSource','foldernames');
    
    %fold 5, T=1-3,5, V=4
    imds_fold5_train = imageDatastore([imds_temp1.Files;imds_temp2.Files;imds_temp3.Files;imds_temp5.Files],'LabelSource','foldernames');
    imds_fold5_val = imageDatastore(imds_temp4.Files,'LabelSource','foldernames');
    
    options1 = trainingOptions('adam',...
        MiniBatchSize=miniBatchSize,...
        MaxEpochs=mEpochs,...
        SequencePaddingDirection='left',...
        ValidationData=imds_fold1_val,...
        Plots='training-progress',...
        Shuffle='once',...
        ExecutionEnvironment='gpu',...
        Verbose=0);
    [net(1) info(1)] = trainNetwork(imds_fold1_train,layers,options1);
    delete(findall(0));

    options2 = trainingOptions('adam',...
        MiniBatchSize=miniBatchSize,...
        MaxEpochs=mEpochs,...
        SequencePaddingDirection='left',...
        ValidationData=imds_fold2_val,...
        Plots='training-progress',...
        Shuffle='once',...
        ExecutionEnvironment='gpu',...
        Verbose=0);
    [net(2) info(2)] = trainNetwork(imds_fold2_train,layers,options2);
    delete(findall(0));

    options3 = trainingOptions('adam',...
        MiniBatchSize=miniBatchSize,...
        MaxEpochs=mEpochs,...
        SequencePaddingDirection='left',...
        ValidationData=imds_fold3_val,...
        Plots='training-progress',...
        Shuffle='once',...
        ExecutionEnvironment='gpu',...
        Verbose=0);
    [net(3) info(3)] = trainNetwork(imds_fold3_train,layers,options3);
    delete(findall(0));

    options4 = trainingOptions('adam',...
        MiniBatchSize=miniBatchSize,...
        MaxEpochs=mEpochs,...
        SequencePaddingDirection='left',...
        ValidationData=imds_fold4_val,...
        Plots='training-progress',...
        Shuffle='once',...
        ExecutionEnvironment='gpu',...
        Verbose=0);
    [net(4) info(4)] = trainNetwork(imds_fold4_train,layers,options4);
    delete(findall(0));

    options5 = trainingOptions('adam',...
        MiniBatchSize=miniBatchSize,...
        MaxEpochs=mEpochs,...
        SequencePaddingDirection='left',...
        ValidationData=imds_fold5_val,...
        Plots='training-progress',...
        Shuffle='once',...
        ExecutionEnvironment='gpu',...
        Verbose=0);
    [net(5) info(5)] = trainNetwork(imds_fold5_train,layers,options5);
    delete(findall(0));
    
    %calc k fold CV statistics
    for ii=1:1:folds
        CV.acc(ii) = info(ii).FinalValidationAccuracy;
    end
    CV.acc_mean = mean(CV.acc);
    CV.acc_std = std(CV.acc);
    
    % Save Network and Info
    save([outputDIR,'\2DConvNetwork.mat'],'net','info','CV')
end