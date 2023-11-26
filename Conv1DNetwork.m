function Conv1DNetwork(N_tv,numClasses,filterSize,numfilters,mEpochs,folds,miniBatchSize,trainDIR,outputDIR)
    %% Setup
    data = [];
    order = [];
    SNRdB = [];
    labels = [];
    peakSNR = [];
    totalSNR = [];
    
    %% Load Data and Preprocess
    %Load Data
    for nn=1:1:numClasses
        temp_path = [trainDIR,'\Case',num2str(nn)]; %Create Path for specific case
        for ff=1:1:N_tv
            temp = load([temp_path,'\Count',num2str(ff),'.mat']);
            data_temp{ff,:} = temp.data.time; %copy time data
            order_temp{ff,:} = temp.data.orders; %copy order information
            SNRdB_temp{ff,:} = temp.data.SNR_dB; %copy SNR information
            labels_temp(ff,1) = categorical(nn); %make label
            peakSNR_temp(ff,1) = max(temp.data.SNR_dB); %make peak SNR
            totalSNR_temp(ff,1) = sum(temp.data.SNR_dB); %make total SNR
        end
    
        data = [data;data_temp];
        order = [order;order_temp];
        SNRdB = [SNRdB;SNRdB_temp];
        labels = [labels;labels_temp];
        peakSNR = [peakSNR;peakSNR_temp];
        totalSNR = [totalSNR;totalSNR_temp];
    
        clearvars data_temp order_temp SNRdB_temp labels_temp peakSNR_temp toalSNR_temp
    end
    
    %Scale data
    for ii=1:1:size(data,1)
        temp_max(ii) = max(data{ii,1});
        temp_min(ii) = min(data{ii,1});
    end
    max_data = max(temp_max);
    min_data = min(temp_min);
    for ii=1:1:size(data,1)
        data_scaled{ii,1} = (2*(data{ii,1} - min_data) / (max_data - min_data)) - 1;
    end
    save([outputDIR,'TrainingData_Scaling_Terms.mat'],'max_data','min_data')
    
    %% Split into Train and Validation
    %Shuffle Data
    idx = randperm(size(data_scaled,1));
    block = length(idx)/folds;
    
    train_idx(1,:) = idx(1:end-block);
    val_idx(1,:) = idx(end-block+1:end); %last block
    
    train_idx(2,:) = idx(block+1:end);
    val_idx(2,:) = idx(1:block); %first block
    
    train_idx(3,:) = idx([1:block,(2*block)+1:end]);
    val_idx(3,:) = idx(block+1:2*block); %second block
    
    train_idx(4,:) = idx([1:(2*block),(3*block)+1:end]);
    val_idx(4,:) = idx((2*block)+1:3*block); %third block
    
    train_idx(5,:) = idx([1:(3*block),(4*block)+1:end]);
    val_idx(5,:) = idx((3*block)+1:4*block); %fourth block
    
    %% Build Network
    layers = [...
        sequenceInputLayer(size(data_scaled{1,1},2))
        
        convolution1dLayer(filterSize,numfilters,Padding='causal')
        reluLayer
        layerNormalizationLayer
    
        convolution1dLayer(filterSize,2*numfilters,Padding='causal')
        reluLayer
        layerNormalizationLayer
    
        convolution1dLayer(filterSize,4*numfilters,Padding='causal')
        reluLayer
        layerNormalizationLayer
    
        convolution1dLayer(filterSize,8*numfilters,Padding='causal')
        reluLayer
        layerNormalizationLayer
    
        globalMaxPooling1dLayer
    
        dropoutLayer
    
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
        ];
    
    %% Train
    for kk=1:1:folds
        for ii=1:1:size(train_idx,2)
            X_Train{ii,1} = data_scaled{train_idx(kk,ii),1}';
            X_Train_folds{ii,kk} = X_Train{ii,:};
        end
        Y_Train(:,kk) = labels(train_idx(kk,:));
        peakSNR_trainfolds(kk,:) = peakSNR(train_idx(kk,:));
        totalSNR_trainfolds(kk,:) = totalSNR(train_idx(kk,:));
    
        for ii=1:1:size(val_idx,2)
            X_Val{ii,1} = data_scaled{val_idx(kk,ii),1}';
            X_Val_folds{ii,kk} = X_Val{ii,:};
        end
        Y_Val(:,kk) = labels(val_idx(kk,:));
        peakSNR_valfolds(kk,:) = peakSNR(val_idx(kk,:));
        totalSNR_valfolds(kk,:) = totalSNR(val_idx(kk,:));
        
        options = trainingOptions('adam',...
            MiniBatchSize=miniBatchSize,...
            MaxEpochs=mEpochs,...
            SequencePaddingDirection='left',...
            ValidationData={X_Val,Y_Val(:,kk)},...
            Plots='training-progress',...
            Shuffle='once',...
            ExecutionEnvironment='gpu',...
            Verbose=0);
    
        [net(kk) info(kk)] = trainNetwork(X_Train,Y_Train(:,kk),layers,options);
        delete(findall(0));
        clearvars X_Train X_Val
    end
    
    %calc k fold CV statistics
    for ii=1:1:folds
        CV.acc(ii) = info(ii).FinalValidationAccuracy;
    end
    CV.acc_mean = mean(CV.acc);
    CV.acc_std = std(CV.acc);
    
    % Save Network and Info
    save([outputDIR,'\1DConvNetwork.mat'],'net','info','CV')
end
