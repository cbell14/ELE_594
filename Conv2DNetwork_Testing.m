function Conv2DNetwork_Testing(N_tv,N_test,folds,numClasses,miniBatchSize,SNR_list,testDIR,outputDIR)
%% Setup
data = [];
order = [];
SNRdB = [];
labels = [];
labels2 = [];
peakSNR = [];
totalSNR = [];
    
%% Load Data
imds = imageDatastore(testDIR,'IncludeSubfolders',true,'LabelSource','foldernames');

%% Load SNR and Label Information
for nn=1:1:numClasses
    temp_path = [testDIR,'\Case',num2str(nn)]; %Create Path for specific case
    for ff=N_tv+1:1:(N_tv+N_test)
        temp = load([temp_path,'\Count',num2str(ff),'.mat']);
        data_temp{ff-N_tv,:} = temp.data.time'; %copy time data
        order_temp{ff-N_tv,:} = temp.data.orders; %copy order information
        SNRdB_temp{ff-N_tv,:} = temp.data.SNR_dB; %copy SNR information
        labels_temp(ff-N_tv,1) = categorical(nn); %make label
        peakSNR_temp(ff-N_tv,1) = max(temp.data.SNR_dB); %make peak SNR
        totalSNR_temp(ff-N_tv,1) = sum(temp.data.SNR_dB); %make total SNR
        labels2_temp(ff-N_tv,1) = nn;
    end

    data = [data;data_temp];
    order = [order;order_temp];
    SNRdB = [SNRdB;SNRdB_temp];
    labels = [labels;labels_temp];
    labels2 = [labels2;labels2_temp];
    peakSNR = [peakSNR;peakSNR_temp];
    totalSNR = [totalSNR;totalSNR_temp];

    clearvars data_temp order_temp SNRdB_temp labels_temp labels2_temp peakSNR_temp toalSNR_temp
end
%% Preprocess
% for nn=1:1:numClasses
%     temp_path = [testDIR,'\Case',num2str(nn)]; %Create Path for specific case
%     for ff=N_tv+1:1:(N_tv+N_test)
%         temp = load([temp_path,'\Count',num2str(ff),'.mat']);
%         labels2_temp(ff-N_tv,1) = nn;
%     end
% 
%     labels2 = [labels2;labels2_temp];
% 
%     clearvars labels2_temp
% end



%% Make Predictions & Visualize Performance
load([outputDIR,'2DConvNetwork.mat']);
preds = zeros(folds,size(imds.Files,1),numClasses);
Acc_peakSNR = NaN(folds,length(SNR_list));
totalSNR_vec = min(totalSNR):1:max(totalSNR);
Acc_totalSNR = NaN(folds,length(totalSNR_vec));

for kk=1:1:folds
    Y_Pred(kk,:,:) = predict(net(kk),imds, ...
            'MiniBatchSize',miniBatchSize, ...
            'SequencePaddingDirection',"left");

    %Format data for Confusion Matrix
    for ii=1:1:size(imds.Files,1)
         [M, idx] = max(Y_Pred(kk,ii,:));
         preds(kk,ii,idx) = idx;
         if idx==labels2(ii)
             preds2(ii,kk) = 1;
         else
             preds2(ii,kk) = 0;
         end
    end
    [m(kk,:,:) cm_order(kk,:)] = confusionmat(squeeze(max(squeeze(preds(kk,:,:)),[],2)),labels2);

    figure('Name',['Network ',num2str(kk),' Confusion Chart']);
    confusionchart(squeeze(m(kk,:,:)),'ColumnSummary','column-normalized',...
        'RowSummary','row-normalized');
    title(get(gcf,'Name'));
    savefig([outputDIR,'Network',num2str(kk),'ConfusionMatrix.fig'])

    % Accuracy against peak SNR
    for ii=1:1:length(SNR_list)
        temp1 = peakSNR==SNR_list(ii);
        temp1_sum = sum(temp1); %total number of data points at this peak SNR
        if temp1_sum==0
            Acc_peakSNR(kk,ii) = NaN;
        else
            temp1_correct = sum(preds2(temp1,kk));
            Acc_peakSNR(kk,ii) = temp1_correct/temp1_sum;
        end
    
        clearvars temp1 temp1_sum temp1_correct
    end

    figure('Name','2DConv CNN Accuracy vs Peak SNR')
    plot(SNR_list,Acc_peakSNR(kk,:)*100,'-o','linewidth',2.5,'markersize',8)
    grid on;
    xlim([min(SNR_list)-0.5,max(SNR_list)+0.5])
    xlabel('Peak SNR, dB')
    ylim([0 100])
    ylabel('Accuracy, %')
    title(get(gcf,'Name'))
    savefig([outputDIR,'Network',num2str(kk),'Accuracy_PeakSNR.fig'])

    % Accuracy against total SNR
    for ii=1:1:length(totalSNR_vec)
        temp1 = totalSNR==totalSNR_vec(ii);
        temp1_sum = sum(temp1); %total number of data points at this peak SNR
        if temp1_sum==0
            Acc_totalSNR(kk,ii) = NaN;
        else
            temp1_correct = sum(preds2(temp1,kk));
            Acc_totalSNR(kk,ii) = temp1_correct/temp1_sum;
        end

        clearvars temp1 temp1_sum temp1_correct
    end

    figure('Name','2DConv CNN Accuracy vs Total SNR')
    plot(totalSNR_vec,Acc_totalSNR(kk,:),'-o','linewidth',2.5,'markersize',8)
    grid on;
    xlim([min(totalSNR_vec)-0.5,max(totalSNR_vec)+0.5])
    xlabel('Total SNR, dB')
    ylim([min(Acc_totalSNR(kk,:))*0.9,max(Acc_totalSNR(kk,:))*1.1])
    ylabel('Accuracy, %')
    title(get(gcf,'Name'))
    savefig([outputDIR,'Network',num2str(kk),'Accuracy_TotalSNR.fig'])
end
