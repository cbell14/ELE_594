clearvars; close all; clc;
%% Setup
N_tv = 8000; %1000 %Number of training and test data samples for each class
N_test = 2000; %100 %Number of test data samples for each class

harmonic_list = 1:1:10; %List of possible Harmonics Orders
SNR_list = -10:1:10; %List of possible SNR values in dB

f_source(1) = 40; %Shaft Rate, Hz
f_source(2) = 60; %Electrical Line Freq, Hz
f_source(3) = 80; %Electric Motor, Hz
f_source(4) = 14; %Train Freq, Hz
f_source(5) = 314; %Ball Pass Inner, Hz
f_source(6) = 166; %Ball Pass Outer, Hz
f_source(7) = 9; %Ball Spin, Hz
f_source(8) = 0; %No Source Case


fs = 8192;%2^nextpow2(max(f_source*max(harmonic_list))); %Time Domain Sampling Rate
t_len = 1; %Length of time signal
samples = 1:1:(t_len*fs);
rng(1);

trainDIR = 'C:\Users\cjbell\Desktop\ELE594_Data\8Classes\TrainingData';
testDIR = 'C:\Users\cjbell\Desktop\ELE594_Data\8Classes\TestData';

mkdir(trainDIR);
mkdir(testDIR);

for nn=1:1:size(f_source,2)
    mkdir([trainDIR,'\','Case',num2str(nn)]);
    mkdir([testDIR,'\','Case',num2str(nn)]);
end

%% Build Training Data
for ii=1:1:size(f_source,2) %loop over test/validation samples
    count = 1;
    for jj=1:1:(N_tv+N_test) %loop over specific frequency
        nharms = randi(max(harmonic_list),1); %determine number of harmonics present
        data.orders = datasample(harmonic_list,nharms,'replace',false); %determine order of harmonics present
        freqs = data.orders.*f_source(ii); %calulate frequncies of all harmonics
        data.SNR_dB = datasample(SNR_list,nharms); %determine SNR of each harmonic
        SNR = 10.^(data.SNR_dB/10);
        data.time = randn(1,length(samples)); %generate time signal with WGN
        for kk=1:1:length(SNR) %loop over present harmonics
            data.time = data.time + (SNR(kk)*sin(2*pi*(freqs(kk)/fs)*samples)); %recursively add signals to the WGN term
        end
        data.fft = fft(data.time);
        data.fft_stacked = [real(data.fft);imag(data.fft)]; %Stack real and imaginary portions of fft
        y = mat2gray(data.fft_stacked); %convert to image
        
        folder_str = ['Case',num2str(ii)];
        filename = ['Count',num2str(count)];

        if jj<=N_tv %Test/Validation Training Samples
            %save time data
            save([trainDIR,'\',folder_str,'\',filename,'.mat'],'data')
            %save freq data
            imwrite(y,[trainDIR,'\',folder_str,'\',filename,'.png']);
        else %Test Data Samples
            %save time data
            save([testDIR,'\',folder_str,'\',filename],'data')
            %save freq data
            imwrite(y,[testDIR,'\',folder_str,'\',filename,'.png']);
        end
        count = count + 1;
        clearvars nharms orders freqs SNR_dB SNR data_time data_fft y folder_str order_str SNR_str filename
    end
end

