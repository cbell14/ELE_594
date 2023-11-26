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
fs = 2^nextpow2(max(2*f_source*max(harmonic_list))); %Time Domain Sampling Rate
t_len = 1; %Length of time signal
samples = 1:1:(t_len*fs);
rng(1);

numClasses = length(f_source);
filterSize = 8;
numfilters = 8;
mEpochs = 5;
folds=5;
miniBatchSize = 48;

trainDIR = 'C:\Users\cjbell\Desktop\ELE594_Data\8Classes\TrainingData';
testDIR = 'C:\Users\cjbell\Desktop\ELE594_Data\8Classes\TestData';
outputDIR_1D = 'C:\Users\cjbell\Desktop\ELE594_Data\8Classes\1DConv_Files\';
outputDIR_2D = 'C:\Users\cjbell\Desktop\ELE594_Data\8Classes\2DConv_Files\';

%% Create Data
Data_Generation(N_tv,N_test,harmonic_list,SNR_list,f_source,fs,samples,trainDIR,testDIR)

%% Train 1DConv Network
Conv1DNetwork(N_tv,numClasses,filterSize,numfilters,mEpochs,folds,miniBatchSize,trainDIR,outputDIR_1D)

%% Test 1DConv Network
Conv1DNetwork_Testing(N_tv,N_test,folds,numClasses,miniBatchSize,SNR_list,testDIR,outputDIR_1D)

%% Train 2DConv Network
Conv2DNetwork(N_tv,numClasses,filterSize,numfilters,mEpochs,folds,miniBatchSize,trainDIR,outputDIR_2D)

%% Test 1DConv Network
Conv2DNetwork_Testing(N_tv,N_test,folds,numClasses,miniBatchSize,SNR_list,testDIR,outputDIR_2D)