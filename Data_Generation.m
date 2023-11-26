function Data_Generation(N_tv,N_test,harmonic_list,SNR_list,f_source,fs,samples,trainDIR,testDIR)
    %% Make Folders
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
end