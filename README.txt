%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Scripts:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data_Generation.m -> 
	Description: This script generates the synthetic data for the CNNs.

	Outputs: Training and Test Data in folders labelled "TrainingData" and "TestData". Time domain data for 1D CNNs is saved in .mat 		 files and frequency domain data for 2D CNNs are save in gray scale .png files. The files are save in subfolders by lass.

Conv1DNetwork.m -> 
	Description: This script loads in the training data, builds a 1D CNN, and trains the network.

	Outputs: 1 Mat file with the maximum and minimum values from training to use to scale the test data, all trained networks from 			 cross validation, and the cross validation statistics.

Conv1DNetwork_Testing.m -> 
	Description: This script loads in the testing data, trained 1D networks, and tests the networks on the test data.

	Outputs: Plots of Confusion Matrix, Accuracy vs. Peak SNR, and Accuracy vs. Total SNR for each trained network.

Conv2DNetwork.m -> 
	Description: This script loads in the training data, builds a 2D CNN, and trains the network.

	Outputs: 1 Mat file with all trained networks from cross validation and the cross validation statistics.

Conv2DNetwork_Testing.m -> 
	Description: This script loads in the testing data, trained 2D networks, and tests the networks on the test data.

	Outputs: Plots of Confusion Matrix, Accuracy vs. Peak SNR, and Accuracy vs. Total SNR for each trained network.

Run_All.m -> 
	Description: This script runs all scripts

	Outputs: See description of outputs for each individual script.
