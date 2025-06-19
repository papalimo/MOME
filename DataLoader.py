from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch
import numpy as np
from DataProcessing import CustomDataset
import os
def DataloaderConstructing(path,batch_size,
                      shuffle=True,pin_memory=True):
    file_path = __file__
    dir_name = os.path.dirname(file_path)

    dir_name_only = os.path.basename(dir_name)
    path=dir_name_only

    time_series_data_train = np.load(path+'_train_x.npy')
    fourier_data_train = np.load(path+'_train_x_DFT.npy')  
    wavelet_data_train = np.load(path+'_train_x_wavelet.npy')  
    linear_data_train = np.load(path+'_train_x_LinearTransform.npy') 
    labels_train = np.load(path+'_train_y.npy')
    time_series_data_test = np.load(path+'_test_x.npy')
    fourier_data_test = np.load(path+'_test_x_DFT.npy')  
    wavelet_data_test = np.load(path+'_test_x_wavelet.npy')  
    linear_data_test = np.load(path+'_test_x_LinearTransform.npy') 
    labels_test = np.load(path+'_test_y.npy')
    Deal_train_dataset, Deal_test_dataset = CustomDataset(time_series_data_train, fourier_data_train, wavelet_data_train, linear_data_train, labels_train), \
                                            CustomDataset(time_series_data_test, fourier_data_test, wavelet_data_test, linear_data_test, labels_test)
    Train_loader, Test_loader = DataLoader(dataset=Deal_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           pin_memory=pin_memory,
                                           ), \
                                DataLoader(dataset=Deal_test_dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           pin_memory=pin_memory,
                                           )
    return Train_loader, Test_loader
def main():
    dataset_train,dataset_test = DataloaderConstructing(path='BasicMotions',batch_size=4,
                      shuffle=True,pin_memory=True)

    for batch in dataset_train:
        time_series, fourier, wavelet, linear, label = batch
        print("Time Series Shape:", time_series.shape)
        print("Fourier Shape:", fourier.shape)
        print("Wavelet Shape:", wavelet.shape)
        print("Linear Shape:", linear.shape)
        print("Labels Shape:", label.shape)
        break
if __name__ == "__main__":
    main()
