import numpy as np
import os
def DFT(mode):
    file_path = __file__
    dir_name = os.path.dirname(file_path)
    dir_name_only = os.path.basename(dir_name)
    path = dir_name_only
    original_data = np.load(path + '_' + mode + '_x.npy')
    transformed_data = np.fft.fft(original_data, axis=1)
    real_part = transformed_data.real
    imaginary_part = transformed_data.imag
    combined_data = np.empty((original_data.shape[0], original_data.shape[1], 2, original_data.shape[2]))
    combined_data[:, :, 0, :] = real_part
    combined_data[:, :, 1, :] = imaginary_part  
    return combined_data,path
def main():
    combined_data_train,path=DFT('train')
    np.save(path+'_train_x_DFT.npy', combined_data_train)
    combined_data_test,path=DFT('test')
    np.save(path+'_test_x_DFT.npy', combined_data_test)



if __name__ == '__main__':
    main()

