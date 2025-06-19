import numpy as np
import os

def LinearTransform(mode):
    file_path = __file__
    dir_name = os.path.dirname(file_path)
    dir_name_only = os.path.basename(dir_name)
    path = dir_name_only
    original_data = np.load(path + '_' + mode + '_x.npy')
    for i in range(original_data.shape[0]):
        for j in range(original_data.shape[1]):
            series = original_data[i, j, :]
            processed_data = np.empty((original_data.shape[0], original_data.shape[1], 3, original_data.shape[2]),
                                      dtype=float)

            values = series 
            value_changes = np.diff(series, prepend=series[0])  
            changes_of_value_changes = np.diff(value_changes, prepend=value_changes[0])  
            processed_data[i, j, 0, :] = values  
            processed_data[i, j, 1, :] = value_changes  
            processed_data[i, j, 2, :] = changes_of_value_changes  

    return processed_data,path
def main():
    combined_data_train,path=LinearTransform('train')
    np.save(path+'_train_x_LinearTransform.npy', combined_data_train)
    combined_data_test,path=LinearTransform('test')
    np.save(path+'_test_x_LinearTransform.npy', combined_data_test)


if __name__ == '__main__':
    main()








