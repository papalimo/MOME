import numpy as np
import os
# 假设你的原始时间序列数据是一个形状为 (40, 6, 100) 的numpy array
# original_data = np.random.rand(40, 6, 100)  # 示例数据

def LinearTransform(mode):
    file_path = __file__
    dir_name = os.path.dirname(file_path)
    # 获取目录名称
    dir_name_only = os.path.basename(dir_name)
    path = dir_name_only
    original_data = np.load(path + '_' + mode + '_x.npy')
    for i in range(original_data.shape[0]):
        for j in range(original_data.shape[1]):
            # 获取当前时间序列
            series = original_data[i, j, :]
            processed_data = np.empty((original_data.shape[0], original_data.shape[1], 3, original_data.shape[2]),
                                      dtype=float)
            # 计算值、值的变化和变化的变化
            values = series  # 原始值
            value_changes = np.diff(series, prepend=series[0])  # 一阶差分
            changes_of_value_changes = np.diff(value_changes, prepend=value_changes[0])  # 二阶差分

            # 存储结果
            processed_data[i, j, 0, :] = values  # 原始值
            processed_data[i, j, 1, :] = value_changes  # 一阶差分
            processed_data[i, j, 2, :] = changes_of_value_changes  # 二阶差分

    return processed_data,path
def main():
    combined_data_train,path=LinearTransform('train')
    np.save(path+'_train_x_LinearTransform.npy', combined_data_train)
    combined_data_test,path=LinearTransform('test')
    np.save(path+'_test_x_LinearTransform.npy', combined_data_test)


if __name__ == '__main__':
    main()








