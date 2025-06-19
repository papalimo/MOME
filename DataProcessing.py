import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
# 假设你的输入数据是这样的格式

class CustomDataset(Dataset):
    def __init__(self, time_series, fourier, wavelet, linear, labels):
        self.time_series = torch.from_numpy(time_series)
        self.fourier = torch.from_numpy(fourier)
        self.wavelet = torch.from_numpy(wavelet)
        self.linear = torch.from_numpy(linear)
        self.labels = torch.from_numpy(self.StringToLabel(labels).squeeze())

    def StringToLabel(self,y):
        labels = np.unique(y)
        new_label_list = []
        for label in y:
            for position, StringLabel in enumerate(labels):
                if label == StringLabel:
                    new_label_list.append(position)
                else:
                    continue
        new_label_list = np.array(new_label_list)
        return new_label_list
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.time_series[idx],
                self.fourier[idx],
                self.wavelet[idx],
                self.linear[idx],
                self.labels[idx])

# 创建自定义数据集
def main():
   print('heelo')

if __name__ == "__main__":
    main()

# # 测试数据加载器
# for batch in dataloader:
#     time_series, fourier, wavelet, linear, label = batch
#     print("Time Series Shape:", time_series.shape)
#     print("Fourier Shape:", fourier.shape)
#     print("Wavelet Shape:", wavelet.shape)
#     print("Linear Shape:", linear.shape)
#     print("Labels Shape:", label.shape)
#     break