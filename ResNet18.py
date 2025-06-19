import torch
import torch.nn as nn
import math
from torch.utils.tensorboard import SummaryWriter
import time
import os
from torchmetrics.classification import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score
class SimpleResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut=nn.Conv1d(in_channels,out_channels,kernel_size=1,padding=0)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 残差连接
        out = self.relu(out)
        return out

class SimpleResNetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleResNetBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut=nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 残差连接
        out = self.relu(out)
        return out

class MultiInputResNet(nn.Module):
    def __init__(self, dropout,in_channels,num_classes):
        super(MultiInputResNet, self).__init__()
        self.in_channels=in_channels
        self.num_classes=num_classes
        # 时间序列输入 (Conv1d)
        self.time_series_conv = nn.Sequential(
            SimpleResNetBlock1D(self.in_channels, 16),
            SimpleResNetBlock1D(16, 32),
            SimpleResNetBlock1D(32, 64),
            nn.AdaptiveAvgPool1d(1)  # 池化到 1
        )

        # 傅里叶变换输入 (Conv2d)
        self.fourier_conv = nn.Sequential(
            SimpleResNetBlock2D(self.in_channels, 16),
            SimpleResNetBlock2D(16, 32),
            SimpleResNetBlock2D(32, 64),
            nn.AdaptiveAvgPool2d((1, 1))  # 池化到 (1, 1)
        )

        # 小波变换输入 (Conv2d)
        self.wavelet_conv = nn.Sequential(
            SimpleResNetBlock2D(self.in_channels, 16),
            SimpleResNetBlock2D(16, 32),
            SimpleResNetBlock2D(32, 64),
            nn.AdaptiveAvgPool2d((1, 1))  # 池化到 (1, 1)
        )

        # 线性变换输入 (Conv2d)
        self.linear_conv = nn.Sequential(
            SimpleResNetBlock2D(self.in_channels, 16),
            SimpleResNetBlock2D(16, 32),
            SimpleResNetBlock2D(32, 64),
            nn.AdaptiveAvgPool2d((1, 1))  # 池化到 (1, 1)
        )

        # 最终的分类层
        self.dropout=nn.Dropout(dropout)
        self.fc = nn.Linear(64 * 4, self.num_classes)  # 32维特征 * 4个输入

    def forward(self, time_series, fourier, wavelet, linear):
        # 通过各自的网络提取特征
        time_series_features = self.time_series_conv(time_series)  # (40, 32, 1)
        fourier_features = self.fourier_conv(fourier)  # (40, 32, 2, 100)
        wavelet_features = self.wavelet_conv(wavelet)  # (40, 32, 2, 100)
        linear_features = self.linear_conv(linear)  # (40, 32, 3, 100)

        # 将特征展平并拼接
        combined_features = torch.cat((time_series_features.view(time_series_features.size(0), -1),
                                        fourier_features.view(fourier_features.size(0), -1),
                                        wavelet_features.view(wavelet_features.size(0), -1),
                                        linear_features.view(linear_features.size(0), -1)), dim=1)

        # 分类层
        output = self.fc(self.dropout(combined_features))
        return output

    def trainining(self, model, dataloader,loss_f, optim):
        self.train()
        train_accuracy = MulticlassAccuracy(self.num_classes)
        train_precision = MulticlassPrecision(self.num_classes)
        train_recall = MulticlassRecall(self.num_classes)
        train_F1 = MulticlassF1Score(self.num_classes)

        if torch.cuda.is_available():
            train_accuracy = train_accuracy.cuda()
            train_precision = train_precision.cuda()
            train_recall = train_recall.cuda()
            train_F1 = train_F1.cuda()
        running_loss = 0.0
        for batch, data in enumerate(dataloader):
            time_series, fourier, wavelet, linear, label = data
            time_series, fourier, wavelet, linear= (time_series.float(),
                                                    fourier.float(),
                                                    wavelet.float(),
                                                    linear.float())
            label=label.long()
            if torch.cuda.is_available():
                time_series, fourier, wavelet, linear, label = (time_series.cuda(),
                                                                fourier.cuda(),
                                                                wavelet.cuda(),
                                                                linear.cuda(),
                                                                label.cuda())
            y_pred = model(time_series,fourier,wavelet,linear)
            _, pred = torch.max(y_pred, 1)
            optim.zero_grad()
            loss = loss_f(y_pred, label)
            loss.backward()
            optim.step()
            train_accuracy(label, pred)
            train_precision(label, pred)
            train_recall(label, pred)
            train_F1(label, pred)
            running_loss += loss.item()
        epoch_loss = running_loss / (len(dataloader))
        epoch_accuracy = train_accuracy.compute()
        epoch_precision = train_precision.compute()
        epoch_fscore = train_recall.compute()
        epoch_recall = train_F1.compute()
        return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_fscore, optim

    def test(self, model, loss_f, dataloader):
        self.eval()
        running_loss = 0.0
        test_accuracy = MulticlassAccuracy(self.num_classes)
        test_precision = MulticlassPrecision(self.num_classes)
        test_recall = MulticlassRecall(self.num_classes)
        test_F1 = MulticlassF1Score(self.num_classes)
        if torch.cuda.is_available():
            test_accuracy = test_accuracy.cuda()
            test_precision = test_precision.cuda()
            test_recall = test_recall.cuda()
            test_F1 = test_F1.cuda()
        for batch, data in enumerate(dataloader):
            time_series, fourier, wavelet, linear, label =  data
            time_series, fourier, wavelet, linear= (time_series.float(),
                                                    fourier.float(),
                                                    wavelet.float(),
                                                    linear.float())
            label = label.long()
            if torch.cuda.is_available():
                time_series, fourier, wavelet, linear, label=(time_series.cuda(),
                                                              fourier.cuda(),
                                                              wavelet.cuda(),
                                                              linear.cuda(),
                                                              label.cuda())

            y_pred = model(time_series,fourier,wavelet,linear)
            _, pred = torch.max(y_pred, 1)
            loss = loss_f(y_pred, label)
            test_accuracy(label, pred)
            test_precision(label, pred)
            test_recall(label, pred)
            test_F1(label, pred)
            running_loss += loss.item()
        epoch_loss = running_loss / (len(dataloader))
        epoch_accuracy = test_accuracy.compute()
        epoch_precision = test_precision.compute()
        epoch_fscore = test_recall.compute()
        epoch_recall = test_F1.compute()
        return epoch_loss, epoch_accuracy, epoch_precision, epoch_recall, epoch_fscore

    def Train(self,epochs,lr,weight_decay,step_size,gamma,optim,
              SummaryName,Train_loader,Test_loader,save_checkpoint,interval):
        writer = SummaryWriter(SummaryName)
        optimizer = optim(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_function = nn.CrossEntropyLoss()
        best_acc = 0
        best_pre = 0
        best_recall = 0
        best_f1 = 0
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        for epoch in range(epochs):
            start = time.time()
            print('%d/%d' % (epoch + 1, epochs))
            print('Training')
            print('-' * 30)
            train_loss, train_accuracy, train_precision, train_recall, train_fscore, op = \
                self.trainining(model=self, optim=optimizer,
                                loss_f=loss_function, dataloader=Train_loader)
            scheduler.step()
            if save_checkpoint:
                checkpoint = {"model_state_dict": self.state_dict(),
                              "optimizer_state_dict": op.state_dict(),
                              "epoch": epoch}
                path_checkpoint = self.path + "/checkpoint_{}_epoch.pkl".format(epoch)
                torch.save(checkpoint, path_checkpoint)
            end = time.time()
            writer.add_scalar("scalar/train_loss", train_loss, epoch)
            writer.add_scalar("scalar/train_accuracy", train_accuracy, epoch)
            writer.add_scalar("scalar/train_precision", train_precision, epoch)
            writer.add_scalar("scalar/train_recall", train_recall, epoch)
            writer.add_scalar("scalar/train_fscore", train_fscore, epoch)
            writer.add_scalar('scalar/trainingtime', end - start, epoch)
            print('Training time is', end - start)
            print('Loss is ', train_loss)
            print('accuracy is', train_accuracy)
            print('precision is', train_precision)
            print('recall is', train_recall)
            print('fscore is', train_fscore)
            print('-' * 30)
            if epoch % interval == 0:
                start = time.time()
                print('Validating')
                print('-' * 30)
                Test_loss, Test_accuracy, Test_precision, Test_recall, Test_fscore = \
                    self.test(model=self,
                              loss_f=loss_function, dataloader=Test_loader)
                if Test_accuracy.item()>best_acc:
                    best_acc=Test_accuracy.item()
                if Test_precision.item()>best_pre:
                    best_pre=Test_precision.item()
                if Test_recall.item()>best_recall:
                    best_recall=Test_recall.item()
                if Test_fscore>best_f1:
                    best_f1=Test_fscore.item()
                end = time.time()
                writer.add_scalar("scalar/Test_loss", Test_loss, epoch)
                writer.add_scalar("scalar/Test_accuracy", Test_accuracy, epoch)
                writer.add_scalar("scalar/Test_precision", Test_precision, epoch)
                writer.add_scalar("scalar/Test_recall", Test_recall, epoch)
                writer.add_scalar("scalar/Test_fscore", Test_fscore, epoch)
                writer.add_scalar('scalar/Testtime', end - start, epoch)
                print('Test time is', end - start)
                print('Loss is ', Test_loss)
                print('accuracy is', Test_accuracy)
                print('precision is', Test_precision)
                print('recall is', Test_recall)
                print('fscore is', Test_fscore)
                print('-' * 30)
        print('Best acc is ', best_acc)
        print('Best pre is ', best_pre)
        print('Best recall is ', best_recall)
        print('Best fscore is ', best_f1)



# 定义超参数
def main():
    time_series=torch.rand(4,6,100)
    wavelet = torch.rand(4, 6, 2,100)
    fourier = torch.rand(4, 6, 2,100)
    linear = torch.rand(4, 6,3, 100)
    label = torch.rand(4)
    net=MultiInputResNet(6,10)
    out=net(time_series,fourier,wavelet,linear)



if __name__ == "__main__":
    main()
