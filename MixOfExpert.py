import DataLoader
import ResNet18
import argparse
import numpy
import torch
import os
def main():
    file_path = __file__
    dir_name = os.path.dirname(file_path)
    dir_name_only = os.path.basename(dir_name)
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',default=32)
    parser.add_argument('--path', default=dir_name_only)
    parser.add_argument('--inplanes',default=2)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--num_class',default=3)
    parser.add_argument('--dropout',default=0.4)
    parser.add_argument('--step_size', default=10)
    parser.add_argument('--gamma', default=0.8)
    parser.add_argument('--lr', default=0.1)
    parser.add_argument('--interval', default=1)
    parser.add_argument('--weight_decay', default=0)
    parser.add_argument('--save_checkpoint', default=False)
    parser.add_argument('--SummaryName', default=dir_name_only)
    parser.add_argument('--MultiGPU', default=True)
    parser.add_argument('--GPUiD', default=[0,1,2])

    args = parser.parse_args()
    dataset_train, dataset_test = DataLoader.DataloaderConstructing(path=args.path, batch_size=args.batch_size,
                                                         shuffle=True, pin_memory=True)
    model=ResNet18.MultiInputResNet(dropout=args.dropout,in_channels=args.inplanes,num_classes=args.num_class)
    if torch.cuda.is_available():
        model=model.cuda()

    print('Training preparation finished! Start to train! Good Luck!:)')
    model.Train(epochs=args.epochs,lr=args.lr,weight_decay=args.weight_decay,
                step_size=args.step_size,
                gamma=args.gamma,optim=torch.optim.Adadelta,SummaryName=args.SummaryName,
                Train_loader=dataset_train,Test_loader=dataset_test,
                save_checkpoint=args.save_checkpoint,
                interval=args.interval)


if __name__ == '__main__':
    main()
