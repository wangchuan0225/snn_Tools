import os
import time
import argparse
import sys
import matplotlib as plt
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import numpy as np
import requests
from tqdm import tqdm

from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer


class SNN(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)

# 下载训练以及测试文件
def download_file(url, dest):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(dest, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


def download_mnist_dataset(data_dir):
    urls = {
        "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
    }

    os.makedirs(data_dir, exist_ok=True)
    for file_name in os.listdir(data_dir):
        print(file_name)
    for filename, url in urls.items():
        dest = os.path.join(data_dir, filename)
        if not os.path.exists(dest):
            print(f"Downloading {filename} from alternative source...")
            download_file(url, dest)
        else:
            print(f"{filename} already exists, skipping download.")


def main():
# 规定好参数，包括测试的次数，数据集位置，膜电位，学习率等
    parser = argparse.ArgumentParser(description='LIF MNIST Training')
    parser.add_argument('-T', default=100, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cpu', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, default='./data', help='root dir of MNIST dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-opt', type=str, choices=['sgd', 'adam'], default='adam',
                        help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-tau', default=2.0, type=float, help='parameter tau of LIF neuron')

    args = parser.parse_args()
    print(args)

    net = SNN(tau=args.tau)

    print(net)

    net.to(args.device)

    # Manually download the MNIST dataset from an alternative source
    download_mnist_dataset(args.data_dir)

    # Initialize data loaders
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),# 将内容转化为张量
        download=False  # Set to False since we're manually downloading
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor(),
        download=False  # Set to False since we're manually downloading
    )
# 确定训练分组中每组的元素个数
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=False
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=False
    )

    start_epoch = 0
    max_test_acc = -1
    # 选择激活函数sigmod以及优化器adam
    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    encoder = encoding.PoissonEncoder()

    for epoch in range(start_epoch, 5):
        # 确定开始的时间，初始损失率为0，准确率为0
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            out_fr = 0.
            for t in range(args.T):
                encoded_img = encoder(img)
                out_fr += net(encoded_img)
            out_fr = out_fr / args.T
            loss = F.mse_loss(out_fr, label_onehot)
            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = 0.
                for t in range(args.T):
                    encoded_img = encoder(img)
                    out_fr += net(encoded_img)
                out_fr = out_fr / args.T
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(
            f'epoch = {epoch}, train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, test_loss = {test_loss:.4f}, test_acc = {test_acc:.4f}, max_test_acc = {max_test_acc:.4f}')
        print(f'train speed = {train_speed:.4f} images/s, test speed = {test_speed:.4f} images/s')
        print(
            f'Estimated completion time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

        # Save data for plotting
    net.eval()
    # Register hooks
    output_layer = net.layer[-1]  # Output layer
    output_layer.v_seq = []
    output_layer.s_seq = []

    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)

    with torch.no_grad():
        img, label = test_dataset[0]
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
        img = img.to(args.device)
        out_fr = 0.
        for t in range(args.T):
            encoded_img = encoder(img)
            out_fr += net(encoded_img)
        out_spikes_counter_frequency = (out_fr / args.T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = output_layer.v_seq.cpu().numpy().squeeze()  # v_t_array[i][j] represents the membrane potential of neuron i at time step j
        np.save("v_t_array.npy", v_t_array)
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze()  # s_t_array[i][j] represents whether neuron i fired a spike at time step j (0 or 1)
        np.save("s_t_array.npy", s_t_array)


if __name__ == '__main__':
    main()
