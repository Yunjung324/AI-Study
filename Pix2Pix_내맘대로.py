import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt


# Load Dataset from ImageFolder
class Dataset(data.Dataset):  # torch 기본 Dataset 상속 받기
    def __init__(self, image_dir, direction):
        super(Dataset, self).__init__()  # 초기화 상속
        self.direction = direction  #
        self.a_path = os.path.join(image_dir, "a")  # a는 건물 사진
        self.b_path = os.path.join(image_dir, "b")  # b는 Segmentation Mask
        self.image_filenames = [x for x in os.listdir(self.a_path)]  # a 폴더에 있는 파일 목록
        self.seg_filenames = [x for x in os.listdir(self.b_path)]   # b 폴더에 있는 파일 목록
        self.transform = transforms.Compose([transforms.Resize((256, 256)),  # 이미지 크기 조정
                                             transforms.ToTensor(),  # Numpy -> Tensor
                                             transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                  std=(0.5, 0.5, 0.5))  # Normalization : -1 ~ 1 range
                                             ])
        self.len = len(self.image_filenames)

    def __getitem__(self, index):
        # 건물 사진과 Segmentation mask를 각각 a,b 폴더에서 불러 오기
        # PIL에서는 이미지 처리를 수행할 때 색상 순서가 BGR이므로 RGB로 변환해준다
        a = Image.open(os.path.join(self.a_path, self.image_filenames[index])).convert('RGB')  # 건물 사진
        b = Image.open(os.path.join(self.b_path, self.seg_filenames[index])).convert('RGB')  # Segmentation 사진

        # 이미지 전처리
        a = self.transform(a)
        b = self.transform(b)

        if self.direction == "a2b":  # 건물 -> Segmentation
            return a, b
        else:  # Segmentation -> 건물
            return b, a

    def __len__(self):
        return self.len


train_dataset = Dataset("C:/Users/Synapse/Downloads/train", "b2a")
test_dataset = Dataset("C:/Users/Synapse/Downloads/test", "b2a")

train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=1)  # Shuffle
test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=1)


# -1 ~ 1사이의 값을 0~1사이로 만들어 준다
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# 이미지 시각화 함수
def show_images(real_a, real_b, fake_b):
    plt.figure(figsize=(30, 90))
    plt.subplot(131)  # 여러 개의 그래프를 하나의 그림에 나타내도록 함, 축 공유, 각 정수를 이어서 3자리 수의 정수로 입력 받는 방식
    plt.imshow(real_a.cpu().data.numpy().transpose(1, 2, 0))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(132)
    plt.imshow(real_b.cpu().data.numpy().transpose(1, 2, 0))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(133)
    plt.imshow(fake_b.cpu().data.numpy().transpose(1, 2, 0))
    plt.xticks([])
    plt.yticks([])

    plt.show()


# Conv -> Batchnorm -> Activate function Layer
'''
코드 단순화를 위한 convolution block 생성을 위한 함수
Encoder에서 사용될 예정
'''
def maxout(inputs, num_units, axis=None, tf=None):
    shape = inputs.get_shape().as_list()
    if axis is None:
        # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a multiple of num_units({})'
             .format(num_channels, num_units))
    shape[axis] = -1
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, activation='ReLU'):
    layers = []

    # Conv layer
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))

    # Batch Normalization
    if bn:
        layers.append(nn.BatchNorm2d(c_out))

    # Activation
    if activation == 'LeakyReLU':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'ReLU':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'maxout':
        layers.append(nn.maxout())
    elif activation == 'none':
        pass

    return nn.Sequential(*layers)


# Deconv -> BatchNorm -> Activate function Layer
'''
코드 단순화를 위한 convolution block 생성을 위한 함수
Decoder에서 이미지 복원을 위해 사용될 예정
'''


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True, activation='LeakyReLU'):
    layers = []

    # Deconv.
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))

    # Batchnorm
    if bn:
        layers.append(nn.BatchNorm2d(c_out))

    # Activation
    if activation == 'LeakyReLU':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'ReLU':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'max_out':
        layers.append(nn.maxout())
    elif activation == 'none':
        pass

    return nn.Sequential(*layers)


class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        # Unet encoder
        self.conv1 = conv(3, 64, 4, bn=False, activation='LeakyReLU')  # (B, 64, 128, 128)
        self.conv2 = conv(64, 128, 4, activation='LeakyReLU')  # (B, 128, 64, 64)
        self.conv3 = conv(128, 256, 4, activation='LeakyReLU')  # (B, 256, 32, 32)
        self.conv4 = conv(256, 512, 4, activation='LeakyReLU')  # (B, 512, 16, 16)
        self.conv5 = conv(512, 512, 4, activation='LeakyReLU')  # (B, 512, 8, 8)
        self.conv6 = conv(512, 512, 4, activation='LeakyReLU')  # (B, 512, 4, 4)
        self.conv7 = conv(512, 512, 4, activation='LeakyReLU')  # (B, 512, 2, 2)
        self.conv8 = conv(512, 512, 4, bn=False, activation='ReLU')  # (B, 512, 1, 1)


        # Unet decoder
        self.deconv1 = deconv(512, 512, 4, activation='ReLU')  # (B, 512, 2, 2)
        self.deconv2 = deconv(1024, 512, 4, activation='ReLU')  # (B, 512, 4, 4)
        self.deconv3 = deconv(1024, 512, 4, activation='ReLU')  # (B, 512, 8, 8) # Hint : U-Net에서는 Encoder에서 넘어온 Feature를 Concat합니다! (Channel이 2배)
        self.deconv4 = deconv(1024, 512, 4, activation='ReLU')  # (B, 512, 16, 16)
        self.deconv5 = deconv(1024, 256, 4, activation='ReLU')  # (B, 256, 32, 32)
        self.deconv6 = deconv(512, 128, 4, activation='ReLU')  # (B, 128, 64, 64)
        self.deconv7 = deconv(256, 64, 4, activation='ReLU')  # (B, 64, 128, 128)
        self.deconv8 = deconv(128, 3, 4, activation='tanh')  # (B, 3, 256, 256)

    # forward method
    def forward(self, input):
        # Unet encoder
        e1 = self.conv1(input)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)
        e8 = self.conv8(e7)

        # Unet decoder
        d1 = F.dropout(self.deconv1(e8), 0.5, training=True)
        d2 = F.dropout(self.deconv2(torch.cat([d1, e7], 1)), 0.5, training=True)
        d3 = F.dropout(self.deconv3(torch.cat([d2, e6], 1)), 0.5, training=True)
        d4 = self.deconv4(torch.cat([d3, e5], 1))
        d5 = self.deconv5(torch.cat([d4, e4], 1))
        d6 = self.deconv6(torch.cat([d5, e3], 1))
        d7 = self.deconv7(torch.cat([d6, e2], 1))
        output = self.deconv8(torch.cat([d7, e1], 1))

        return output


class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = conv(6, 64, 4, bn=False, activation=maxout)
        self.conv2 = conv(64, 128, 4, activation=maxout)
        self.conv3 = conv(128, 256, 4, activation=maxout)
        self.conv4 = conv(256, 512, 4, 1, 1, activation=maxout)
        self.conv5 = conv(512, 1, 4, 1, 1, activation='none')

    # forward method
    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out


# Generator와 Discriminator를 GPU로 보내기
G = Generator().cuda()
D = Discriminator().cuda()

criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()

# Setup optimizer
g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Train
for epoch in range(1, 500):
    for i, (real_a, real_b) in enumerate(train_loader, 1):
        # forward
        real_a, real_b = real_a.cuda(), real_b.cuda()
        real_label = torch.ones(real_a.size(0), 1, 30, 30).cuda()
        fake_label = torch.zeros(real_b.size(0), 1, 30, 30).cuda()

        fake_b = G(real_a)  # G가 생성한 fake Segmentation mask

        # ============= Train the discriminator =============#
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = D.forward(fake_ab.detach())
        loss_d_fake = criterionMSE(pred_fake, fake_label)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = D.forward(real_ab)
        loss_d_real = criterionMSE(pred_real, real_label)

        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        # Backprop + Optimize
        D.zero_grad()
        loss_d.backward()
        d_optimizer.step()

        # =============== Train the generator ===============#
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = D.forward(fake_ab)
        loss_g_gan = criterionMSE(pred_fake, real_label)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * 10

        loss_g = loss_g_gan + loss_g_l1

        # Backprop + Optimize
        G.zero_grad()
        D.zero_grad()
        loss_g.backward()
        g_optimizer.step()

    print('Epoch [%d/%d] d_loss: %.4f, g_loss: %.4f'
          % (epoch, 500, loss_d.item(), loss_g.item()))
    print(
        '======================================================================================================')
    if epoch % 10 == 0:
        show_images(denorm(real_a.squeeze()), denorm(real_b.squeeze()), denorm(fake_b.squeeze()))