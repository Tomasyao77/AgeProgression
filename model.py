from utils import *
import consts

import logging
import random
from collections import OrderedDict
import cv2
import imageio
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.functional import l1_loss, mse_loss
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits_loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import smtp
import time
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        num_conv_layers = 6

        self.conv_layers = nn.ModuleList()

        def add_conv(module_list, name, in_ch, out_ch, kernel, stride, padding, act_fn):
            return module_list.add_module(
                name,
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride,
                    ),
                    act_fn
                )
            )

        add_conv(self.conv_layers, 'e_conv_1', in_ch=3, out_ch=64, kernel=5, stride=2, padding=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_2', in_ch=64, out_ch=128, kernel=5, stride=2, padding=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_3', in_ch=128, out_ch=256, kernel=5, stride=2, padding=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_4', in_ch=256, out_ch=512, kernel=5, stride=2, padding=2, act_fn=nn.ReLU())
        add_conv(self.conv_layers, 'e_conv_5', in_ch=512, out_ch=1024, kernel=5, stride=2, padding=2, act_fn=nn.ReLU())

        self.fc_layer = nn.Sequential(
            OrderedDict(
                [
                    ('e_fc_1', nn.Linear(in_features=1024, out_features=consts.NUM_Z_CHANNELS)),  # 100 50
                    ('tanh_1', nn.Tanh())  # normalize to [-1, 1] range
                ]
            )
        )

    def forward(self, face):
        out = face
        # print(out.shape)
        for conv_layer in self.conv_layers:
            # print("H")
            # 用for循环可还行
            out = conv_layer(out)
            # print(out.shape)
            # print("W")
        # print(out.shape)
        out = out.flatten(1, -1)  # flatten函数返回的是拷贝 卷积层展平后连接全连接层
        # print(out.shape)
        out = self.fc_layer(out)
        return out


class DiscriminatorZ(nn.Module):  # 就是4个全连接层
    def __init__(self):
        super(DiscriminatorZ, self).__init__()
        # (100, 64, 32, 16)
        dims = (consts.NUM_Z_CHANNELS, consts.NUM_ENCODER_CHANNELS, consts.NUM_ENCODER_CHANNELS // 2,
                consts.NUM_ENCODER_CHANNELS // 4)
        self.layers = nn.ModuleList()
        # in_dim: 100, 64, 32
        # out_dim: 64, 32, 16
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]), 1):
            self.layers.add_module(
                'dz_fc_%d' % i,
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                )
            )

        self.layers.add_module(
            'dz_fc_%d' % (i + 1),  # i是上面for循环的 这里也可以访问到!
            nn.Sequential(
                nn.Linear(out_dim, 1),
                # nn.Sigmoid()  # commented out because logits are needed
            )
        )

    def forward(self, z):
        out = z
        # print(out.shape)
        for layer in self.layers:
            out = layer(out)
            # print(out.shape)
        return out


class DiscriminatorImg(nn.Module):
    def __init__(self):
        super(DiscriminatorImg, self).__init__()
        in_dims = (3, 16 + consts.LABEL_LEN_EXPANDED, 32, 64)
        out_dims = (16, 32, 64, 128)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        # 3, 36, 32, 64
        # 16, 32, 64, 128
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims), 1):
            self.conv_layers.add_module(
                'dimg_conv_%d' % i,
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU()
                )
            )

        self.fc_layers.add_module(
            'dimg_fc_1',
            nn.Sequential(
                # 8192 1024
                nn.Linear(128 * 8 * 8, 1024),
                nn.LeakyReLU()
            )
        )

        self.fc_layers.add_module(
            'dimg_fc_2',
            nn.Sequential(
                nn.Linear(1024, 1),
                # nn.Sigmoid()  # commented out because logits are needed
            )
        )

    def forward(self, imgs, labels, device):  # labels: [batch_size, 20]
        out = imgs
        # print("输入", out.shape)
        # run convs
        for i, conv_layer in enumerate(self.conv_layers, 1):
            # print(out.shape)
            # print(conv_layer)
            out = conv_layer(out)
            # print(out.shape)
            if i == 1:
                # concat labels after first conv
                # 这一段很重要 很难
                labels_tensor = torch.zeros(torch.Size((out.size(0), labels.size(1), out.size(2), out.size(3))),
                                            device=device)
                for img_idx in range(out.size(0)):
                    for label in range(labels.size(1)):
                        labels_tensor[img_idx, label, :, :] = labels[img_idx, label]  # fill a square
                out = torch.cat((out, labels_tensor), 1)
                # print("连接标签后", out.shape)

        # run fcs
        out = out.flatten(1, -1)
        # print("flatten后",out.shape)
        for fc_layer in self.fc_layers:
            out = fc_layer(out)
            # print(out.shape)

        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 反卷积层
        num_deconv_layers = 5
        mini_size = 4
        self.fc = nn.Sequential(
            # 100+20 1024*16
            nn.Linear(
                consts.NUM_Z_CHANNELS + consts.LABEL_LEN_EXPANDED,
                consts.NUM_GEN_CHANNELS * mini_size ** 2
            ),
            nn.ReLU()
        )
        # need to reshape now to ?,1024,8,8

        self.deconv_layers = nn.ModuleList()

        def add_deconv(name, in_dims, out_dims, kernel, stride, actf):
            self.deconv_layers.add_module(
                name,
                nn.Sequential(
                    easy_deconv(
                        in_dims=in_dims,
                        out_dims=out_dims,
                        kernel=kernel,
                        stride=stride,
                    ),
                    actf
                )
            )

        add_deconv('g_deconv_1', in_dims=(1024, 4, 4), out_dims=(512, 8, 8), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_2', in_dims=(512, 8, 8), out_dims=(256, 16, 16), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_3', in_dims=(256, 16, 16), out_dims=(128, 32, 32), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_4', in_dims=(128, 32, 32), out_dims=(64, 64, 64), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_5', in_dims=(64, 64, 64), out_dims=(32, 128, 128), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_6', in_dims=(32, 128, 128), out_dims=(16, 128, 128), kernel=5, stride=1, actf=nn.ReLU())
        add_deconv('g_deconv_7', in_dims=(16, 128, 128), out_dims=(3, 128, 128), kernel=1, stride=1, actf=nn.Tanh())

    def _decompress(self, x):
        return x.view(x.size(0), 1024, 4, 4)  # TODO - replace hardcoded

    def forward(self, z, age=None, gender=None):
        out = z
        if age is not None and gender is not None:
            label = Label(age, gender).to_tensor() \
                if (isinstance(age, int) and isinstance(gender, int)) \
                else torch.cat((age, gender), 1)
            out = torch.cat((out, label), 1)  # z_l
        # print(out.shape)  # [batch_size, 120]
        out = self.fc(out)
        # print(out.shape) # [batch_size, 16384]
        out = self._decompress(out)
        # print(out.shape) # [batch_size, 1024, 4, 4] 全连接层变成卷积层
        for i, deconv_layer in enumerate(self.deconv_layers, 1):
            out = deconv_layer(out)
            # print(out.shape)
        return out


class Net(object):
    def __init__(self):
        self.E = Encoder()
        self.Dz = DiscriminatorZ()
        self.Dimg = DiscriminatorImg()
        self.G = Generator()

        self.eg_optimizer = Adam(list(self.E.parameters()) + list(self.G.parameters()))
        self.dz_optimizer = Adam(self.Dz.parameters())
        self.di_optimizer = Adam(self.Dimg.parameters())

        self.device = None
        self.cpu()  # initial, can later move to cuda

    def __call__(self, *args, **kwargs):
        self.test_single(*args, **kwargs)

    def __repr__(self):
        return os.linesep.join([repr(subnet) for subnet in (self.E, self.Dz, self.G)])

    def morph(self, image_tensors, ages, genders, length, target):

        self.eval()

        original_vectors = [None, None]
        for i in range(2):
            # 这维度操作太6了
            z = self.E(image_tensors[i].unsqueeze(0))
            l = Label(ages[i], genders[i]).to_tensor(normalize=True).unsqueeze(0).to(device=z.device)
            z_l = torch.cat((z, l), 1)
            original_vectors[i] = z_l

        z_vectors = torch.zeros((length + 1, z_l.size(1)), dtype=z_l.dtype)
        for i in range(length + 1):
            z_vectors[i, :] = original_vectors[0].mul(length - i).div(length) + original_vectors[1].mul(i).div(length)

        generated = self.G(z_vectors)
        dest = os.path.join(target, 'morph.png')
        save_image_normalized(tensor=generated, filename=dest, nrow=generated.size(0))
        print_timestamp("Saved test result to " + dest)
        return dest

    def kids(self, image_tensors, length, target):

        self.eval()

        original_vectors = [None, None]
        for i in range(2):
            z = self.E(image_tensors[i].unsqueeze(0)).squeeze(0)
            original_vectors[i] = z

        z_vectors = torch.zeros((length, consts.NUM_Z_CHANNELS), dtype=z.dtype)
        z_l_vectors = torch.zeros((length, consts.NUM_Z_CHANNELS + consts.LABEL_LEN_EXPANDED), dtype=z.dtype)
        for i in range(length):
            for j in range(consts.NUM_Z_CHANNELS):
                r = random.random()
                z_vectors[i][j] = original_vectors[0][j].mul(r) + original_vectors[1][j].mul(1 - r)

            fake_age = 0
            fake_gender = random.choice([consts.MALE, consts.FEMALE])
            l = Label(fake_age, fake_gender).to_tensor(normalize=True).to(device=z.device)
            z_l = torch.cat((z_vectors[i], l), 0)
            z_l_vectors[i, :] = z_l

        generated = self.G(z_l_vectors)
        dest = os.path.join(target, 'kids.png')
        save_image_normalized(tensor=generated, filename=dest, nrow=generated.size(0))
        print_timestamp("Saved test result to " + dest)
        return dest

    def test_single(self, image_tensor, age, gender, target, watermark, load=None, img_name=None):

        self.eval()
        # image_tensor [1, 3, 128, 128] repeat -> [10, 3, 128, 128]
        batch = image_tensor.repeat(consts.NUM_AGES, 1, 1, 1).to(device=self.device)  # N x D x H x W
        # [10, 100]
        z = self.E(batch)  # N x Z

        gender_tensor = -torch.ones(consts.NUM_GENDERS)
        gender_tensor[int(gender)] *= -1
        # gender_tensor [2] repeat -> [10, 10] 牛皮
        gender_tensor = gender_tensor.repeat(consts.NUM_AGES,
                                             consts.NUM_AGES // consts.NUM_GENDERS)  # apply gender on all images

        # age_tensor [10, 10]
        age_tensor = -torch.ones(consts.NUM_AGES, consts.NUM_AGES)
        for i in range(consts.NUM_AGES): # 10 厉害了
            age_tensor[i][i] *= -1  # apply the i'th age group on the i'th image

        # cat((10, 10), (10, 10)) -> [10, 20]
        l = torch.cat((age_tensor, gender_tensor), 1).to(self.device)
        # cat((10, 100), (10, 20)) -> [10, 120]
        z_l = torch.cat((z, l), 1)

        # [10, 3, 128, 128]
        generated = self.G(z_l)

        if watermark:
            image_tensor = image_tensor.permute(1, 2, 0)
            image_tensor = 255 * one_sided(image_tensor.cpu().numpy())
            image_tensor = np.ascontiguousarray(image_tensor, dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (2, 25)
            fontScale = 0.5
            fontColor = (0, 128, 0)  # dark green, should be visible on most skin colors
            lineType = 2
            cv2.putText(
                image_tensor,
                '{}, {}'.format(["Male", "Female"][gender], age),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,

            )
            image_tensor = two_sided(torch.from_numpy(image_tensor / 255.0)).float().permute(2, 0, 1)

        joined = torch.cat((image_tensor.unsqueeze(0), generated.cpu()), 0)

        joined = nn.ConstantPad2d(padding=4, value=-1)(joined)
        for img_idx in (0, Label.age_transform(age) + 1):
            for elem_idx in (0, 1, 2, 3, -4, -3, -2, -1):
                joined[img_idx, :, elem_idx, :] = 1  # color border white
                joined[img_idx, :, :, elem_idx] = 1  # color border white

        dest = os.path.join(target, load[load.rindex("/") + 1:] + "_" + img_name + '.png')
        save_image_normalized(tensor=joined, filename=dest, nrow=joined.size(0))
        print_timestamp("Saved test result to " + dest)
        return dest

    def teach(
            self,
            utkface_path,
            batch_size=128,
            epochs=1,
            weight_decay=1e-5,
            lr=2e-4,
            should_plot=False,
            betas=(0.9, 0.999),
            valid_size=None,
            where_to_save=None,
            models_saving='always',
            reg_loss_ratio=0.0
    ):
        print("开始训练时间：")
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(start_time)
        where_to_save = where_to_save or default_where_to_save()
        dataset = get_utkface_dataset(utkface_path)
        valid_size = valid_size or batch_size
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, (len(dataset) - valid_size, valid_size))

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}  # 骚操作
        # print(dataset.class_to_idx)
        # # {0: '0.0', 1: '0.1', 2: '1.0', 3: '1.1', 4: '2.0', 5: '2.1', 6: '3.0',
        # #  7: '3.1', 8: '4.0', 9: '4.1', 10: '5.0', 11: '5.1', 12: '6.0', 13: '6.1',
        # #  14: '7.0', 15: '7.1', 16: '8.0', 17: '8.1', 18: '9.0', 19: '9.1'}
        # print(idx_to_class)
        # exit(1)

        input_output_loss = l1_loss
        nrow = round((2 * batch_size) ** 0.5)  # 我知道了 原图与生成图合并图的矩阵维度 16*16=256

        # save_image_normalized(tensor=validate_images, filename=where_to_save+"/base.png")

        # 超参数赋值 有点6
        for optimizer in (self.eg_optimizer, self.dz_optimizer, self.di_optimizer):
            for param in ('weight_decay', 'betas', 'lr'):
                val = locals()[param]
                if val is not None:
                    optimizer.param_groups[0][param] = val

        loss_tracker = LossTracker(plot=should_plot)
        where_to_save_epoch = ""
        save_count = 0
        paths_for_gif = []

        # 模型加速
        device_ = "cuda" if torch.cuda.is_available() else "cpu"
        if device_ == "cuda":
            cudnn.benchmark = True
        # tensorboard记录
        train_eg_writer = SummaryWriter(log_dir=consts.TF_LOG + "/" + "_train_eg")
        train_reg_writer = SummaryWriter(log_dir=consts.TF_LOG + "/" + "_train_reg")
        train_dz_writer = SummaryWriter(log_dir=consts.TF_LOG + "/" + "_train_dz")
        train_ez_writer = SummaryWriter(log_dir=consts.TF_LOG + "/" + "_train_ez")
        train_di_writer = SummaryWriter(log_dir=consts.TF_LOG + "/" + "_train_di")
        train_dg_writer = SummaryWriter(log_dir=consts.TF_LOG + "/" + "_train_dg")
        val_writer = SummaryWriter(log_dir=consts.TF_LOG + "/" + "_val")

        for epoch in range(1, epochs + 1):
            where_to_save_epoch = os.path.join(where_to_save, "epoch" + str(epoch))
            try:
                if not os.path.exists(where_to_save_epoch):
                    os.makedirs(where_to_save_epoch)
                paths_for_gif.append(where_to_save_epoch)
                losses = defaultdict(lambda: [])
                self.train()  # move to train mode
                with tqdm(train_loader) as _tqdm:
                    for i, (images, labels) in enumerate(_tqdm, 1):
                        # for i, (images, labels) in tqdm(enumerate(train_loader, 1)):
                        images = images.to(device=self.device)
                        # stack(默认dim=0)就是把多个一维向量(这里)堆叠起来成为二维数组(batch)
                        # 年龄和性别标签
                        labels = torch.stack([str_to_tensor(idx_to_class[l], normalize=True) for l in
                                              list(labels.numpy())])  # todo - can remove list() ?
                        # print(labels.shape)  # [128, 20]
                        # exit(1)
                        labels = labels.to(device=self.device)
                        # print ("iteration: "+str(i)+" images shape: "+str(images.shape)) # [128, 3, 128, 128]
                        # exit(1)
                        z = self.E(images)
                        # print(z.shape) # [128, 100]
                        # exit(1)

                        # Input\Output Loss
                        z_l = torch.cat((z, labels), 1)
                        # print(z_l.shape)  # [128, 120]
                        # exit(1)
                        generated = self.G(z_l)
                        # print(generated.shape)  # [128, 3, 128, 128]
                        # exit(1)
                        # eg_loss = input_output_loss(generated, images)  # 顺序是不是反了 --鄙人疑虑?
                        eg_loss = input_output_loss(images, generated)  # input target
                        losses['eg'].append(eg_loss.item())

                        # Total Variance Regularization Loss 全变分损失，可以使图像更平滑
                        reg = l1_loss(generated[:, :, :, :-1], generated[:, :, :, 1:]) + l1_loss(
                            generated[:, :, :-1, :],
                            generated[:, :, 1:, :])

                        # reg = (
                        #        torch.sum(torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:])) +
                        #        torch.sum(torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :]))
                        # ) / float(generated.size(0))
                        reg_loss = reg_loss_ratio * reg  # 乘以0干嘛?是暂时不用吗
                        reg_loss.to(self.device)
                        losses['reg'].append(reg_loss.item())

                        # DiscriminatorZ Loss
                        # 这个损失不好理解!?
                        # torch.rand_like返回跟input的tensor一样size的0-1随机数
                        z_prior = two_sided(torch.rand_like(z, device=self.device))  # [-1 : 1]
                        d_z_prior = self.Dz(z_prior)
                        d_z = self.Dz(z)
                        # bce_with_logits_loss 二分类交叉熵计算
                        dz_loss_prior = bce_with_logits_loss(d_z_prior, torch.ones_like(d_z_prior))
                        dz_loss = bce_with_logits_loss(d_z, torch.zeros_like(d_z))
                        dz_loss_tot = (dz_loss + dz_loss_prior)
                        losses['dz'].append(dz_loss_tot.item())

                        # Encoder\DiscriminatorZ Loss
                        # 为什么这就是ez_loss呢?我看曲线图就是一条0线 懂了z就是E生成的
                        ez_loss = 0.0001 * bce_with_logits_loss(d_z, torch.ones_like(d_z))
                        ez_loss.to(self.device)
                        losses['ez'].append(ez_loss.item())

                        # DiscriminatorImg Loss
                        d_i_input = self.Dimg(images, labels, self.device)
                        d_i_output = self.Dimg(generated, labels, self.device)

                        di_input_loss = bce_with_logits_loss(d_i_input, torch.ones_like(d_i_input))
                        di_output_loss = bce_with_logits_loss(d_i_output, torch.zeros_like(d_i_output))
                        di_loss_tot = (di_input_loss + di_output_loss)
                        losses['di'].append(di_loss_tot.item())

                        # Generator\DiscriminatorImg Loss
                        # d_i_output就是Dimg(generated)生成的
                        dg_loss = 0.0001 * bce_with_logits_loss(d_i_output, torch.ones_like(d_i_output))
                        losses['dg'].append(dg_loss.item())

                        # this loss is only for debugging
                        uni_diff_loss = (uni_loss(z.cpu().detach()) - uni_loss(z_prior.cpu().detach())) / batch_size
                        # losses['uni_diff'].append(uni_diff_loss)


                        # Start back propagation

                        # Back prop on Encoder\Generator
                        self.eg_optimizer.zero_grad()
                        loss = eg_loss + reg_loss + ez_loss + dg_loss
                        loss.backward(retain_graph=True)
                        self.eg_optimizer.step()

                        # Back prop on DiscriminatorZ
                        self.dz_optimizer.zero_grad()
                        dz_loss_tot.backward(retain_graph=True)
                        self.dz_optimizer.step()

                        # Back prop on DiscriminatorImg
                        self.di_optimizer.zero_grad()
                        di_loss_tot.backward()
                        self.di_optimizer.step()

                        now = datetime.datetime.now()
                        # record loss
                        sample_num = images.size(0)
                        _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss.item()),
                                          sample_num=sample_num)

                # 记录train tensorboard
                train_eg_writer.add_scalar("loss", np.array(losses["eg"]).mean(), epoch)
                train_reg_writer.add_scalar("loss", np.array(losses["reg"]).mean(), epoch)
                train_dz_writer.add_scalar("loss", np.array(losses["dz"]).mean(), epoch)
                train_ez_writer.add_scalar("loss", np.array(losses["ez"]).mean(), epoch)
                train_di_writer.add_scalar("loss", np.array(losses["di"]).mean(), epoch)
                train_dg_writer.add_scalar("loss", np.array(losses["dg"]).mean(), epoch)

                logging.info('[{h}:{m}[Epoch {e}] Loss: {t}'.format(h=now.hour, m=now.minute, e=epoch, t=loss.item()))
                print_timestamp(f"[Epoch {epoch:d}] Loss: {loss.item():f}")
                to_save_models = models_saving in ('always', 'tail')
                cp_path = self.save(where_to_save_epoch, to_save_models=to_save_models)
                if models_saving == 'tail':
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, 'losses.png'))

                with torch.no_grad():  # validation
                    self.eval()  # move to eval mode

                    with tqdm(valid_loader) as _tqdm_val:
                        for ii, (images, labels) in enumerate(_tqdm_val, 1):
                            # for ii, (images, labels) in enumerate(valid_loader, 1):
                            images = images.to(self.device)
                            labels = torch.stack(
                                [str_to_tensor(idx_to_class[l], normalize=True) for l in list(labels.numpy())])
                            labels = labels.to(self.device)
                            validate_labels = labels.to(self.device)

                            z = self.E(images)
                            z_l = torch.cat((z, validate_labels), 1)
                            generated = self.G(z_l)

                            loss = input_output_loss(images, generated)

                            # merge_images [128, 3, 128, 128] -> [128*2, 3, 128, 128]
                            joined = merge_images(images, generated)  # torch.cat((generated, images), 0)

                            file_name = os.path.join(where_to_save_epoch, 'validation.png')
                            # save_image_normalized 保存成一张图原图右边接着生成图 调用的是第三方函数save_image
                            save_image_normalized(tensor=joined, filename=file_name, nrow=nrow)

                            losses['valid'].append(loss.item())
                            # record loss
                            sample_num = images.size(0)
                            _tqdm_val.set_postfix(OrderedDict(stage="validate", epoch=epoch, loss=loss.item()),
                                                  sample_num=sample_num)
                            # break #感觉不需要break batchsize和valsize大小一样的
                            # 记录val tensorboard
                            val_writer.add_scalar("loss", np.array(losses["valid"]).mean(), epoch)

                loss_tracker.append_many(**{k: mean(v) for k, v in losses.items()})
                loss_tracker.plot()

                logging.info(
                    '[{h}:{m}[Epoch {e}] Loss: {l}'.format(h=now.hour, m=now.minute, e=epoch, l=repr(loss_tracker)))

            except KeyboardInterrupt:
                print_timestamp("{br}CTRL+C detected, saving model{br}".format(br=os.linesep))
                if models_saving != 'never':
                    cp_path = self.save(where_to_save_epoch, to_save_models=True)
                if models_saving == 'tail':
                    prev_folder = os.path.join(where_to_save, "epoch" + str(epoch - 1))
                    remove_trained(prev_folder)
                loss_tracker.save(os.path.join(cp_path, 'losses.png'))
                raise

        if models_saving == 'last':
            cp_path = self.save(where_to_save_epoch, to_save_models=True)
        loss_tracker.plot()

        # 代码训练完毕
        print("结束训练时间: ")
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(end_time)
        print("训练耗时: " + smtp.date_gap(start_time, end_time))
        smtp.main(dict_={"epochs: ": epochs,
                         "project: ": "AgeProgression",
                         "训练耗时: ": smtp.date_gap(start_time, end_time)})

    def _mass_fn(self, fn_name, *args, **kwargs):
        """Apply a function to all possible Net's components.

        :return:
        """

        for class_attr in dir(self):
            if not class_attr.startswith('_'):  # ignore private members, for example self.__class__
                class_attr = getattr(self, class_attr)
                if hasattr(class_attr, fn_name):
                    fn = getattr(class_attr, fn_name)
                    fn(*args, **kwargs)

    def to(self, device):
        self._mass_fn('to', device=device)

    def cpu(self):
        self._mass_fn('cpu')
        self.device = torch.device('cpu')

    def cuda(self):
        self._mass_fn('cuda')
        self.device = torch.device('cuda')

    def eval(self):
        """Move Net to evaluation mode.

        :return:
        """
        self._mass_fn('eval')

    def train(self):
        """Move Net to training mode.

        :return:
        """
        self._mass_fn('train')

    def save(self, path, to_save_models=True):
        """Save all state dicts of Net's components.

        :return:
        """
        if not os.path.isdir(path):
            os.mkdir(path)
        # path = os.path.join(path, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        if not os.path.isdir(path):
            os.mkdir(path)

        saved = []
        if to_save_models:
            for class_attr_name in dir(self):
                if not class_attr_name.startswith('_'):
                    class_attr = getattr(self, class_attr_name)
                    if hasattr(class_attr, 'state_dict'):
                        state_dict = class_attr.state_dict
                        fname = os.path.join(path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name))
                        torch.save(state_dict, fname)
                        saved.append(class_attr_name)

        if saved:
            print_timestamp("Saved {} to {}".format(', '.join(saved), path))
        elif to_save_models:
            raise FileNotFoundError("Nothing was saved to {}".format(path))
        return path

    def load(self, path, slim=True):
        """Load all state dicts of Net's components.

        :return:
        """
        loaded = []
        for class_attr_name in dir(self):
            if (not class_attr_name.startswith('_')) and ((not slim) or (class_attr_name in ('E', 'G'))):
                class_attr = getattr(self, class_attr_name)
                fname = os.path.join(path, consts.TRAINED_MODEL_FORMAT.format(class_attr_name))
                if hasattr(class_attr, 'load_state_dict') and os.path.exists(fname):
                    class_attr.load_state_dict(torch.load(fname)())
                    loaded.append(class_attr_name)
        if loaded:
            print_timestamp("Loaded {} from {}".format(', '.join(loaded), path))
        else:
            raise FileNotFoundError("Nothing was loaded from {}".format(path))


def create_list_of_img_paths(pattern, start, step):
    result = []
    fname = pattern.format(start)
    while os.path.isfile(fname):
        result.append(fname)
        start += step
        fname = pattern.format(start)
    return result


def create_gif(img_paths, dst, start, step):
    BLACK = (255, 255, 255)
    WHITE = (255, 255, 255)
    MAX_LEN = 1024
    frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    corner = (2, 25)
    fontScale = 0.5
    fontColor = BLACK
    lineType = 2
    for path in img_paths:
        image = cv2.imread(path)
        height, width = image.shape[:2]
        current_max = max(height, width)
        if current_max > MAX_LEN:
            height = int(height / current_max * MAX_LEN)
            width = int(width / current_max * MAX_LEN)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        image = cv2.copyMakeBorder(image, 50, 0, 0, 0, cv2.BORDER_CONSTANT, WHITE)
        cv2.putText(image, 'Epoch: ' + str(start), corner, font, fontScale, fontColor, lineType)
        image = image[..., ::-1]
        frames.append(image)
        start += step
    imageio.mimsave(dst, frames, 'GIF', duration=0.5)


if __name__ == '__main__':
    # encoder
    # encoder = Encoder()
    # print(encoder)
    # image_tensors = torch.ones(64, 3, 128, 128)  # N x D x H x W
    # z = encoder(image_tensors)

    # generator
    # generator = Generator()
    # print(generator)
    # z_tensors = torch.ones(64, 120)  # N x Z
    # g = generator(z_tensors)

    # discriminatorZ
    # discriminatorZ = DiscriminatorZ()
    # print(discriminatorZ)
    # z_tensors = torch.ones(64, 100)  # N x Z
    # dz = discriminatorZ(z_tensors)

    # discriminatorImg
    discriminatorImg = DiscriminatorImg()
    # print(discriminatorImg)
    image_tensors = torch.ones(64, 3, 128, 128)  # N x D x H x W
    labels = torch.ones(64, 20)
    di = discriminatorImg(image_tensors, labels, "cpu")
    # print(di.shape)
