import os
import datetime
from shutil import copyfile
from collections import namedtuple, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.regression import mean_squared_error as mse

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import consts
import cv2


def save_image_normalized(*args, **kwargs):
    save_image(*args, **kwargs, normalize=True, range=(-1, 1), padding=4)


def two_sided(x):
    # [0: 1] -> [-1: 1] 零点的两边
    return 2 * (x - 0.5)


def one_sided(x):
    # [-1: 1] -> [0: 1] 零点的一边
    return (x + 1) / 2


pil_to_model_tensor_transform = transforms.Compose(
    [
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.mul(2).sub(1))  # Tensor elements domain: [0:1] -> [-1:1]
    ]
)


def get_utkface_dataset(root):
    print(root)
    # 有标签的 0.0表示第0个年龄段 男性
    # 哦我知道了标签0到9刚好10段，感觉这还是分组学习，学习到每个年龄端的特征
    ret = lambda: ImageFolder(os.path.join(root, 'labeled'), transform=pil_to_model_tensor_transform)
    try:
        return ret()
    except (RuntimeError, FileNotFoundError):
        sort_to_classes(os.path.join(root, 'unlabeled'), print_cycle=1000)
        return ret()


# 对未打标签的图片进行打标签分类
def sort_to_classes(root, print_cycle=np.inf):
    # Example UTKFace cropped and aligned image file format: [age]_[gender]_[race]_[date&time].jpg.chip.jpg
    # Should be 23613 images, use print_cycle >= 1000
    # Make sure you have > 100 MB free space

    def log(text):
        print('[UTKFace dset labeler] ' + text)

    log('Starting labeling process...')
    # 善用这些骚操作可以简化很多代码 一行即可
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    if not files:
        raise FileNotFoundError('No image files in ' + root)
    copied_count = 0
    sorted_folder = os.path.join(root, '..', 'labeled')
    if not os.path.isdir(sorted_folder):
        os.mkdir(sorted_folder)

    for f in files:
        matcher = consts.UTKFACE_ORIGINAL_IMAGE_FORMAT.match(f)
        # [age]_[gender]_[race]_[date&time].jpg.chip.jpg
        # 如果不是以上格式命名的文件则跳过后续步骤
        if matcher is None:
            continue
        # re.compile("^(\d+)_(\d+)_\d+_(\d+)\.jpg\.chip\.jpg$")
        # group是针对()来说的
        age, gender, dtime = matcher.groups()
        srcfile = os.path.join(root, f)
        # 变成0.0这种
        label = Label(int(age), int(gender))
        dstfolder = os.path.join(sorted_folder, label.to_str())
        dstfile = os.path.join(dstfolder, dtime + '.jpg')
        if os.path.isfile(dstfile):  # 如果已经有这张图片了则跳过
            continue
        if not os.path.isdir(dstfolder):
            os.mkdir(dstfolder)
        copyfile(srcfile, dstfile)
        copied_count += 1
        if copied_count % print_cycle == 0:
            log('Copied %d files.' % copied_count)
    log('Finished labeling process.')


def get_fgnet_person_loader(root):
    return DataLoader(dataset=ImageFolder(root, transform=pil_to_model_tensor_transform), batch_size=1)


def str_to_tensor(text, normalize=False):
    age_group, gender = text.split('.')
    age_tensor = -torch.ones(consts.NUM_AGES)  # 10
    age_tensor[int(age_group)] *= -1  # 变成正数
    gender_tensor = -torch.ones(consts.NUM_GENDERS)  # 2
    gender_tensor[int(gender)] *= -1
    if normalize:  # 扩展 最后result长度为20
        gender_tensor = gender_tensor.repeat(consts.NUM_AGES // consts.NUM_GENDERS)
    result = torch.cat((age_tensor, gender_tensor), 0)
    return result


class Label(namedtuple('Label', ('age', 'gender'))):
    def __init__(self, age, gender):
        super(Label, self).__init__()
        self.age_group = self.age_transform(self.age)

    def to_str(self):
        return '%d.%d' % (self.age_group, self.gender)

    @staticmethod
    # 共10个年龄段0-80岁
    def age_transform(age):
        age -= 1
        if age < 20:
            # first 4 age groups are for kids <= 20, 5 years intervals
            # 0-5 6-10 11-15 16-20
            return max(age // 5, 0)
        else:
            # last (6?) age groups are for adults > 20, 10 years intervals
            # 21-30 31-40 41-50 51-60 61-70 71-80
            return min(4 + (age - 20) // 10, consts.NUM_AGES - 1)

    def to_tensor(self, normalize=False):
        return str_to_tensor(self.to_str(), normalize=normalize)


fmt_t = "%H_%M"
fmt = "%Y_%m_%d"


def default_train_results_dir():
    return os.path.join('.', 'trained_models', datetime.datetime.now().strftime(fmt),
                        datetime.datetime.now().strftime(fmt_t))


def default_where_to_save(eval=True):
    path_str = os.path.join('.', 'results', datetime.datetime.now().strftime(fmt),
                            datetime.datetime.now().strftime(fmt_t))
    if not os.path.exists(path_str):
        os.makedirs(path_str)


def default_test_results_dir(eval=True):
    return os.path.join('.', 'test_results', datetime.datetime.now().strftime(fmt) if eval else fmt)


def print_timestamp(s):
    print("[{}] {}".format(datetime.datetime.now().strftime(fmt_t.replace('_', ':')), s))


class LossTracker(object):
    # Python中使用了某些启发式算法(heuristics)来加速垃圾回收
    def __init__(self, use_heuristics=False, plot=False, eps=1e-3):
        # assert 'train' in names and 'valid' in names, str(names)
        self.losses = defaultdict(lambda: [])
        self.paths = []
        self.epochs = 0
        self.use_heuristics = use_heuristics
        if plot:
            # print("names[-1] - "+names[-1])
            plt.ion()
            plt.show()
        else:
            plt.switch_backend("agg")

    # deprecated
    def append(self, train_loss, valid_loss, tv_loss, uni_loss, path):
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        self.tv_losses.append(tv_loss)
        self.uni_losses.append(uni_loss)
        self.paths.append(path)
        self.epochs += 1
        if self.use_heuristics and self.epochs >= 2:
            delta_train = self.train_losses[-1] - self.train_losses[-2]
            delta_valid = self.valid_losses[-1] - self.valid_losses[-2]
            # 6啊还有这种操作
            if delta_train < -self.eps and delta_valid < -self.eps:
                pass  # good fit, continue training
            elif delta_train < -self.eps and delta_valid > +self.eps:
                pass  # overfit, consider stop the training now
            elif delta_train > +self.eps and delta_valid > +self.eps:
                pass  # underfit, if this is in an advanced epoch, break
            elif delta_train > +self.eps and delta_valid < -self.eps:
                pass  # unknown fit, check your model, optimizers and loss functions
            elif 0 < delta_train < +self.eps and self.epochs >= 3:
                prev_delta_train = self.train_losses[-2] - self.train_losses[-3]
                if 0 < prev_delta_train < +self.eps:
                    pass  # our training loss is increasing but in less than eps,
                    # this is a drift that needs to be caught, consider lower eps next time
            else:
                pass  # saturation \ small fluctuations

    def append_single(self, name, value):
        self.losses[name].append(value)

    def append_many(self, **names):
        for name, value in names.items():
            self.append_single(name, value)

    def append_many_and_plot(self, **names):
        self.append_many(**names)

    def plot(self):
        print("in plot")
        plt.clf()
        graphs = [plt.plot(loss, label=name)[0] for name, loss in self.losses.items()]
        plt.legend(handles=graphs)
        plt.xlabel('Epochs')
        plt.ylabel('Averaged loss')
        plt.title('Losses by epoch')
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)

    @staticmethod
    def show():
        print("in show")
        plt.show()

    @staticmethod
    def save(path):
        plt.savefig(path, transparent=True)

    def __repr__(self):
        ret = {}
        for name, value in self.losses.items():
            ret[name] = value[-1]
        return str(ret)


def mean(l):
    return np.array(l).mean()


def uni_loss(input):
    assert len(input.shape) == 2
    batch_size, input_size = input.size()
    hist = torch.histc(input=input, bins=input_size, min=-1, max=1)  # 计算输入张量的直方图
    return mse(hist, batch_size * torch.ones_like(hist)) / input_size


def easy_deconv(in_dims, out_dims, kernel, stride=1, groups=1, bias=True, dilation=1):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)

    c_in, h_in, w_in = in_dims
    c_out, h_out, w_out = out_dims

    padding = [0, 0]
    output_padding = [0, 0]

    lhs_0 = -h_out + (h_in - 1) * stride[0] + kernel[0]  # = 2p[0] - o[0]
    if lhs_0 % 2 == 0:
        padding[0] = lhs_0 // 2
    else:
        padding[0] = lhs_0 // 2 + 1
        output_padding[0] = 1

    lhs_1 = -w_out + (w_in - 1) * stride[1] + kernel[1]  # = 2p[1] - o[1]
    if lhs_1 % 2 == 0:
        padding[1] = lhs_1 // 2
    else:
        padding[1] = lhs_1 // 2 + 1
        output_padding[1] = 1

    return torch.nn.ConvTranspose2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=kernel,
        stride=stride,
        padding=tuple(padding),
        output_padding=tuple(output_padding),
        groups=groups,
        bias=bias,
        dilation=dilation
    )


def remove_trained(folder):
    if os.path.isdir(folder):
        removed_ctr = 0
        for tm in os.listdir(folder):
            tm = os.path.join(folder, tm)
            if os.path.splitext(tm)[1] == consts.TRAINED_MODEL_EXT:
                try:
                    os.remove(tm)
                    removed_ctr += 1
                except OSError as e:
                    print("Failed removing {}: {}".format(tm, e))
        if removed_ctr > 0:
            print("Removed {} trained models from {}".format(removed_ctr, folder))


def merge_images(batch1, batch2):
    # [128, 3, 128, 128]
    assert batch1.shape == batch2.shape
    merged = torch.zeros(batch1.size(0) * 2, batch1.size(1), batch1.size(2), batch1.size(3), dtype=batch1.dtype)
    for i, (image1, image2) in enumerate(zip(batch1, batch2)):
        merged[2 * i] = image1
        merged[2 * i + 1] = image2
    return merged


def cv_resize(filename):
    img = cv2.imread(filename)
    img1 = cv2.resize(img, (128, 128))
    cv2.imwrite(filename, img1)


if __name__ == '__main__':
    # cv_resize("/media/zouy/workspace/gitcloneroot/AgeProgression/input_test/25_0_me.jpg")

    # 调试labels的维度
    # data_src = consts.UTKFACE_DEFAULT_PATH
    # dataset = get_utkface_dataset(data_src)
    # valid_size = 64
    # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, (len(dataset) - valid_size, valid_size))
    # train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)
    # valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False, drop_last=True)
    # # print(dataset.class_to_idx.items())
    # idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    #
    # for i, (images, labels) in enumerate(train_loader, 1):
    #     labels = torch.stack([str_to_tensor(idx_to_class[l], normalize=True) for l in
    #                           list(labels.numpy())])
    #     print(labels.shape)
    #     break
    print(1)
