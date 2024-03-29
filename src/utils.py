import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pykitti
import torch
import torch.optim as optims
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader

from radam import RAdam
from constants import *

# epsilon value for euler transformation
EPS = np.finfo(float).eps * 4.0

kittiTransform = transforms.Compose([transforms.ToTensor()])

# value found using function from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/preprocess.py
normalizer = transforms.Compose(
    [
        transforms.Normalize(
            (0.19007764876619865, 0.15170388157131237, 0.10659445665650864),
            (0.2610784009469139, 0.25729316928935814, 0.25163823815039915),
        )
    ]
)


# Not used but fundamental kitti dataset
class KittiOdometryDataset(Dataset):
    def __init__(
        self,
        seq,
        original=False,
        path=KITTI_PATH,
        transform=kittiTransform,
        left=True,
        stereo=False,
    ):
        self.odom = pykitti.odometry(path, seq)
        self.transform = transform
        self.left = left
        self.original = original

    def __len__(self):
        return len(self.odom)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        left = 0 if self.left else 1

        if self.original:
            cur_rgb = self.odom.get_rgb(idx)[left]
        else:
            cur_rgb = self.transform(self.odom.get_rgb(idx)[left])
        cur_odom = torch.Tensor(self.odom.poses[idx])

        if HALF:
            return cur_rgb.half(), cur_odom.half()
        else:
            return cur_rgb, cur_odom


# for predefined kitti
class KittiPredefinedDataset(Dataset):
    def __init__(
        self,
        seqs=TRAIN_SEQ,
        path=KITTI_PATH,
        transform=kittiTransform,
        left=True,
        stereo=False,
    ):
        self.train_sets = []
        for seq in seqs:
            self.train_sets.append(KittiOdometryRandomSequenceDataset(seq, path=path))
        self.idxs = []
        self.seps = []
        for name, ts in enumerate(self.train_sets):
            for i in range(len(ts) - SEQ_LENGTH - 1):
                self.idxs.append(i)
                self.seps.append(name)

    def __len__(self):
        return len(self.seps)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        return self.train_sets[self.seps[idx]][self.idxs[idx]]


# for validation set where I want to reduce the length of dataset to VALI_LENGTH
class KittiPredefinedValidationDataset(Dataset):
    def __init__(
        self,
        seqs=VALI_SEQ,
        path=KITTI_PATH,
        transform=kittiTransform,
        left=True,
        stereo=False,
    ):
        self.train_sets = []
        for seq in seqs:
            self.train_sets.append(KittiOdometryRandomSequenceDataset(seq, path=path))
        self.idxs = []
        self.seps = []
        for name, ts in enumerate(self.train_sets):
            for i in range(0, len(ts) - SEQ_LENGTH - 1):
                self.idxs.append(i)
                self.seps.append(name)
        np.random.seed(42)
        rand_idxs = np.random.choice(range(len(self.idxs)), VALI_LENGTH)
        self.idxs = np.stack(self.idxs)
        self.seps = np.stack(self.seps)
        self.idxs = self.idxs[rand_idxs]
        self.seps = self.seps[rand_idxs]

    def __len__(self):
        return VALI_LENGTH

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        return self.train_sets[self.seps[idx]][self.idxs[idx]]


# base dataset to load the subsequences and do preprocessing
class KittiOdometryRandomSequenceDataset(Dataset):
    def __init__(
        self,
        seq,
        path=KITTI_PATH,
        transform=kittiTransform,
        normalizer=normalizer,
        left=True,
        stereo=False,
    ):
        self.odom = pykitti.odometry(path, seq)
        self.transform = transform
        self.left = left
        self.normalizer = normalizer

    def __len__(self):
        return len(self.odom) - 1 - SEQ_LENGTH

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        left = 0 if self.left else 1
        rgb = []
        pos = []
        angle = []
        original_angle = torch.FloatTensor(euler_from_matrix(self.odom.poses[idx]))
        original_pos = torch.FloatTensor(se3_to_position(self.odom.poses[idx]))
        original_rot = se3_to_rot(self.odom.poses[idx]).T
        pos.append(original_pos)
        angle.append(original_angle)
        for i in range(SEQ_LENGTH):
            cur_rgb = self.transform(self.odom.get_rgb(idx + i)[left])
            next_rgb = self.transform(self.odom.get_rgb(idx + i + 1)[left])
            cur_rgb = cur_rgb - 0.5
            next_rgb = next_rgb - 0.5
            cur_rgb = self.normalizer(cur_rgb)
            next_rgb = self.normalizer(next_rgb)
            next_pos = torch.FloatTensor(se3_to_position(self.odom.poses[idx + i + 1]))
            next_angle = torch.FloatTensor(
                euler_from_matrix(self.odom.poses[idx + i + 1])
            )
            rgb.append(torch.cat((cur_rgb, next_rgb), dim=0))
            pos.append(next_pos)
            angle.append(next_angle)
        rgb = torch.stack(rgb)
        pos = torch.stack(pos)
        angle = torch.stack(angle)

        # preprocessing
        # relative poses and orientation
        pos[1:] = pos[1:] - original_pos
        angle[1:] = angle[1:] - original_angle

        for i in range(1, len(pos)):
            loc = torch.FloatTensor(original_rot.dot(pos[i]))
            pos[i][:] = loc[:]

        pos[2:] = pos[2:] - pos[1:-1]
        angle[2:] = angle[2:] - angle[1:-1]

        for i in range(1, len(angle)):
            angle[i][0] = normalize_angle_delta(angle[i][0])
            angle[i][1] = normalize_angle_delta(angle[i][1])
            angle[i][2] = normalize_angle_delta(angle[i][2])
        if HALF:
            return (
                rgb.type(torch.HalfTensor),
                pos.type(torch.HalfTensor),
                angle.type(torch.HalfTensor),
            )
        else:
            return rgb, pos, angle


# if we want to resize the entire original dataset before preprocessing
def transform_data():
    path = "{KITTI_PATH}/sequences/"
    seqs = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    for seq in seqs:
        dirs = os.listdir(path + seq + "/image_2")
        for item in dirs:
            if os.path.isfile(path + seq + "/image_2/" + item):
                im = Image.open(path + seq + "/image_2/" + item)
                f, e = os.path.splitext(path + seq + "/image_2/" + item)
                imResize = im.resize((620, 188))
                imResize.save(f + ".png", "PNG", quality=100)


# play sequence of images in opencv
def play_sequence(tl):
    for s in tl:
        image = s[0][0]
        opencvImage = cv2.cvtColor(np.array(image), cv2.CV_16U)
        cv2.imshow("seq", opencvImage)
        cv2.waitKey(1)


# draw ground truth
def draw_gt(seq):
    x = []
    y = []
    odom = pykitti.odometry(KITTI_PATH, seq)
    for i in range(len(odom)):
        t = se3_to_position(odom.poses[i])
        x.append(t[0])
        y.append(t[2])
    plt.plot(x, y, color="g", label="ground truth")


# draw ground truth and predicted
def draw_route(y, y_hat, name, weight_folder=WEIGHT_FOLDER, c_y="r", c_y_hat="b"):
    plt.clf()
    x = [v[0] for v in y]
    y = [v[2] for v in y]
    plt.plot(x, y, color=c_y, label="ground truth")

    x = [v[0] for v in y_hat]
    y = [v[2] for v in y_hat]
    plt.plot(x, y, color=c_y_hat, label="ground truth")
    plt.savefig(f"{weight_folder}/{name}")
    plt.gca().set_aspect("equal", adjustable="datalim")


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def euler_from_matrix(matrix):
    # y-x-z Tait–Bryan angles intrincic
    # the method code is taken from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py
    i = 2
    j = 0
    k = 1
    repetition = 0
    frame = 1
    parity = 0

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def rev_euler_from_matrix(matrix):
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    M = R.from_dcm(M)
    M = M.inv()
    M = M.as_dcm(M)
    return euler_from_matrix(M)


def euler_to_rot(euler):
    r = R.from_euler("zyx", euler, degrees=True)
    return r.as_dcm()


def se3_to_euler(mat):
    r = np.array(mat, dtype=np.float64)[:3, :3]
    r = R.from_dcm(r)
    return r.as_euler("zyx", degrees=True)


def rot_to_euler(mat):
    r = np.array(mat, dtype=np.float64)
    return r.as_euler("zyx", degrees=True)


def se3_to_rot(mat):
    return mat[:3, :3]


def se3_to_xy(mat):
    t = mat[:, -1]
    return np.array([t[0], t[2]])


def se3_to_position(mat):
    t = mat[:, -1][:-1]
    return t


# due to -pi to pi discontinuity
def normalize_angle_delta(angle):
    if angle > np.pi:
        angle = angle - 2 * np.pi
    elif angle < -np.pi:
        angle = 2 * np.pi + angle
    return angle


def test(
    model, dataloader, device, epoch, draw_seq=None, path=KITTI_PATH, test_seq=TEST_SEQ
):
    model.eval()
    answer = [[0.0] * 6]
    gt = []
    odom = pykitti.odometry(path, test_seq[0])
    traj = []
    cur_R = np.eye(3)
    cur_t = np.zeros((3, 1))
    for i in range(len(odom)):
        gt.append(se3_to_position(odom.poses[i]))
    for i, batch in enumerate(dataloader):
        seq, _, _ = batch
        seq = seq.to(device)
        predicted = model(seq)
        predicted = predicted.data.cpu().numpy()
        if i == 0:
            for pose in predicted[0]:
                for i in range(len(pose)):
                    pose[i] = pose[i] + answer[-1][i]
                answer.append(pose.tolist())
            traj.append(np.concatenate((cur_R.copy(), cur_t.copy()), axis=1).flatten())
            predicted = predicted[1:]
        for poses in predicted:
            ang = eulerAnglesToRotationMatrix([0, answer[-1][0], 0])

            location = ang.dot(poses[-1][3:])
            poses[-1][3:] = location[:]

            last_pose = poses[-1]
            ang_full = eulerAnglesToRotationMatrix(
                [last_pose[1], last_pose[0], last_pose[2]]
            )
            cur_R = ang_full @ cur_R
            for j in range(len(last_pose)):
                last_pose[j] = last_pose[j] + answer[-1][j]
            cur_t = np.array(last_pose[3:]).reshape((3, 1))
            last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi

            answer.append(last_pose.tolist())
            traj.append(np.concatenate((cur_R.copy(), cur_t.copy()), axis=1).flatten())

    if draw_seq:
        np.savetxt(f"{RESULT_FOLDER}/{draw_seq}_{epoch}_pred.txt", traj, fmt="%1.8f")
    else:
        np.savetxt(f"{RESULT_FOLDER}/{DRAW_SEQ}_{epoch}_pred.txt", traj, fmt="%1.8f")


def load_pretrained(model, optimizer, path, device):
    pretrained = torch.load(path, map_location=device)
    model_current_state_dict = model.state_dict()
    model_update_state_dict = {}
    optimizer_current_state_dict = optimizer.state_dict()
    optimizer_update_state_dict = {}

    if path.split("/")[-1] == "pretrained.weights":
        pretrained = {"model_state_dict": pretrained}

    if "model_state_dict" in pretrained.keys():
        for k, v in pretrained["model_state_dict"].items():
            if k in model_current_state_dict.keys():
                model_update_state_dict[k] = v
    if "optimizer_state_dict" in pretrained.keys():
        for k, v in pretrained["optimizer_state_dict"].items():
            if k in optimizer_current_state_dict.keys():
                optimizer_update_state_dict[k] = v

    model_current_state_dict.update(model_update_state_dict)
    optimizer_current_state_dict.update(optimizer_update_state_dict)

    model.load_state_dict(model_current_state_dict)
    optimizer.load_state_dict(optimizer_current_state_dict)


def load_model(model, device, optimizer, lr=0.001, path=""):
    if path != "":
        model.to(device)
        if optimizer == "adagrad":
            optimizer = optims.Adagrad(model.parameters(), lr=lr)
        elif optimizer == "radam":
            optimizer = RAdam(model.parameters(), lr=lr)
        load_pretrained(model, optimizer, path, device)
        cur = int(path.split("/")[-1].split(".")[0])
    else:
        model.load_pretrained_flow(device)
        model.to(device)

        if optimizer == "":
            optimizer = optims.Adagrad(model.parameters(), lr=lr)
        if optimizer == "adagrad":
            optimizer = optims.Adagrad(model.parameters(), lr=lr)
        if optimizer == "radam":
            optimizer = RAdam(model.parameters(), lr=lr)
        cur = 0
    scheduler = optims.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)
    return cur, model, optimizer, scheduler


def train(model, train_loader, optimizer, device, epoch, weight_folder, loss_folder):
    i = 0
    model.train()
    train_losses = 0.0
    immediate_losses = 0.0
    loss_to_plot = 0.0
    losses = []
    for seq, pos, ang in train_loader:
        optimizer.zero_grad()
        i = i + 1
        seq = seq.to(device)
        pos = pos.to(device)
        ang = ang.to(device)
        loss = model.get_loss(seq, pos, ang)
        train_losses += loss.item()
        immediate_losses += loss.item()
        loss_to_plot += loss.item()
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            losses.append(loss_to_plot / 20)
            loss_to_plot = 0.0
    plt.clf()
    plt.plot(losses)
    plt.savefig(f"{loss_folder}/{epoch}.png")
    train_losses /= len(train_loader)
    print(f"Train Epoch {epoch}th loss: {train_losses}")
    return train_losses


def validate(model, test_loader, device, epoch):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for seq, pos, ang in test_loader:
            seq = seq.to(device)
            pos = pos.to(device)
            ang = ang.to(device)
            tloss = model.get_loss(seq, pos, ang)
            test_loss += tloss.item()
    test_loss /= len(test_loader)
    print(f"Validate Epoch {epoch}th loss: {test_loss}")
    return test_loss
