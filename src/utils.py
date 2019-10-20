import cv2
import matplotlib.pyplot as plt
import numpy as np
import pykitti
import torch
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader

kittiTransform = transforms.Compose(
    [transforms.Resize((188, 620)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))]
)
SEQ_LENGTH = 20
TRAIN_SEQ = ["00", "02", "08", "09"]
VALI_SEQ = ["01"]

class KittiOdometryDataset(Dataset):
    def __init__(self, seq, original=False, path="dataset", transform=kittiTransform, left=True, stereo=False):
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
        cur_odom = torch.from_numpy(self.odom.poses[idx])

        return cur_rgb, cur_odom


class KittiPredefinedDataset(Dataset):
    def __init__(self, seqs=TRAIN_SEQ, path="../dataset", transform=kittiTransform, left=True, stereo=False):
        self.train_sets = []
        for seq in seqs:
            self.train_sets.append(KittiOdometryRandomSequenceDataset(seq))
        self.idxs = []
        self.seps = []
        for name, ts in enumerate(self.train_sets):
            for i in range(0, len(ts)-SEQ_LENGTH-1, 5):
                self.idxs.append(i)
                self.seps.append(name)
                
    def __len__(self):
        return len(self.seps)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        return self.train_sets[self.seps[idx]][self.idxs[idx]]


class KittiOdometryRandomSequenceDataset(Dataset):
    def __init__(self, seq, path="../dataset", transform=kittiTransform, left=True, stereo=False):
        self.odom = pykitti.odometry(path, seq)
        self.transform = transform
        self.left = left

    def __len__(self):
        return len(self.odom) - 1 - SEQ_LENGTH

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        left = 0 if self.left else 1
        rgb = []
        pos = []
        angle = []
        initial_angle = se3_to_rot(self.odom.poses[idx]).T
        
        for i in range(SEQ_LENGTH):
            cur_rgb = self.transform(self.odom.get_rgb(idx + i)[left])
            next_rgb = self.transform(self.odom.get_rgb(idx + i + 1)[left])
            cur_pos = torch.from_numpy(se3_to_position(self.odom.poses[idx + i]))
            next_pos = torch.from_numpy(se3_to_position(self.odom.poses[idx + i + 1]))
            next_angle = se3_to_rot(self.odom.poses[idx + i + 1])
            rgb.append(torch.cat((cur_rgb, next_rgb), dim=0))
            pos.append(next_pos - cur_pos)
            angle.append(next_angle)
        for i in range(len(angle)):
            angle[i] = torch.from_numpy(rot_to_euler(initial_angle.dot(angle[i])))
        return torch.stack(rgb), torch.stack(pos).type(torch.float32), torch.stack(angle).type(torch.float32)


# https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/4
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


def load_test_data():
    test_seq = ["03", "04", "05", "06", "07", "10"]
    test_loader = DataLoader(
        KittiOdometryDataset(test_seq[0], original=True), batch_size=1, shuffle=False, num_workers=0,
        collate_fn=my_collate
    )
    return test_loader


def load_train_data():
    train_seq = ["00", "02", "08", "09"]

    train_loaders = []
    for seq in train_seq:
        train_loaders.append(DataLoader(
            KittiOdometryRandomSequenceDataset(seq), batch_size=8, shuffle=True, num_workers=0, collate_fn=my_collate
        ))
    return train_loaders

def generate_train_data():
    train_seq = ["00", "02", "08", "09"]
    # train_seq = ["00"]
    train_sets = []
    for seq in train_seq:
        train_sets.append(
            KittiOdometryRandomSequenceDataset(seq)
        )

    seqs = []
    poses = []
    angs = []
    for ts in train_sets:
        for i in range(0, len(ts), 5):
            seq, pos, ang = ts[i]
            seqs.append(seq)
            poses.append(pos)
            angs.append(ang)
            if (i % 100 == 0):
                print(i)
    return seqs, poses, angs

def shuffle_load(train_loaders):
    train_loader = np.random.choice(train_loaders)
    x = iter(train_loader).next()
    return x


def play_sequence():
    tl = load_test_data()
    for s in tl:
        image = s[0][0]
        opencvImage = cv2.cvtColor(np.array(image), cv2.CV_16U)
        cv2.imshow("seq", opencvImage)
        cv2.waitKey(1)


def draw_route(y, y_hat, c_y="r", c_y_hat="b"):
    x = [v[0] for v in y]
    y = [v[2] for v in y]
    plt.plot(x, y, color=c_y, label="ground truth")

    x = [v[0] for v in y_hat]
    y = [v[2] for v in y_hat]
    plt.plot(x, y, color=c_y_hat, label="ground truth")


def se3_to_euler(mat):
    r = mat[:3, :3]
    r = R.from_dcm(r)
    return r.as_euler('xyz', degrees=True)


def rot_to_euler(mat):
    r = R.from_dcm(mat)
    return r.as_euler('xyz', degrees=True)


def se3_to_rot(mat):
    return mat[:3, :3]


def se3_to_xy(mat):
    t = mat[:, -1]
    return np.array([t[0], t[2]])


def se3_to_position(mat):
    t = mat[:, -1][:-1]
    return t


def train(model, train_loader, optimizer, device, epoch):
    i = 0
    model.train()
    for seq, pos, ang in train_loader:
        optimizer.zero_grad()
        i = i + 1
        seq = seq.to(device)
        pos = pos.to(device)
        ang = ang.to(device)
        loss = model.get_loss(seq, pos, ang)
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            print(f"{i}th loss: {loss.item()}")
    print(f"Epoch {epoch}th loss: {loss.item()}")
    return loss.item()

def validate(model, test_loader, optimizer, device, epoch):
    i = 0
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for seq, pos, ang in test_loader:
            seq = seq.to(device)
            pos = pos.to(device)
            ang = ang.to(device)
            test_loss += model.get_loss(seq, pos, ang).item()
    test_loss /= len(test_loss.dataset)
    print(f"Epoch {epoch}th loss: {test_loss.item()}")
    return test_loss.item()