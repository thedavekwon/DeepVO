import torch
import torch.nn as nn

PRETRAIN_PATH = "../weights/flownets_EPE1.951.pth.tar"


class DeepVO(nn.Module):
    def __init__(self):
        super(DeepVO, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)

        self.lstm = nn.LSTM(
                        input_size=3*10*1024,
                        hidden_size=1000,
                        num_layers=2,
                        batch_first=True)
        self.linear = nn.Linear(1000, 6)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size*seq_len, x.size(2), x.size(3), x.size(4))
        x = self.flow(x)
        x = x.view(batch_size, seq_len, -1)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def flow(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)
        x = self.conv4_1(x)
        x = self.conv5(x)
        x = self.conv5_1(x)
        x = self.conv6(x)
        return x

    def get_loss(self, seq, pos, ang):
        y_hat = self.forward(seq)
        pos_loss = nn.functional.mse_loss(y_hat[:,:,:3], pos)
        ang_loss = nn.functional.mse_loss(y_hat[:,:,3:], ang)
        return 100 * ang_loss + pos_loss



    def load_pretrained(self, device):
        pretrained_flownet = torch.load(PRETRAIN_PATH, map_location=device)
        current_state_dict = self.state_dict()
        update_state_dict = {}
        for k, v in pretrained_flownet['state_dict'].items():
            if len(k.split(".")) == 3:
                k = "{}.{}".format(k.split(".")[0], k.split(".")[2])
            if k in current_state_dict.keys():
                update_state_dict[k] = v
        current_state_dict.update(update_state_dict)
        self.load_state_dict(current_state_dict)