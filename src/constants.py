import os

SEQ_LENGTH = 6
VALI_LENGTH = 100
EPOCH = 400
EPOCH_SAVE = 5
BATCH_SIZE = 6
TEST_BATCH_SIZE = 6
DRAW_BATCH_SIZE = 6
NUM_WORKERS = 12
DRAW_NUM_WORKERS = 12
HALF = False
POSE = True

TRAIN_SEQ = ["00", "02", "08", "09"]  #  "03", "04", "05", "06", "07", "10"]
VALI_SEQ = ["01", "04"]
TEST_SEQ = ["01", "03", "05", "06", "10"]
DRAW_SEQ = "00"
PRETRAIN_FLOW_PATH = "weights/flownets_bn_EPE2.459.pth.tar"

VO = "deepvo" + "_weight1"
WEIGHT_FOLDER = f"../{VO}_w"
LOSS_FOLDER = f"../{VO}_l"
RESULT_FOLDER = f"../{VO}_results"
KITTI_PATH = "../dataset3"

for p in [WEIGHT_FOLDER, LOSS_FOLDER, RESULT_FOLDER]:
    if not os.path.exists(p):
        os.makedirs(p)

CUDA_DEVICE_NUM = "cuda:0"
OPTIMIZER = "radam"