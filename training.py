# Importing Main Libraries
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
from datasets import CircleImages
from create_circles import img_dir
from network import CircleNet
import utils
import os
from main import noisy_circle, iou
import sys
import logging
import warnings

warnings.filterwarnings("ignore")

model_save = "/media/HDD_2TB.1/"

# Command Line Arguments
parser = argparse.ArgumentParser(description='Circle Detection Model Training')

parser.add_argument('--train_data', default=img_dir + 'train_data.csv', type=str,
                    help='path to train images csv file')
parser.add_argument('--epochs', default=150, type=int,
                    help='Number of epochs required to run for experiment')
parser.add_argument('--lr', default=0.01, type=float,
                    help='Learning Rate')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch Size for training')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of Data Loading Workers (Multiprocessing)')
parser.add_argument('--resume', default='', type=str,
                    help='Location to load Model from checkpoint if resume')
parser.add_argument('--report_freq', default=1, type=int,
                    help='Number of Epoch to print results after')
parser.add_argument('--save_freq', default=5, type=int,
                    help='Number of Epochs to save the model after')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')

args = parser.parse_args()

# Logging to save output.txt
log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format)
fh = logging.FileHandler(os.path.join(model_save, 'output.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Fixed List of Validation Circle Dimensions
validation_params_img = []
# List to store validation accuracies
validation_acc = []
# List for storing epochs
epochs = []


# Training Functions
def train(model, device, train_queue, optimizer, epoch, criterion1, criterion2, scheduler):
    # Setting Model in Train Mode, total loss to 0
    model.train()
    loss_total = 0.0
    total = 0

    # Iterating over whole train dataset
    for i, (images, target) in enumerate(train_queue):
        # Pushing images, target (y value) to GPU
        images = images.to(device)
        target = target.to(device)
        # Normalizing the target
        target = target / 200

        # FORWARD PASS
        optimizer.zero_grad()
        output = model(images)

        # Train Loss Calculated using HuberLoss with MSE Loss
        loss1 = criterion1(output, target)
        loss2 = criterion2(output, target)
        loss = loss1 + loss2

        # BACKWARD AND OPTIMIZE
        loss.backward()
        optimizer.step()

        # Adding Loss for all the images in batch
        loss_total += loss.item() * target.shape[0]

        # Incrementing total number of images
        total += target.shape[0]

    # Calculating Validation Accuracies and stories them
    acc = validate(model, device)
    validation_acc.append(acc)
    epochs.append(epoch + 1)

    # Saving Accuracies and count of Epochs
    np.savetxt('acc.txt', np.array(validation_acc).reshape((-1, len(validation_acc))), fmt='%.3f')
    np.savetxt('epochs.txt', np.array(epochs).reshape((-1, len(epochs))), fmt='%d')

    # Reporting Results
    if (epoch + 1) % args.report_freq == 0:
        logging.info("Epoch: %d, Train loss: %.8f, Learning Rate: %.8f, Validation Accuracy with minimum AP@0.7: %.3f" %
                     (epoch + 1, loss_total / total, scheduler.get_lr()[0], acc))

    # Saving Model after args.sav_freq
    if (epoch + 1) % args.save_freq == 0:
        logging.info("Saving Model")
        utils.save(model, os.path.join(model_save, 'model_checkpoint.pth.tar'))


def validate(model, device):
    # Model in Evaluation Model
    model.eval()
    results = []

    # Validating Model on Circle Dimensions size 200, Radius 50, noise 2
    for (params, img) in validation_params_img:
        with torch.no_grad():
            image = np.expand_dims(np.asarray(img), axis=0)
            image = torch.from_numpy(np.array(image, dtype=np.float32))
            transforms = utils.image_transforms()
            image = transforms(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            output = model(image)
        predicted = np.round(np.array((200 * output).tolist()[0])).tolist()
        results.append(iou(params, predicted))

    results = np.array(results)
    return (results > 0.7).mean()


def main():
    logging.info("Arguments are --> " + str(vars(args)))

    # CUDA Environment Settings
    np.random.seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True
    torch.manual_seed(1)
    cudnn.enabled = True
    torch.cuda.manual_seed(1)

    logging.info("Creating Model")
    model = CircleNet()

    # Model to Resume or Creating from Scratch
    if args.resume:
        logging.info("Loading Saved Model at location from Checkpoint: " + args.resume)
        model.load_state_dict(torch.load(args.resume))
    else:
        logging.info("Building the Model from Scratch")

    # MSE Loss and HuberLoss
    criterion1 = nn.MSELoss()
    criterion2 = nn.SmoothL1Loss()

    # Pushing the model to GPU
    model = model.to(device)
    criterion1, criterion2 = criterion1.to(device), criterion2.to(device)

    # Adam Optimizer and Scheduler to adjust learning Rate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.epochs, steps_per_epoch=1,
                                                    epochs=args.epochs)

    # Image Transformations
    transforms = utils.image_transforms()
    training_data = CircleImages(args.train_data, transforms)
    # Train Data Loader
    train_queue = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Creating validation set of Circle Dimensions size 200, Radius 50, noise 2
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        validation_params_img.append((params, img))

    # Model Size
    logging.info("Approximate Model Size in (MBs) --> %.8f" % (utils.count_parameters_in_mb(model)))

    # Training the model to epochs
    for epoch in range(args.epochs):
        train(model, device, train_queue, optimizer, epoch, criterion1, criterion2, scheduler)
        scheduler.step()


if __name__ == '__main__':
    main()
