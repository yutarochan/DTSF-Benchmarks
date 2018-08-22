'''
Deep Time Series Forecasting Benchmarking Tool
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import os
import sys
import time
import math
import tqdm
import argparse
import numpy as np
import configparser

import torch
import torch.nn as nn
import torch.functional as F

import utils.config as cfg
from utils.data import DataUtil
from utils.logger import Logger
import utils.optimizer as optim
from models import LSTNet, DeepTime

# Model Training
def train(data, X, Y, model, criterion, optim, batch_size, evaluateL2, evaluateL1):
    model.train()   # Enable Training Mode

    # Setup Loss Variables
    total_loss = 0
    total_l1 = 0
    total_l2 = 0
    n_samples = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        # Feedforward Pass
        model.zero_grad()
        output = model(X)

        # Apply Scaling and Compute Primary Loss
        scale = data.scale.expand(output.shape[0], data.m)
        loss = criterion(output * scale, Y * scale)

        # Compute Loss Metrics
        total_l1 += evaluateL1(output * scale, Y * scale)
        total_l2 += evaluateL2(output * scale, Y * scale)

        # Backpropagation
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.data
        n_samples += (output.size(0) * data.m)

    # Compute Total Loss
    total_loss /= n_samples
    total_l1 /= n_samples
    total_l2 /= n_samples

    return total_loss, total_l1, total_l2

# Model Evaluation
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()

    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X);
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict,output));
            test = torch.cat((test, Y));

            scale = data.scale.expand(output.size(0), data.m)
            total_loss += evaluateL2(output * scale, Y * scale).data
            total_loss_l1 += evaluateL1(output * scale, Y * scale).data
            n_samples += (output.size(0) * data.m);

    rse = math.sqrt(total_loss / n_samples)/data.rse
    rae = (total_loss_l1/n_samples)/data.rae

    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();

    return total_loss_l1, total_loss, rse, rae, correlation

print('='*150)

# Parse Arguments
# TODO: Fix out the primary codebase to prioritize hyperparameter tuning (use argument based approach, then autogen config file to auto-config?)
parser = argparse.ArgumentParser(description='Deep Time Series Forecasting Benchmark Tool')
parser.add_argument('--data', type=str, required=True, help='Location of the data file.')
parser.add_argument('--model', type=str, required=True, help='Select the model experiment you want to execute.')
parser.add_argument('--param', type=str, default=None, help='Hyperparameter configuration file for the model.')
parser.add_argument('--name', type=str, required=True, help='Model/Experiment Reference Name (For Automatically Generated Experiment Metadata Files)')
parser.add_argument('--cuda', type=str, default=False)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--visualize', type=bool, default=False)
parser.add_argument('--seed', type=int, default=9892, help='Random Seed')
parser.add_argument('--horizon', type=int, default=3, help='Prediction Horizon Size')
parser.add_argument('--cnn_window', type=int, default=3, help='Volumetric CNN Window Size')
parser.add_argument('--window', type=int, default=24 * 7, help='Window Size')
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='Training Batch Size [Default: 128]')
parser.add_argument('--epochs', type=int, default=300, help='Upper Training Epoch Limit [Default: 100]')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to minimize as the cost function. [adam*, sgd, adadelta, adagrad]')
parser.add_argument('--loss_fn', type=str, default='MSE', help='Primary loss function to minimize. [MSE*, MAE]')
parser.add_argument('--chkpt_dir', type=str, required=True, help='Directory for model checkpoint persistence.')
parser.add_argument('--log_dir', type=str, required=True, help='Output file for logging output metrics.')
# TODO: Parameterize Split Ratio (Although we will don't prob have to touch this...)

args = parser.parse_args()

# Load Model Configuraiton File
if args.param: args = cfg.parser(args)

# Enable CUDA GPU
args.cuda = args.gpu is not None
if args.cuda:
    print('> CUDA ENABLED, USING GPU')
    torch.cuda.set_device(args.gpu)

# Initialize Seed Parameter
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Load Dataset
print('> LOADING DATASET')
data = DataUtil(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)
print('DATA RSE: ' + str(data.rse.cpu().numpy()))
print('DATA RAE: ' + str(data.rae.cpu().numpy()))
print()

# Initialize Model
print('> LOAD MODEL\n')
model = eval(args.model).Model(args, data);
if args.cuda: model.cuda()

# Initialize Model Loss Function
if args.loss_fn == 'MSE':
    criterion = nn.MSELoss(size_average=False)
elif args.loss_fn == 'MAE':
    criterion = nn.L1Loss(size_average=False)

evaluateL1 = nn.L1Loss(size_average=False)
evaluateL2 = nn.MSELoss(size_average=False)

if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()

# Initialize Model Optimizer
# TODO: Make optimizer flexible for encoder/decoder structures.
best_val = 10000000
opt = optim.Optim(model.parameters(), args.optimizer, args.lr, args.clip)

# Initialize Logger
logger = Logger(args.log_dir, 'train_l1,train_l2,valid_l1,valid_l2,valid_rae,valid_rse,valid_corr,test_l1,test_l2,test_rae,test_rse,test_corr')

print('> INITIALIZE TRAINING')
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        # Training Phase
        train_loss, train_l1, train_l2  = train(data, data.train[0], data.train[1], model, criterion, opt, args.batch_size, evaluateL2, evaluateL1)

        # Validation Phase
        val_l1, val_l2, val_rse, val_rae, val_corr = evaluate(data, data.valid[0], data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)
        print('| EPOCH {:3d} | Time: {:5.2f}s | TRAIN_LOSS {:5.4f} | VALID_RSE {:5.4f} | VALID_RAE {:5.4f} | VALID_CORR {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_rse, val_rae, val_corr))

        # Model Persistence
        if val_rse < best_val: torch.save(model.state_dict(), args.chkpt_dir)

        # Model Test (For Each 5 Epochs)
        test_l1, test_l2, test_rse, test_rae, test_corr = evaluate(data, data.test[0], data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
        if epoch % 5 == 0:
            print ("TEST_RSE {:5.4f} | TEST_RAE {:5.4f} | TEST_CORR {:5.4f}".format(test_rse, test_rae, test_corr))

        # Metric Reporting & Logging
        def convert(x):
            if type(x) and type(x) is type(torch.Tensor([0.0])): return str(x.detach().cpu().numpy())
            else: return str(x)

        # Output Metrics to Log File
        log_out = ','.join(list(map(convert, [train_l1, train_l2, val_l1, val_l2, val_rae, val_rse, val_corr, test_l1, test_l2, test_rae, test_rse, test_corr])))
        logger.log(log_out)

except KeyboardInterrupt:
    print('=' * 150)
    print('> TERMINATING TRAINING PROCESS...')
