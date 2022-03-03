import argparse
import logging
import os
from os import listdir
from os.path import isfile, join, isdir
import time
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils

# import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
# import cv2
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import StepLR
from p3d_model import P3D199
# from i3dpt import Unit3Dpy, I3D
from utils import transfer_model
from dataset import divingDataset
# from visualize import make_dot
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from total_score import  TTC_con
from tensorboardX import SummaryWriter
np.seterr(divide='ignore', invalid='ignore')

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
parser = argparse.ArgumentParser(description="Diving")


parser.add_argument("--load", default=0, type=int,
                    help="Load saved network weights. 0 represent don't load; other number represent the model number")
parser.add_argument("--save", default=0, type=int,
                    help="Save network weights. 0 represent don't save; number represent model number")
parser.add_argument("--epochs", default=300, type=int,
                    help="Epochs through the data. (default=65)")
parser.add_argument("--learning_rate", "-lr", default=0.0001, type=float,
                    help="Learning rate of the optimization. (default=0.0001)")
parser.add_argument("--batch_size", default=8, type=int,
                    help="Batch size for training. (default=16)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=[], nargs='+', type=str,
                    help="ID of gpu device to use. Empty implies cpu usage.")
parser.add_argument("--size", default=160, type=int,
                    help="size of images.")
parser.add_argument("--task", default='exe', type=str,
                    help="task to be overall score or the difficulity level")
parser.add_argument("--only_last_layer", default=0, type=int,
                    help="whether choose to freezen the parameters for all the layers except the linear layer on the pre-trained model")
parser.add_argument("--normalize", default=1, type=int,
                    help="do the normalize for the images")
parser.add_argument("--lr_steps", default=[30, 90], type=int, nargs="+",
                    help="steps to decay learning rate")
parser.add_argument("--use_trained_model", default=1, type=int,
                    help="whether use the pre-trained model on kinetics or not")
parser.add_argument("--random", default=0, type=int,
                    help="random sapmling in training")
parser.add_argument("--test", default=0, type=int,
                    help="whether get into the whole test mode (not recommend) ")
parser.add_argument("--stop", default=0.79, type=float,
                    help="Perform early stop")
parser.add_argument("--tcn_range", default=[1, 2, 3, 4, 5], type=list,
                    help="which part of tcn to use (0 is not using)")
parser.add_argument("--downsample", default=2, type=int,
                    help="downsample rate for stages")
parser.add_argument("--region", default=0, type=int,
                    help="1 or 2. 1 is stage 0, 1, 2, 3 (without sending); 2 is stage 0, 1, 2 (without entering into water and ending)")
parser.add_argument("--allstage", default=1, type=int,
                    help="sampled cover all stage")

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# setup_seed(20)

def main(options):
    # Path to the directories of features and labels
    train_file = './data_files/training_idx.npy'
    test_file = './data_files/testing_idx.npy'
    data_folder = '/home/donglijia/dlj/diving-score-master/frames'
    diff_file = './data_files/difficulty_level.npy'

    range_file = './data_files/tcn_time_point.npy'
    if options.task == "score":
        label_file = './data_files/overall_scores.npy'
    else:
        label_file = './data_files/execution_score.npy'
        score_file = './data_files/overall_scores.npy'

    if options.normalize:
        transformations = transforms.Compose([transforms.Scale((options.size, options.size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])
    else:
        transformations = transforms.Compose([transforms.Scale((options.size, options.size)),
                                              transforms.ToTensor()
                                              ])
    # 这步加载数据的
    #     if options.allstage:
    #         for stage in range(1,5):
    dset_train = divingDataset(data_folder, train_file, label_file, score_file, diff_file, range_file, transformations,
                               tcn_range=options.tcn_range, random=options.random, size=options.size,
                               downsample=options.downsample, region=options.region, allstage=options.allstage)
    # else:
    #     dset_train = divingDataset(data_folder, train_file, label_file, range_file, transformations,
    #                            tcn_range=options.tcn_range, random=options.random, size=options.size,
    #                            downsample=options.downsample, region=options.region, allstage=options.allstage)

    if options.test:
        # print 'test in train'
        dset_test = divingDataset(data_folder, test_file, label_file, score_file,diff_file, range_file, transformations, test=1,
                                  tcn_range=options.tcn_range,
                                  size=options.size)
        options.batch_size = 10
    else:
        # print 'no test in train'

        writer1 = SummaryWriter('runs/exp/train')
        writer2 = SummaryWriter('runs/exp/test')
        dset_test = divingDataset(data_folder, test_file, label_file,score_file, diff_file,range_file, transformations,
                                  tcn_range=options.tcn_range, random=options.random, test=1, size=options.size,
                                  downsample=options.downsample, region=options.region, allstage=options.allstage)

    train_loader = DataLoader(dset_train,
                              batch_size = options.batch_size,
                              shuffle=True,
                              )

    test_loader = DataLoader(dset_test,
                             # batch_size=int(options.batch_size/2),
                             batch_size= 10  ,
                             shuffle=True,
                             )

    # use_cuda=1
    use_cuda = (len(options.gpuid) >= 1)
    # if options.gpuid:
    # cuda.set_device(int(options.gpuid[0]))

    # Initial the model
    ### 更改：新建一个feNet
    if options.use_trained_model:
        model = P3D199(pretrained=True, num_classes=400)
        # model.conv1_custom = nn.Conv3d(15, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
    else:
        model = P3D199(pretrained=False, num_classes=400)

    for param in model.parameters():
        # if param
        # model.fc.weight.requires_grad = False
        param.requires_grad = True

    model.fc.weight.requires_grad = False

    if options.only_last_layer:
        for param in model.parameters():
            param.requires_grad = False

    # model = transfer_model(model, num_classes=1, model_type="P3D")
    ttc_model = TTC_con().cuda()
    if use_cuda > 0:
        model.cuda()
    #	model = nn.DataParallel(model, devices=gpuid)

    start_epoch = 0
    if options.load:
        logging.info("=> loading checkpoint" + str(options.load) + ".tar")
        checkpoint = torch.load('./result【最终版】-0.80/checkpoint' + str(options.load) + '.tar')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict1'])
        ttc_model.load_state_dict(checkpoint['state_dict2'])


    # criterion = nn.MSELoss()
    criterion = nn.MSELoss()


    if options.only_last_layer:
        optimizer = eval("torch.optim." + options.optimizer)([{'params':[param for name, param in model.named_parameters()
                                                                         if 'fc' not in name]}], lr=options.learning_rate)
    else:
        if options.optimizer == "SGD":
            optimizer = eval("torch.optim.SGD")([{'params':[param for name, param in model.named_parameters()
                                                                         if 'fc' not in name]},{'params':ttc_model.parameters(),
                                                                     }],
                                        lr=options.learning_rate,
                                        momentum=0.9,
                                        weight_decay=5e-4)
        else:
            optimizer = eval("torch.optim." + options.optimizer)([{'params':[param for name, param in model.named_parameters()
                                                                         if 'fc' not in name]},{'params':ttc_model.parameters(),
                                                                     }], lr=options.learning_rate,weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=options.lr_steps[0], gamma=0.1)
    all_test_loss =[]
    if not options.test:
        # main training loop
        # last_dev_avg_loss = float("inf")
        all_train_loss = []
        if options.allstage:
            for epoch_i in range(0, options.epochs):
                logging.info("At {0}-th epoch.".format(epoch_i))
                # if (epoch_i == 30):
                #     print("At epoch 30:",all_train_output)
                #     print(all_labels)
                train_loss = 0.0
                all_train_output = []
                all_scores = []
                # get_item方法
                for it, train_data in enumerate(train_loader, 0):
                    loss_sum = 0
                    # 每个it里8个batch
                    vid_tensor_sum = []
                    vid_tensor_save = []
                    for tcncover in range(1, 6):
                        vid_tensor, labels, diffs, scores = train_data
                        if use_cuda:
                            vid_tensor_single, labels_single, diffs_single,scores_single = Variable(vid_tensor[tcncover - 1]).cuda(), Variable(
                                labels[tcncover - 1]).cuda(),Variable(diffs[tcncover - 1]).cuda(),Variable(scores[tcncover - 1]).cuda()
                            labels_single = labels_single[:,np.newaxis]
                            diffs_single = diffs_single[:,np.newaxis]
                            scores_single = scores_single[:,np.newaxis]
                            # print('vid_tensor:', vid_tensor)
                            # print('labels:', labels)
                        else:
                            vid_tensor, labels = Variable(vid_tensor), Variable(labels)
                        model.train()
                        train_output = model(vid_tensor_single)
                ## train_output[1]代表fc层前面的特征 [2,2048]
                        # train_output = train_output[1]
                        vid_tensor_save.append(train_output)
                    train_output_sum = torch.cat((vid_tensor_save[0],vid_tensor_save[1],
                                                   vid_tensor_save[2],vid_tensor_save[3],vid_tensor_save[4]),1)
                    ## train_output_sum [2,10240]
                    ttc_model.train()
                    train_output_sum = ttc_model(train_output_sum)
                    train_output_sum = train_output_sum * 30
                    loss = criterion(train_output_sum, labels_single)
                    train_output_sum_final = train_output_sum * diffs_single
                    all_train_output = np.append(all_train_output, train_output_sum_final.data.cpu().numpy()[:, 0])
                    all_scores = np.append(all_scores, scores_single.data.cpu().numpy())
                    if it % 100 == 0:
                    # # print(train_output.data.cpu().numpy()[0][0], '-', labels_single.data.cpu().numpy()[0])
                    #     # logging.info("loss at iteration {0}: {1}".format(it, loss.item()))
                        f = open('./result/train_result.txt', mode='a')
                        f.write('\n' + 'train：'+ str(train_output_sum_final.data.cpu().numpy()[0][0]) + '-' + str(scores_single.data.cpu().numpy()[0]))
                        f.write('\n')
                    train_loss = train_loss + loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                    #Variable转成numpy
                train_avg_loss = (train_loss / (len(dset_train) / options.batch_size))

                f = open('./result/train—loss.txt', mode='a')
                f.write('\n' + 'train loss at epoch' + str(epoch_i) + ':' + '\n')
                f.write(str(train_avg_loss))
                f.write('\n')

                rho, p_val = spearmanr(all_train_output, all_scores)
                logging.info(
                    "Average training loss value per instance is {0}, the corr is {1} at the end of epoch {2}".format(
                        train_avg_loss, rho, epoch_i))

                all_train_loss = np.append(all_train_loss, train_avg_loss)

                # f = open('./results/weights_of_5scores.txt', mode='a')
                # f.write('weights of 5scores:' + str(weights_tscore[0]))
                # f.write('\n')

                writer1.add_scalar('10240features-concat-deeper-execution', train_avg_loss, epoch_i + 1)

                if options.save:
                    torch.save({
                        'epoch': epoch_i + 1,
                        'state_dict1': model.state_dict(),
                        'state_dict2':ttc_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, './models/checkpoint' + str(options.save) + '.tar')

                # # main test loop
                with torch.no_grad():
                    model.eval()
                    ttc_model.eval()
                    test_loss = 0.0
                    all_test_output = []
                    all_scores = []
                    for it, test_data in enumerate(test_loader, 0):
                        test_output_sum = 0
                        vid_tensor_sum = 0
                        vid_tensor_save = []
                        for tcncover in range(1, 6):
                            vid_tensor, labels,diffs, scores = test_data
                            if use_cuda:
                                vid_tensor_single, labels_single, diffs_single, scores_single = Variable(
                                    vid_tensor[tcncover - 1]).cuda(), Variable(
                                    labels[tcncover - 1]).cuda(), Variable(diffs[tcncover - 1]).cuda(), Variable(
                                    scores[tcncover - 1]).cuda()
                                labels_single = labels_single[:, np.newaxis]
                                diffs_single = diffs_single[:, np.newaxis]
                                scores_single = scores_single[:, np.newaxis]
                            else:
                                vid_tensor, labels = Variable(vid_tensor), Variable(labels)
                            test_output = model(vid_tensor_single)
                            vid_tensor_save.append(test_output)
                        test_output_sum = torch.cat((vid_tensor_save[0], vid_tensor_save[1],
                                                        vid_tensor_save[2], vid_tensor_save[3], vid_tensor_save[4]), 1)
                        test_output_sum = ttc_model(test_output_sum)
                        test_output_sum = test_output_sum *30
                        loss = criterion(test_output_sum, labels_single)
                        test_output_sum_final = test_output_sum *diffs_single
                        all_test_output = np.append(all_test_output, test_output_sum_final.data.cpu().numpy()[:, 0])
                        all_scores = np.append(all_scores, scores_single.data.cpu().numpy())
                        if it % 10 == 0:
                            # # print(train_output.data.cpu().numpy()[0][0], '-', labels_single.data.cpu().numpy()[0])
                            #     # logging.info("loss at iteration {0}: {1}".format(it, loss.item()))
                            f = open('./result/test_result.txt', mode='a')
                            f.write('\n' +'test：' + str(test_output_sum_final.data.cpu().numpy()[0][0]) + '-' + str(
                                scores_single.data.cpu().numpy()[0]))
                            f.write('\n')
                        test_loss += loss.item()

                    test_avg_loss = test_loss / (len(dset_test) / options.batch_size)
                    # logging.info("Average test loss value per instance is {0}".format(test_avg_loss))
                    # plt.figure()
                    # plt.plot(it, loss, label="loss")
                    # plt.draw()
                    # plt.show()
                    rho, p_val = spearmanr(all_test_output, all_scores)
                    logging.info(
                        "Average test loss value per instance is {0}, the corr is {1} at the end of epoch {2}".format(
                            test_avg_loss, rho, epoch_i))
                    # test_corr = test_corr.append(rho)
                    # print('test_corr=',test_corr)
                    all_test_loss = np.append(all_test_loss, test_avg_loss)

                    writer2.add_scalar('10240features-concat-deeper-execution', test_avg_loss, epoch_i + 1)

                    if rho > options.stop:
                        break

        epoch_range = []
        for i in range(0, epoch_i + 1):
            epoch_range.append(i)
        plt.figure()
        plt.plot(epoch_range, all_train_loss, 'b')
        plt.plot(epoch_range, all_test_loss, 'g')
        plt.show()
    else:
        with torch.no_grad():
        # the last test for visualization
            all_corr = []
            all_mse = []
            all_mde = []
            all_times = []
            for times in range(0,35):
                model.eval()
                ttc_model.eval()
                test_loss = 0.0
                all_test_output = []
                all_scores = []
                start_time = time.time()
                for it, test_data in enumerate(test_loader, 0):
                    test_output_sum = 0
                    vid_tensor_sum = 0
                    vid_tensor_save = []
                    for tcncover in range(1, 6):
                        vid_tensor, labels, diffs, scores = test_data
                        if use_cuda:
                            vid_tensor_single, labels_single, diffs_single, scores_single = Variable(
                                vid_tensor[tcncover - 1]).cuda(), Variable(
                                labels[tcncover - 1]).cuda(), Variable(diffs[tcncover - 1]).cuda(), Variable(
                                scores[tcncover - 1]).cuda()
                            labels_single = labels_single[:, np.newaxis]
                            diffs_single = diffs_single[:, np.newaxis]
                            scores_single = scores_single[:, np.newaxis]
                        else:
                            vid_tensor, labels = Variable(vid_tensor), Variable(labels)
                        test_output = model(vid_tensor_single)
                        vid_tensor_save.append(test_output)
                    test_output_sum = torch.cat((vid_tensor_save[0], vid_tensor_save[1],
                                                 vid_tensor_save[2], vid_tensor_save[3], vid_tensor_save[4]), 1)
                    test_output_sum = ttc_model(test_output_sum)
                    test_output_sum = test_output_sum * 30
                    loss = criterion(test_output_sum, labels_single)
                    test_output_sum_final = test_output_sum * diffs_single
                    all_test_output = np.append(all_test_output, test_output_sum_final.data.cpu().numpy()[:, 0])
                    all_scores = np.append(all_scores, scores_single.data.cpu().numpy())
                        # for i in range(len(labels_single.data.cpu().numpy())):
                        #     logging.info(
                        #         "{0}-{1}".format(test_output_sum.data.cpu().numpy()[i], labels_single.data.cpu().numpy()[i]))

                rho, p_val = spearmanr(all_test_output, all_scores)
                n = len(all_scores)
                mse = sum(np.square(all_scores - all_test_output)) / n
                mde = sum(np.abs(all_scores - all_test_output)) / n
                elapse_time = time.time() - start_time
                logging.info(
                    "TEST TIME {0} ==> The corr is {1} , the MSE is {2}, the overall MDE is {3}, elapse time is {4}".format(
                        times, rho, mse, mde, elapse_time))
                f = open('./result-rw1/metrics1.txt', mode='a')
                f.write('\n' + str(times) + '-' + 'rho:' + str(rho) + '-' + 'mse:' + str(mse) + '-' + 'med:' + str(
                    mde) + '-' + 'elapse time:' + str(elapse_time))
                all_corr = np.append(all_corr, rho)
                all_mse = np.append(all_mse, mse)
                all_mde = np.append(all_mde, mde)
                all_times = np.append(all_times, elapse_time)
        avg_corr = np.average(all_corr)
        avg_mse = np.average(all_mse)
        avg_mde = np.average(all_mde)
        avg_time = np.average(all_times)
        logging.info(
            "Average corr is {0}, MSE is {1},overall MDE is {2}, avg time is {3}".format(
                avg_corr, avg_mse, avg_mde, avg_time))

if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)
