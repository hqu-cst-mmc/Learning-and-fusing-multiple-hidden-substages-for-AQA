import argparse
import logging
import os
from os import listdir
from os.path import isfile, join, isdir

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
from total_score import TTC
from feature_concat import TFN1,TFN2,TFN3,TFN4,TFN5
# from i3dpt import Unit3Dpy, I3D
from utils import transfer_model
from dataset import divingDataset
# from visualize import make_dot
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import time

np.seterr(divide='ignore', invalid='ignore')


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# setup_seed(20)


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
parser = argparse.ArgumentParser(description="Diving")

parser.add_argument("--load", default=0, type=int,
                    help="Load saved network weights. 0 represent don't load; other number represent the model number")
parser.add_argument("--save", default=0, type=int,
                    help="Save network weights. 0 represent don't save; number represent model number")
parser.add_argument("--epochs", default=150, type=int,
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
parser.add_argument("--task", default='score', type=str,
                    help="task to be overall score or the difficulity level")
parser.add_argument("--only_last_layer", default=0, type=int,
                    help="whether choose to freezen the parameters for all the layers except the linear layer on the pre-trained model")
parser.add_argument("--normalize", default=1, type=int,
                    help="do the normalize for the images")
parser.add_argument("--lr_steps", default=[30, 60], type=int, nargs="+",
                    help="steps to decay learning rate")
parser.add_argument("--use_trained_model", default=1, type=int,
                    help="whether use the pre-trained model on kinetics or not")
parser.add_argument("--random", default=0, type=int,
                    help="random sapmling in training")
parser.add_argument("--test", default=0, type=int,
                    help="whether get into the whole test mode (not recommend) ")
parser.add_argument("--stop", default=0.88, type=float,
                    help="Perform early stop")
parser.add_argument("--tcn_range", default=[1, 2, 3, 4, 5], type=list,
                    help="which part of tcn to use (0 is not using)")
parser.add_argument("--downsample", default=2, type=int,
                    help="downsample rate for stages")
parser.add_argument("--region", default=0, type=int,
                    help="1 or 2. 1 is stage 0, 1, 2, 3 (without sending); 2 is stage 0, 1, 2 (without entering into water and ending)")
parser.add_argument("--allstage", default=1, type=int,
                    help="sampled cover all stage")


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
        label_file = './data_files/difficulty_level.npy'

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
    dset_train = divingDataset(data_folder, train_file, label_file, diff_file, range_file, transformations,
                               tcn_range=options.tcn_range, random=options.random, size=options.size,
                               downsample=options.downsample, region=options.region, allstage=options.allstage)
    # else:
    #     dset_train = divingDataset(data_folder, train_file, label_file, range_file, transformations,
    #                            tcn_range=options.tcn_range, random=options.random, size=options.size,
    #                            downsample=options.downsample, region=options.region, allstage=options.allstage)

    if options.test:
        # print 'test in train'
        dset_test = divingDataset(data_folder, test_file, label_file, diff_file, range_file, transformations, test=1,
                                  tcn_range=options.tcn_range,
                                  size=options.size)
        options.batch_size = 1
    else:
        # print 'no test in train'
        writer1 = SummaryWriter('runs/exp/train-ag')
        writer2 = SummaryWriter('runs/exp/test-ag')

        dset_test = divingDataset(data_folder, test_file, label_file, diff_file, range_file, transformations,
                                  tcn_range=options.tcn_range, random=options.random, test=0, size=options.size,
                                  downsample=options.downsample, region=options.region, allstage=options.allstage)

    train_loader = DataLoader(dset_train,
                              batch_size=options.batch_size,
                              shuffle=True,
                              )

    test_loader = DataLoader(dset_test,
                             # batch_size=int(options.batch_size/2),
                             batch_size=options.batch_size,
                             shuffle=True,
                             )

    # use_cuda=1
    use_cuda = (len(options.gpuid) >= 1)
    # if options.gpuid:
    # cuda.set_device(int(options.gpuid[0]))

    # Initial the model
    if options.use_trained_model:
        model = P3D199(pretrained=True, num_classes=400)
        # for name, value in model.named_parameters():
        #     if name == 'fc':
        #         value.requires_grad = False
    else:
        model = P3D199(pretrained=False, num_classes=400)

    for param in model.parameters():
        param.requires_grad = True

    if options.only_last_layer:
        for param in model.parameters():
            param.requires_grad = False

    model = transfer_model(model, num_classes=1, model_type="P3D")
    ttc_model = TTC().cuda()
    tfc_model1 = TFN1().cuda()
    tfc_model2 = TFN2().cuda()
    tfc_model3 = TFN3().cuda()
    tfc_model4 = TFN4().cuda()
    tfc_model5 = TFN5().cuda()

    if use_cuda:
        model.cuda()
    #	model = nn.DataParallel(model, devices=gpuid)

    start_epoch = 0
    if options.load:
        logging.info("=> loading checkpoint" + str(options.load) + ".tar")
        # checkpoint = torch.load('./results(new-split)/checkpoint' + str(options.load) + '.tar')
        checkpoint = torch.load('./results【最终版】/checkpoint' + str(options.load) + '.tar')
        # checkpoint = torch.load('./models/checkpoint' + str(options.load) + '.tar')

        start_epoch = checkpoint['epoch']
        logging.info("=> start epoch: " + str(start_epoch))
        model.load_state_dict(checkpoint['state_dict1'])
        ttc_model.load_state_dict(checkpoint['state_dict2'])
        tfc_model1.load_state_dict(checkpoint['state_dict3'])
        tfc_model2.load_state_dict(checkpoint['state_dict4'])
        tfc_model3.load_state_dict(checkpoint['state_dict5'])
        tfc_model4.load_state_dict(checkpoint['state_dict6'])
        tfc_model5.load_state_dict(checkpoint['state_dict7'])



    # criterion = nn.MSELoss()
    criterion = nn.MSELoss()

    # fc_for_total_score = nn.Linear(5, 1).cuda()

    if options.only_last_layer:
        optimizer = eval("torch.optim." + options.optimizer)(model.fc.parameters(), lr=options.learning_rate,weight_decay=0.01)
    else:
        if options.optimizer == "SGD":
            optimizer = torch.optim.SGD([{'params':[param for name, param in model.named_parameters()
                                                                         if 'fc' not in name]},{'params':ttc_model.parameters()}],
                                        options.learning_rate,
                                        momentum=0.9,
                                        weight_decay=5e-4)
        else:
            optimizer = eval("torch.optim." + options.optimizer)([{'params':model.parameters()},{'params':ttc_model.parameters()},
                                                                  {'params':tfc_model1.parameters()},{'params':tfc_model2.parameters()}
                                                                  ,{'params':tfc_model3.parameters()}
                                                                  ,{'params':tfc_model4.parameters()},{'params':tfc_model5.parameters()},
                                                                  ], lr=options.learning_rate, weight_decay=5e-4)

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
                all_labels = []
                # get_item方法
                loss_sum = []
                score=0
                weights_tscore = 0
                for it, train_data in enumerate(train_loader, 0):
                    loss_sum = 0
                    # 每个it里8个batch
                    vid_tensor_save =[]
                    train_output_sum = []
                    vid_tensor_sum = []
                    all_score = 0
                    train_output_sum = 0
                    vid_tensor_save = []
                    for tcncover in range(1, 6):
                        vid_tensor, labels, diffs = train_data
                        if use_cuda:
                            vid_tensor_single, labels_single, diffs_single = Variable(vid_tensor[tcncover - 1]).cuda(), Variable(
                                labels[tcncover - 1]).cuda(), Variable(diffs[tcncover - 1]).cuda()
                            labels_single = labels_single[:,np.newaxis]
                            diffs_single = diffs_single[:,np.newaxis]
                            # vid_tensor_single shape:[2,3,16,160,160]
                            # print('vid_tensor:', vid_tensor)
                            # print('labels:', labels)
                        else:
                            vid_tensor, labels = Variable(vid_tensor), Variable(labels)
                        model.train()
                        train_output = model(vid_tensor_single)
                        ## [batch,2048]
                        # train_output = train_output[1]
                        tfc_model1.train()
                        tfc_model2.train()
                        tfc_model3.train()
                        tfc_model4.train()
                        tfc_model5.train()
                        ## [batch,1]
                        tfc_model = eval('tfc_model' + str(tcncover))
                        train_output = tfc_model(train_output)
                        vid_tensor_save.append(train_output)
                    ## [2,5]
                    train_output_sum = torch.cat((vid_tensor_save[0],vid_tensor_save[1],
                                                   vid_tensor_save[2],vid_tensor_save[3],vid_tensor_save[4]),1)
                    ttc_model.train()
                    ## [2,1]
                    ## 结果是归一化后的分数
                    train_output_sum = ttc_model(train_output_sum)
                    train_output_sum = train_output_sum * 30 * diffs_single
                    weights_tscore = ttc_model.fc_ttc.weight.data.cpu().numpy()
                    ##存放整个视频 每个阶段的特征
                    loss = criterion(train_output_sum, labels_single)
                    # loss = criterion_new(train_output_sum.squeeze(), labels_single.long().squeeze())

                    all_train_output = np.append(all_train_output, train_output_sum.data.cpu().numpy()[:, 0])
                    all_labels = np.append(all_labels, labels_single.data.cpu().numpy())
                    if it % 100 == 0:
                    # # print(train_output.data.cpu().numpy()[0][0], '-', labels_single.data.cpu().numpy()[0])
                    #     # logging.info("loss at iteration {0}: {1}".format(it, loss.item()))
                        f = open('./res-newloss/result_ours_train.txt', mode='a')
                        f.write('\n' + str(train_output_sum.data.cpu().numpy()[0][0]) + '-' + str(labels_single.data.cpu().numpy()[0]))
                        f.write('\n')
                    train_loss = train_loss + loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()



                scheduler.step()
                    #Variable转成numpy
                train_avg_loss = (train_loss / (len(dset_train) / options.batch_size))

                rho, p_val = spearmanr(all_train_output, all_labels)
                logging.info(
                    "Average training loss value per instance is {0}, the corr is {1} at the end of epoch {2}".format(
                        train_avg_loss, rho, epoch_i))
                all_train_loss = np.append(all_train_loss, train_avg_loss)
                f = open('./res-newloss/train—loss.txt', mode='a')
                f.write('\n' + 'train loss at epoch' + str(epoch_i) + ':' + '\n')
                f.write(str(train_avg_loss))
                f.write('\n')

                f = open('./res-newloss/weights_of_5scores.txt', mode='a')
                f.write('weights of 5scores:' + str(weights_tscore[0]))
                f.write('\n')

                writer1.add_scalar('5fc-deeper-execution-score', train_avg_loss, epoch_i + 1)



                if options.save:
                    torch.save({
                        'epoch': epoch_i + 1,
                        'state_dict1': model.state_dict(),
                        'state_dict2':ttc_model.state_dict(),
                        'state_dict3':tfc_model1.state_dict(),
                        'state_dict4': tfc_model2.state_dict(),
                        'state_dict5': tfc_model3.state_dict(),
                        'state_dict6': tfc_model4.state_dict(),
                        'state_dict7': tfc_model5.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, './models/checkpoint' + str(options.save) + '.tar')

                # # main test loop
                with torch.no_grad():
                    model.eval()
                    ttc_model.eval()
                    tfc_model1.eval()
                    tfc_model2.eval()
                    tfc_model3.eval()
                    tfc_model4.eval()
                    tfc_model5.eval()
                    ttc_model.eval()
                    test_loss = 0.0
                    all_test_output = []
                    all_labels = []
                    for it, test_data in enumerate(test_loader, 0):
                        test_output_sum = 0
                        vid_tensor_sum = 0
                        all_score = 0
                        vid_tensor_save = []
                        for tcncover in range(1, 6):
                            vid_tensor, labels, diffs = test_data
                            if use_cuda:
                                vid_tensor_single, labels_single, diffs_single = Variable(vid_tensor[tcncover - 1]).cuda(), Variable(labels[tcncover - 1]).cuda(), Variable(diffs[tcncover - 1]).cuda()
                                labels_single = labels_single[:,np.newaxis]
                                diffs_single = diffs_single[:,np.newaxis]
                            else:
                                vid_tensor, labels = Variable(vid_tensor), Variable(labels)
                            test_output = model(vid_tensor_single)
                            # test_output = test_output[1]
                            ## [batch,1]
                            tfc_model = eval('tfc_model' + str(tcncover))
                            test_output = tfc_model(test_output)
                            vid_tensor_save.append(test_output)
                            ## [2,5]
                        test_output_sum = torch.cat((vid_tensor_save[0], vid_tensor_save[1],
                                                      vid_tensor_save[2], vid_tensor_save[3], vid_tensor_save[4]), 1)
                        test_output_sum = ttc_model(test_output_sum)
                        test_output_sum = test_output_sum *30 *diffs_single
                        # loss = criterion(test_output_sum, labels_single)
                        loss = criterion_new(test_output_sum.squeeze(), labels_single.long().squeeze())
                        all_test_output = np.append(all_test_output, test_output_sum.data.cpu().numpy()[:, 0])
                        all_labels = np.append(all_labels, labels_single.data.cpu().numpy())
                        if it % 10 == 0:
                            # # print(train_output.data.cpu().numpy()[0][0], '-', labels_single.data.cpu().numpy()[0])
                            #     # logging.info("loss at iteration {0}: {1}".format(it, loss.item()))
                            f = open('./res-newloss/result_ours_test.txt', mode='a')
                            f.write('\n' + str(test_output_sum.data.cpu().numpy()[0][0]) + '-' + str(
                                labels_single.data.cpu().numpy()[0]))
                            f.write('\n')
                        test_loss += loss.item()

                    test_avg_loss = test_loss / (len(dset_test) / options.batch_size)
                    rho, p_val = spearmanr(all_test_output, all_labels)
                    logging.info(
                        "Average test loss value per instance is {0}, the corr is {1} at the end of epoch {2}".format(
                            test_avg_loss, rho, epoch_i))
                    # test_corr = test_corr.append(rho)
                    # print('test_corr=',test_corr)
                    all_test_loss = np.append(all_test_loss, test_avg_loss)
                    writer2.add_scalar('two-subNet_combine_2loss', test_avg_loss, epoch_i + 1)

                if rho > options.stop:
                    break

        # f = open('weights_of_5scores.txt', mode='a')
        # f.write('weights of 5scores:' + str(weights_tscore[0]))
        # f.write('\n')
        #######################################################################################################################
        ### loss可视化
        epoch_range = []
        for i in range(0, epoch_i + 1):
            epoch_range.append(i)
        plt.figure()
        plt.plot(epoch_range, all_train_loss, 'b')
        plt.plot(epoch_range, all_test_loss, 'g')
        plt.show()
    else:
        # the last test for visualization
      with torch.no_grad():
            all_corr = []
            all_mse = []
            all_mde = []
            all_times = []
            for time_idx in range(0,35):
              model.eval()
              ttc_model.eval()
              weights_tscore = ttc_model.fc_ttc.weight.data.cpu().numpy()
              f = open('./result-review1/weights_of_5scores1.txt', mode='a')
              f.write('weights of 5scores:' + str(weights_tscore[0]))
              f.write('\n')
              tfc_model1.eval()
              tfc_model2.eval()
              tfc_model3.eval()
              tfc_model4.eval()
              tfc_model5.eval()
              ttc_model.eval()
              test_loss = 0.0
              all_test_output = []
              all_labels = []
              start_time = time.time()
              for it, test_data in enumerate(test_loader, 0):
                    test_output_sum = 0
                    vid_tensor_sum = 0
                    all_score = 0
                    vid_tensor_save = []
                    for tcncover in range(1, 6):
                        vid_tensor, labels, diffs = test_data
                        if use_cuda:
                            vid_tensor_single, labels_single, diffs_single = Variable(
                                vid_tensor[tcncover - 1]).cuda(), Variable(
                                labels[tcncover - 1]).cuda(), Variable(diffs[tcncover - 1]).cuda()
                            labels_single = labels_single[:, np.newaxis]
                            diffs_single = diffs_single[:, np.newaxis]
                        else:
                            vid_tensor, labels = Variable(vid_tensor), Variable(labels)
                        test_output = model(vid_tensor_single)
                        # test_output = test_output[1]
                        ## [batch,1]
                        tfc_model = eval('tfc_model' + str(tcncover))
                        test_output = tfc_model(test_output)
                        vid_tensor_save.append(test_output)
                        ## [2,5]
                    test_output_sum = torch.cat((vid_tensor_save[0], vid_tensor_save[1],
                                                 vid_tensor_save[2], vid_tensor_save[3], vid_tensor_save[4]), 1)
                    sub_output = vid_tensor_save

                    stage_score1 = sub_output[0]
                    stage_score2 = sub_output[1]
                    stage_score3 = sub_output[2]
                    stage_score4 = sub_output[3]
                    stage_score5 = sub_output[4]

                    # f = open('./result-review1/sub_Scores_stage.txt',mode='a')
                    # f.write('video-name' + str(video_name)+'\n')
                    # f.write('stage1:'+ str(stage_score1[0].data.cpu().numpy())+'\n')
                    # f.write('stage2:' + str(stage_score2[0].data.cpu().numpy()) + '\n')
                    # f.write('stage3:' + str(stage_score3[0].data.cpu().numpy()) + '\n')
                    # f.write('stage4:' + str(stage_score4[0].data.cpu().numpy()) + '\n')
                    # f.write('stage5:' + str(stage_score5[0].data.cpu().numpy()) + '\n')
                    # f.write('\n')

                    test_output_sum = ttc_model(test_output_sum)

                    test_output_sum = test_output_sum * 30
                    # f = open('./results/exe_scores_pre.txt', mode='a')
                    # f.write(str(test_output_sum.data.cpu().numpy()[:, 0])
                    # f.write('\n')
                    test_output_sum = test_output_sum* diffs_single
                    all_test_output = np.append(all_test_output, test_output_sum.data.cpu().numpy()[:, 0])
                    all_labels = np.append(all_labels, labels_single.data.cpu().numpy())
                    for i in range(len(labels_single.data.cpu().numpy())):
                        f = open('./result-review1/res_for_test1.txt', mode='a')
                        f.write(
                            '\n' + str(test_output_sum[i].data.cpu().numpy()) + '-' + str(
                                labels_single.data.cpu().numpy()[i]))
              rho, p_val = spearmanr(all_test_output, all_labels)
              n = len(all_labels)
              mse = sum(np.square(all_labels - all_test_output)) / n
              mde = sum(np.abs(all_labels - all_test_output))/ n
              elapse_time = time.time()-start_time
              logging.info("TEST TIME {0} ==> The corr is {1} , the MSE is {2},the MED is {3}, elapse time is {4}".format(time_idx, rho, mse, mde, elapse_time))
              f = open('./result-review1/metrics1.txt', mode='a')
              f.write('\n' + str(time_idx) + '-' + 'rho:' + str(rho) + '-' + 'mse:' + str(mse) + '-' + 'med:' + str(mde) + '-' + 'elapse time:' + str(elapse_time))
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
