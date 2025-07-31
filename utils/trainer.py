import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from .mutual_info import Mutual_info, Mutual_info_cnn
from .distillation import KL_loss


# Final
def trainer(net, contra_head, super_head, awl, criterion2, criterion4, \
          criterion5, data_loader, contrastive_loader, train_optimizer, args):
    net.train()
    contra_head.train()
    super_head.train()

    correct = 0
    # train_bar = tqdm(enumerate(zip(data_loader, memory_loader)))
    train_bar = enumerate(zip(contrastive_loader, data_loader))
    # train_bar = tqdm(enumerate(data_loader))

    for step, ((U_11, U_12, U_21, U_22, target), (S_1, S_2, label)) in train_bar:
        U_11 = U_11.cuda(non_blocking=True)
        U_12 = U_12.cuda(non_blocking=True)
        U_21 = U_21.cuda(non_blocking=True)
        U_22 = U_22.cuda(non_blocking=True)



    #     ########## contra ####################################>>>>>>>>>>>>>>>
        u_out11, u_out_12, u_out_1f = net(U_11, U_21)
        u_f11, u_f12, u_1f = contra_head(u_out11, u_out_12, u_out_1f)

        u_out21, u_out_22, u_out_2f = net(U_12, U_22)
        u_f21, u_f22, u_2f = contra_head(u_out21, u_out_22, u_out_2f)

        loss_contra1 = criterion4(u_f11, u_f21)
        loss_contra2 = criterion4(u_f12, u_f22)
        loss_contra3 = criterion5(u_1f, u_2f, [u_f11, u_f12, u_f21, u_f22])
        # loss_contra = contra_loss1 + contra_loss2
        loss_contra = args.lambda_contra*(loss_contra1 + loss_contra2 + loss_contra3)
    #     ########## contra ######################################<<<<<<<<<<<<<<




    #     ########## super ####################################>>>>>>>>>>>>>>>
        label = label - 1
        S_1  = S_1.cuda(non_blocking=True)
        S_2  = S_2.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        s_out1, s_out_2, s_out_f = net(S_1, S_2)
        s_out = super_head(s_out_f)
        loss_super = criterion2(s_out, label)
        loss_super = args.lambda_super*loss_super
        ########## super ######################################<<<<<<<<<<<<<<



        ########## joint ####################################>>>>>>>>>>>>>>>
        if args.awl:
            loss = awl(loss_contra, loss_super)
        else:
            loss = loss_contra + loss_super
    #     ########## joint ####################################<<<<<<<<<<<<<<<

        pred = s_out.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).sum()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

    train_accuracy = 100. * correct / len(data_loader.dataset)

    return round(loss.item(), 4), round(loss_contra.item(), 4), round(loss_super.item(), 4), round(train_accuracy.item(), 4)





# 第一个固定版本的变体，不要切分高光谱图像了
def train_cnn2(net, contra_head, super_head, awl, criterion1, criterion2, \
          data_loader, contrastive_loader, train_optimizer, args):
    net.train()
    contra_head.train()
    super_head.train()

    total_loss = 0
    total_num = 0
    train_correct = 0
    test_correct = 0
    # train_bar = tqdm(enumerate(zip(data_loader, memory_loader)))
    train_bar = enumerate(zip(data_loader, contrastive_loader))
    # train_bar = tqdm(enumerate(data_loader))

    start_time = time.time()
    for step, ((U_11, U_12, U_21, U_22, target), (S_11, S_2, label)) in train_bar:
        U_11 = [im.cuda(non_blocking=True) for im in U_11]
        U_12 = [im.cuda(non_blocking=True) for im in U_12]
        U_21 = [im.cuda(non_blocking=True) for im in U_21]
        U_22 = [im.cuda(non_blocking=True) for im in U_22]

    #     ########## contra ####################################>>>>>>>>>>>>>>>

        u_out1, u_out_11, u_out_21 = net(U_11, U_21)
        u_out_11, u_out_21 = contra_head(u_out_11, u_out_21)
        u_out_11 = u_out_11.chunk(args.local_crops_number + 2)
        u_out_21 = u_out_21.chunk(args.local_crops_number + 2)

        u_out2, u_out_12, u_out_22 = net(U_12, U_22)
        u_out_12, u_out_22 = contra_head(u_out_12, u_out_22)
        u_out_12 = u_out_12.chunk(args.local_crops_number + 2)
        u_out_22 = u_out_22.chunk(args.local_crops_number + 2)

        contra_loss1 = 0
        for iq, view1 in enumerate(u_out_11):
            for v, view2 in enumerate(u_out_12):
                if iq == v:
                    continue  # 避免自身匹配   
                contra_loss1 += criterion1(view1, view2)
        # print(contra_loss1.item())

        # contra_loss2 = 0
        # for iq, view1 in enumerate(u_out_21):
        #     for v, view2 in enumerate(u_out_22):
        #         if iq == v:
        #             continue  # 避免自身匹配   
        #         contra_loss2 += criterion1(view1, view2)
        # print(contra_loss2.item())

        loss_contra = contra_loss1
        # loss_contra = contra_loss1 + contra_loss2
        loss_contra = args.lambda_contra*loss_contra

        # print(loss_contra.item())
    #     ########## contra ######################################<<<<<<<<<<<<<<


    #     ########## super ####################################>>>>>>>>>>>>>>>

        label = label - 1
        S_11  = S_11.cuda(non_blocking=True)
        # S_12  = S_12.cuda(non_blocking=True)
        S_2  = S_2.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        s_out1, s_out_11, s_out_21 = net(S_11, S_2)
        # s_out2, s_out_11, s_out_21 = net(S_12, S_2)
        s_out_1 = super_head(s_out1)
        # s_out_2 = super_head(s_out2)
        s_out = s_out_1
        # s_out = s_out_1 + s_out_2
        loss_super = criterion2(s_out, label)
        loss_super = args.lambda_super*loss_super

        # print(loss_super.item())
        ########## super ######################################<<<<<<<<<<<<<<


        ########## joint ####################################>>>>>>>>>>>>>>>
        if args.awl:
            loss = awl(loss_contra, loss_super)
        else:
            # print("not joint")
            loss = loss_contra + loss_super
        # print(loss.item())

    #     ########## joint ####################################<<<<<<<<<<<<<<<

        pred = s_out.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

    train_accuracy = 100. * train_correct / len(data_loader.dataset)
    train_time = time.time() - start_time
    return round(loss.item(), 4), round(loss_contra.item(), 4), round(loss_super.item(), 4), round(train_accuracy.item(), 4), round(train_time, 2)


# 把高光谱的图像增强换成正常的版本，不要原来分一半了
def train_ms2ca2(net, contra_head, super_head, awl, criterion1, criterion2, \
          data_loader, memory_loader, test_loader, train_optimizer, args):
    net.train()
    contra_head.train()
    super_head.train()

    total_loss = 0
    total_num = 0
    train_correct = 0
    test_correct = 0
    # train_bar = tqdm(enumerate(zip(data_loader, memory_loader)))
    train_bar = enumerate(zip(data_loader, memory_loader))
    # train_bar = tqdm(enumerate(data_loader))

    start_time = time.time()
    for step, ((U_11, U_12, U_21, U_22, target), (S_11, S_2, label)) in train_bar:
        U_11 = [im.cuda(non_blocking=True) for im in U_11]
        U_12 = [im.cuda(non_blocking=True) for im in U_12]
        U_21 = [im.cuda(non_blocking=True) for im in U_21]
        U_22 = [im.cuda(non_blocking=True) for im in U_22]

    #     ########## contra ####################################>>>>>>>>>>>>>>>

        u_out_11, u_out_21 = net(U_11, U_21)
        u_out_11, u_out_21 = contra_head(u_out_11, u_out_21)
        u_out_11 = u_out_11.chunk(args.local_crops_number + 2)
        u_out_21 = u_out_21.chunk(args.local_crops_number + 2)

        u_out_12, u_out_22 = net(U_12, U_22)
        u_out_12, u_out_22 = contra_head(u_out_12, u_out_22)
        u_out_12 = u_out_12.chunk(args.local_crops_number + 2)
        u_out_22 = u_out_22.chunk(args.local_crops_number + 2)

        contra_loss1 = 0
        for iq, view1 in enumerate(u_out_11):
            for v, view2 in enumerate(u_out_12):
                if iq == v:
                    continue  # 避免自身匹配   
                contra_loss1 += criterion1(view1, view2)
        # print(contra_loss1.item())

        # contra_loss2 = 0
        # for iq, view1 in enumerate(u_out_21):
        #     for v, view2 in enumerate(u_out_22):
        #         if iq == v:
        #             continue  # 避免自身匹配   
        #         contra_loss2 += criterion1(view1, view2)
        # print(contra_loss2.item())

        loss_contra = contra_loss1
        # loss_contra = contra_loss1 + contra_loss2
        loss_contra = args.lambda_contra*loss_contra

        # print(loss_contra.item())
    #     ########## contra ######################################<<<<<<<<<<<<<<


    #     ########## super ####################################>>>>>>>>>>>>>>>

        label = label - 1
        S_11  = S_11.cuda(non_blocking=True)
        S_2  = S_2.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        s_out_11, s_out_21 = net(S_11, S_2)
        s_out_1, s_out_2, s_out_3 = super_head(s_out_11, s_out_21)
        # s_out = s_out_1
        s_out = s_out_1 + s_out_2 + s_out_3

        loss_super1 = criterion2(s_out_1, label)
        loss_super2 = criterion2(s_out_2, label)
        loss_super3 = criterion2(s_out_3, label)
        loss_super = loss_super1 + loss_super2 + loss_super3
        loss_super = args.lambda_super*loss_super

        # print(loss_super.item())
        ########## super ######################################<<<<<<<<<<<<<<


        ########## joint ####################################>>>>>>>>>>>>>>>
        if args.awl:
            loss = awl(loss_contra, loss_super)
        else:
            # print("not joint")
            loss = loss_contra + loss_super
        # print(loss.item())

    #     ########## joint ####################################<<<<<<<<<<<<<<<

        pred = s_out.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        
    train_accuracy = 100. * train_correct / len(memory_loader.dataset)
    train_time = time.time() - start_time
    return round(loss.item(), 4), round(loss_contra.item(), 4), round(loss_super.item(), 4), round(train_accuracy.item(), 4), round(train_time, 2)

