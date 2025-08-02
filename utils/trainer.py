import time
# from tqdm import tqdm

import torch
# from torch import nn
# import torch.nn.functional as F
# from .mutual_info import Mutual_info, Mutual_info_cnn
# from .distillation import KL_loss


def train(net, contra_head, super_head, awl, criterion1, criterion2, \
        criterion3, contrastive_loader, train_loader, train_optimizer, args):
    net.train()
    contra_head.train()
    super_head.train()

    train_correct = 0
    train_bar = enumerate(zip(contrastive_loader, train_loader))
    start_time = time.time()
    
    for step, ((U_11, U_21, U_12, U_22, target), (S_1, S_2, label)) in train_bar:
    # for step, ((U_11, U_12, U_21, U_22, target), (S_1, S_2, label)) in train_bar:
        U_11 = U_11.cuda()
        U_21 = U_21.cuda()
        U_12 = U_12.cuda()
        U_22 = U_22.cuda()
        # target = target.cuda()

        # ########## contra ####################################>>>>>>>>>>>>>>>
        u_xoutput_1, u_youtput_1, u_output_1 = net(U_11, U_21)         # 网络的直接输出都没用，要先经过contra_head 或者 super_head 才能行。
        uc_xoutput_1, uc_youtput_1, uc_output_1 = contra_head(u_xoutput_1, u_youtput_1, u_output_1)

        u_xoutput_2, u_youtput_2, u_output_2 = net(U_12, U_22)
        uc_xoutput_2, uc_youtput_2, uc_output_2 = contra_head(u_xoutput_2, u_youtput_2, u_output_2)

        contra_loss1 = criterion2(uc_xoutput_1, uc_youtput_1)
        contra_loss2 = criterion2(uc_xoutput_2, uc_youtput_2)
        contra_loss3 = criterion3(uc_output_1, uc_output_2, [uc_xoutput_1, uc_youtput_1, uc_xoutput_2, uc_youtput_2])
        loss_contra = contra_loss1 + contra_loss2 + contra_loss3
        loss_contra = args.lambda_contra * loss_contra
        ################ contra ################################<<<<<<<<<<<<<<


        # ################# KL ##################################>>>>>>>>>>>>>>>
        # loss_kl1 = criterion3(u_xoutput_1, u_output_1) # KL_loss 
        # loss_kl2 = criterion3(u_youtput_1, u_output_1) # KL_loss
        # loss_kl = args.lambda_kl * (loss_kl1 + loss_kl2)
        # ################## KL ################################<<<<<<<<<<<<<<


        ################# Orthogonal ##################################>>>>>>>>>>>>>>>
        # loss_orth1 = criterion4(u_xoutput_1, u_youtput_1)
        # loss_orth2 = criterion4(u_xoutput_2, u_youtput_2)
        # loss_orth = loss_orth1 * loss_orth2
        # loss_orth = args.lambda_orth * loss_contra
        loss_orth = torch.tensor(0)
        ################## Orthogonal ################################<<<<<<<<<<<<<<


        ################# Mutual ##################################>>>>>>>>>>>>>>>
        # # loss_mutual1 = criterion4(u_add11, u_add22, mode = 'similar')
        # loss_mutual2 = criterion4(gate_fused1, u_fuse11, mode = 'dissimilar')
        # loss_mutual3 = mutual_loss2(gate_fused1, u_fuse12, mode = 'dissimilar')
        # loss_mutul = args.lambda_mutual*(loss_mutual2 + loss_mutual3)
        # loss_dist = loss_dist1 + loss_dist2
        # loss_dist = loss_dist1 + loss_dist2 + loss_dist3 + loss_dist4
        ################## Mutual ################################<<<<<<<<<<<<<<



        ################ super ##############################>>>>>>>>>>>>>>>
        label = label - 1
        S_1 = S_1.cuda()
        S_2 = S_2.cuda()
        label = label.cuda()
        
        s_base1, s_base2, s_fuse = net(S_1, S_2)
        s_out = super_head(s_fuse)

        loss_super = criterion1(s_out, label)
        loss_super = args.lambda_super * loss_super
        # print(loss_super.item())
        ########## super ######################################<<<<<<<<<<<<<<


        ################# KL ##################################>>>>>>>>>>>>>>>
        # loss_kl1 = criterion3(s_base1, s_fuse) # KL_loss 
        # loss_kl2 = criterion3(s_base2, s_fuse) # KL_loss
        # loss_kl = args.lambda_kl * (loss_kl1 + loss_kl2)
        loss_kl = torch.tensor(0)
        ################## KL ################################<<<<<<<<<<<<<<

        ########## joint ####################################>>>>>>>>>>>>>>>
        if args.awl:
            loss = awl(loss_contra + loss_super + loss_kl)
        else:
            loss = loss_contra + loss_super + loss_kl
    #     ########## joint ####################################<<<<<<<<<<<<<<<

        pred = s_out.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        
    train_accuracy = 100. * train_correct.item() / len(train_loader.dataset)
    train_time = time.time() - start_time
    return round(loss.item(), 4), \
            round(loss_contra.item(), 4), \
            round(loss_orth.item(), 4), \
            round(loss_super.item(), 4), \
            round(loss_kl.item(), 4), \
            round(train_accuracy, 2), \
            round(train_time, 2)


def train_MS2CANet(net, contra_head, super_head, awl, criterion1, criterion2, \
        criterion3, contrastive_loader, train_loader, train_optimizer, args):
    net.train()
    contra_head.train()
    super_head.train()

    train_correct = 0
    train_bar = enumerate(zip(contrastive_loader, train_loader))
    start_time = time.time()
    
    for step, ((U_11, U_21, U_12, U_22, target), (S_1, S_2, label)) in train_bar:
    # for step, ((U_11, U_12, U_21, U_22, target), (S_1, S_2, label)) in train_bar:
        U_11 = U_11.cuda()
        U_21 = U_21.cuda()
        U_12 = U_12.cuda()
        U_22 = U_22.cuda()
        # target = target.cuda()

        # ########## contra ####################################>>>>>>>>>>>>>>>
        u_xoutput_1, u_youtput_1, u_output_1 = net(U_11, U_21)         # 网络的直接输出都没用，要先经过contra_head 或者 super_head 才能行。
        uc_xoutput_1, uc_youtput_1, uc_output_1 = contra_head(u_xoutput_1, u_youtput_1, u_output_1)

        u_xoutput_2, u_youtput_2, u_output_2 = net(U_12, U_22)
        uc_xoutput_2, uc_youtput_2, uc_output_2 = contra_head(u_xoutput_2, u_youtput_2, u_output_2)

        contra_loss1 = criterion2(uc_xoutput_1, uc_youtput_1)
        contra_loss2 = criterion2(uc_xoutput_2, uc_youtput_2)
        contra_loss3 = criterion3(uc_output_1, uc_output_2, [uc_xoutput_1, uc_youtput_1, uc_xoutput_2, uc_youtput_2])
        loss_contra = contra_loss1 + contra_loss2 + contra_loss3
        loss_contra = args.lambda_contra * loss_contra
        ################ contra ################################<<<<<<<<<<<<<<


        # ################# KL ##################################>>>>>>>>>>>>>>>
        # loss_kl1 = criterion3(u_xoutput_1, u_output_1) # KL_loss 
        # loss_kl2 = criterion3(u_youtput_1, u_output_1) # KL_loss
        # loss_kl = args.lambda_kl * (loss_kl1 + loss_kl2)
        # ################## KL ################################<<<<<<<<<<<<<<


        ################# Orthogonal ##################################>>>>>>>>>>>>>>>
        # loss_orth1 = criterion4(u_xoutput_1, u_youtput_1)
        # loss_orth2 = criterion4(u_xoutput_2, u_youtput_2)
        # loss_orth = loss_orth1 * loss_orth2
        # loss_orth = args.lambda_orth * loss_contra
        loss_orth = torch.tensor(0)
        ################## Orthogonal ################################<<<<<<<<<<<<<<


        ################# Mutual ##################################>>>>>>>>>>>>>>>
        # # loss_mutual1 = criterion4(u_add11, u_add22, mode = 'similar')
        # loss_mutual2 = criterion4(gate_fused1, u_fuse11, mode = 'dissimilar')
        # loss_mutual3 = mutual_loss2(gate_fused1, u_fuse12, mode = 'dissimilar')
        # loss_mutul = args.lambda_mutual*(loss_mutual2 + loss_mutual3)
        # loss_dist = loss_dist1 + loss_dist2
        # loss_dist = loss_dist1 + loss_dist2 + loss_dist3 + loss_dist4
        ################## Mutual ################################<<<<<<<<<<<<<<



        ################ super ##############################>>>>>>>>>>>>>>>
        label = label - 1
        S_1 = S_1.cuda()
        S_2 = S_2.cuda()
        label = label.cuda()
        
        s_base1, s_base2, s_fuse = net(S_1, S_2)
        s_out1, s_out2, s_out3 = super_head(s_base1, s_base2)
        s_out = s_out1 + s_out2 + s_out3

        loss_super_1 = criterion1(s_out1, label)
        loss_super_2 = criterion1(s_out2, label)
        loss_super_3 = criterion1(s_out3, label)
        loss_super = args.lambda_super * (loss_super_1 + loss_super_2 + loss_super_3)
        # print(loss_super.item())
        ########## super ######################################<<<<<<<<<<<<<<


        ################# KL ##################################>>>>>>>>>>>>>>>
        # loss_kl1 = criterion3(s_base1, s_fuse) # KL_loss 
        # loss_kl2 = criterion3(s_base2, s_fuse) # KL_loss
        # loss_kl = args.lambda_kl * (loss_kl1 + loss_kl2)
        loss_kl = torch.tensor(0)
        ################## KL ################################<<<<<<<<<<<<<<

        ########## joint ####################################>>>>>>>>>>>>>>>
        if args.awl:
            loss = awl(loss_contra + loss_super + loss_kl)
        else:
            loss = loss_contra + loss_super + loss_kl
    #     ########## joint ####################################<<<<<<<<<<<<<<<

        pred = s_out.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        
    train_accuracy = 100. * train_correct.item() / len(train_loader.dataset)
    train_time = time.time() - start_time
    return round(loss.item(), 4), \
            round(loss_contra.item(), 4), \
            round(loss_orth.item(), 4), \
            round(loss_super.item(), 4), \
            round(loss_kl.item(), 4), \
            round(train_accuracy, 2), \
            round(train_time, 2)