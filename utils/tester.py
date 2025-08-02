import os 
import time
import torch
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,cohen_kappa_score


def test_resNet2(net, super_head, criterion2, data_loader, args, groundTruth=None, visulation=False):
    net.eval()
    super_head.eval()
    
    test_preds = []
    targets = []
    correct = 0

    start_time = time.time()
    with torch.no_grad():
        for S_1, S_2, target in data_loader:
        # for data, _, target in data_loader:
            target = target - 1
            S_1 = S_1.to(args.device)
            S_2 = S_2.to(args.device)
            target = target.to(args.device)
                
            s_base1, s_base2, s_fuse = net(S_1, S_2)
            s_out = super_head(s_fuse)

            test_loss = criterion2(s_out, target).item()
            test_pred = s_out.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(s_out, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            targets.append(target.cpu())

        test_accuracy = 100. * correct.item() / len(data_loader.dataset)
        # print('Accuracy: {}/{} ({:.2f}%)\n'.format(
        #             correct, len(data_loader.dataset), test_accuracy))
    test_time = time.time() - start_time

    if visulation and groundTruth.any() != None:
        hight, width = groundTruth.shape
        test_preds = torch.cat(test_preds, dim=0).numpy()
        predict_labels = test_preds.reshape(hight, width)

        # print(np.unique(predict_labels))
        draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy, 2)) + "_knn_full"))
        # 背景像元置为 0，因为 pred 预测了所有的像元，但是背景像元并不需要画出来
        for i in range(hight):
            for j in range(width):
                if groundTruth[i][j] == 0:
                    predict_labels[i][j] = 0

        draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy, 2)) + "_knn_label"))    

    return test_loss, test_preds, targets, round(test_accuracy, 2), round(test_time, 2)


def test_MS2CANet(net, super_head, criterion1, data_loader, args):
    net.eval()
    super_head.eval()
    test_preds = []
    targets = []
    correct = 0

    start_time = time.time()
    with torch.no_grad():
        for S_1, S_2, target in data_loader:
        # for data, _, target in data_loader:
            target = target - 1
            S_1 = S_1.to(args.device)
            S_2 = S_2.to(args.device)
            target = target.to(args.device)
                
            s_base1, s_base2, s_fuse = net(S_1, S_2)
            out_e1, out_e2, out_e3 = super_head(s_base1, s_base2)
            out_e = out_e1 + out_e2 + out_e3
    
            test_loss = criterion1(out_e, target).item()
            test_pred = out_e.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(s_out, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            targets.append(target.cpu())

        test_accuracy = (100. * correct / len(data_loader.dataset)).item()
        # print('Accuracy: {}/{} ({:.2f}%)\n'.format(
        #             correct, len(data_loader.dataset), test_accuracy))
    test_time = time.time() - start_time

    return test_loss, test_preds, targets, round(test_accuracy, 4), round(test_time, 4)


def linear_test_ms2ca(net, super_head, criterion2, data_loader, args, groundTruth=None, visulation=False):
    net.eval()
    super_head.eval()
    
    test_losses = []
    test_preds = []
    targets = []
    correct = 0

    with torch.no_grad():
        for S_11, S_12, S_2, target in data_loader:
        # for data, _, target in data_loader:
            target = target - 1
            S_11 = S_11.to(args.device)
            S_12  = S_12.to(args.device)
            S_2 = S_2.to(args.device)
            target = target.to(args.device)
                
            s_out_11, s_out_21 = net(S_11, S_2)
            s_out_12, s_out_22 = net(S_12, S_2)
            s_out_11, s_out_21, s_out_31 = super_head(s_out_11, s_out_21)
            s_out_12, s_out_22, s_out_32 = super_head(s_out_12, s_out_22)
            s_out = s_out_11 + s_out_21 + s_out_31 + s_out_12 + s_out_22 + s_out_32

            test_loss1 = criterion2(s_out_11, target).item()
            test_loss2 = criterion2(s_out_21, target).item()
            test_loss3 = criterion2(s_out_31, target).item()
            test_loss4 = criterion2(s_out_12, target).item()
            test_loss5 = criterion2(s_out_22, target).item()
            test_loss6 = criterion2(s_out_32, target).item()
            test_loss = test_loss1 + test_loss2 + test_loss3 + test_loss4 + test_loss5 + test_loss6
            test_pred = s_out.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(s_out, dim=1)  # 这一行和上面的实现效果是一样的
            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = (100. * correct / len(data_loader.dataset)).item()
        # print('Accuracy: {}/{} ({:.2f}%)\n'.format(
        #             correct, len(data_loader.dataset), test_accuracy))
        
    if visulation and groundTruth.any() != None:
        hight, width = groundTruth.shape
        test_preds = torch.cat(test_preds, dim=0).numpy()
        predict_labels = test_preds.reshape(hight, width)

        # print(np.unique(predict_labels))
        draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy, 4)) + "_knn_full"))
        # 背景像元置为 0，因为 pred 预测了所有的像元，但是背景像元并不需要画出来
        for i in range(hight):
            for j in range(width):
                if groundTruth[i][j] == 0:
                    predict_labels[i][j] = 0

        draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy, 4)) + "_knn_label"))    

    return test_losses, test_preds, correct, targets, round(test_accuracy, 4)


def linear_test_ms2ca2(net, super_head, criterion2, data_loader, args, groundTruth=None, visulation=False):
    net.eval()
    super_head.eval()
    
    test_losses = []
    test_preds = []
    targets = []
    correct = 0

    with torch.no_grad():
        for S_11, S_2, target in data_loader:
        # for data, _, target in data_loader:
            target = target - 1
            S_11 = S_11.to(args.device)
            S_2 = S_2.to(args.device)
            target = target.to(args.device)
                
            s_out_11, s_out_21, _ = net(S_11, S_2)
            s_out_1, s_out_2, s_out_3 = super_head(s_out_11, s_out_21)
            s_out = s_out_1 + s_out_2 + s_out_3

            test_loss = criterion2(s_out, target).item()
            test_pred = s_out.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(s_out, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = (100. * correct / len(data_loader.dataset)).item()
        # print('Accuracy: {}/{} ({:.2f}%)\n'.format(
        #             correct, len(data_loader.dataset), test_accuracy))
        
    if visulation and groundTruth.any() != None:
        hight, width = groundTruth.shape
        test_preds = torch.cat(test_preds, dim=0).numpy()
        predict_labels = test_preds.reshape(hight, width)

        # print(np.unique(predict_labels))
        draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy, 4)) + "_knn_full"))
        # 背景像元置为 0，因为 pred 预测了所有的像元，但是背景像元并不需要画出来
        for i in range(hight):
            for j in range(width):
                if groundTruth[i][j] == 0:
                    predict_labels[i][j] = 0

        draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy, 4)) + "_knn_label"))    

    return test_losses, test_preds, correct, targets, round(test_accuracy, 4)


def linear_test_mivit(net, super_head, criterion2, data_loader, args, groundTruth=None, visulation=False):
    net.eval()
    super_head.eval()
    
    test_losses = []
    test_preds = []
    targets = []
    correct = 0

    with torch.no_grad():
        for s_data11, s_data12, s_data13, s_data21, s_data22, s_data23, target in data_loader:
        # for data, _, target in data_loader:
            target = target - 1
            s_data11 = s_data11.to(args.device)
            s_data12 = s_data12.to(args.device)
            s_data13 = s_data13.to(args.device)
            s_data21 = s_data21.to(args.device)
            s_data22 = s_data22.to(args.device)
            s_data23 = s_data23.to(args.device)
            target = target.to(args.device)
                
            s_fuse1, s_fuse2 = net(s_data11, s_data21, s_data12, s_data22, s_data13, s_data23)
            s_out = super_head(s_fuse1, s_fuse2)

            test_loss = criterion2(s_out, target).item()
            test_pred = s_out.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(s_out, dim=1)  # 这一行和上面的实现效果是一样的

            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            test_preds.append(test_pred.cpu())
            test_losses.append(test_loss)
            targets.append(target.cpu())

        test_accuracy = (100. * correct / len(data_loader.dataset)).item()
        # print('Accuracy: {}/{} ({:.2f}%)\n'.format(
        #             correct, len(data_loader.dataset), test_accuracy))
        
    if visulation and groundTruth.any() != None:
        hight, width = groundTruth.shape
        test_preds = torch.cat(test_preds, dim=0).numpy()
        predict_labels = test_preds.reshape(hight, width)

        # print(np.unique(predict_labels))
        draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy, 4)) + "_knn_full"))
        # 背景像元置为 0，因为 pred 预测了所有的像元，但是背景像元并不需要画出来
        for i in range(hight):
            for j in range(width):
                if groundTruth[i][j] == 0:
                    predict_labels[i][j] = 0

        draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy, 4)) + "_knn_label"))    

    return test_losses, test_preds, correct, targets, round(test_accuracy, 4)




def get_results(test_preds, targets):
    y_pred_test = [j for i in test_preds for j in i]
    y_targets = [j for i in targets for j in i]
    classification = classification_report(y_targets, y_pred_test, digits=4)
    kappa = cohen_kappa_score(y_targets, y_pred_test)
    # print(classification, kappa)
    return classification, kappa


def draw(label, name, scale: float = 4.0, dpi: int = 400, save_img=True):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_img:
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)