import os 
import torch
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,cohen_kappa_score


def visulization(net, super_head, data_loader, groundTruth, args):
    net.eval()

    super_head.eval()
    test_preds = []
    targets = []
    correct = 0

    with torch.no_grad():
        for S_11, S_2, target in data_loader:
            target = target - 1
            S_11 = S_11.to(args.device)
            S_2 = S_2.to(args.device)
            target = target.to(args.device)
                
            s_out1, s_out2, s_outf = net(S_11, S_2)
            s_out = super_head(s_outf)

            pred = s_out.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(s_out, dim=1)  # 这一行和上面的实现效果是一样的

            correct += pred.eq(target.data.view_as(pred)).sum()
            test_preds.append(pred.cpu())
            targets.append(target.cpu())

        test_accuracy = (100. * correct / len(data_loader.dataset)).item()

        hight, width = groundTruth.shape
        test_preds = torch.cat(test_preds, dim=0).numpy() + 1
        predict_labels = test_preds.reshape(hight, width)

        draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy, 4)) + "_full"))
        # for i in range(hight):
        #     for j in range(width):
        #         if groundTruth[i][j] == 0:
        #             predict_labels[i][j] = 0

        # draw(predict_labels, os.path.join(args.result_dir, str(round(test_accuracy, 4)) + "_label")) 

        # savemat(os.path.join(args.result_dir, args.dataset_name + "_gt.mat"), \
        #         {args.dataset_name + '_gt': predict_labels})

def test_visulization(net, super_head, data_loader, args, groundTruth):
    net.eval()
    super_head.eval()
    
    test_preds = []

    with torch.no_grad():
        for S_1, S_2, target in data_loader:
        # for data, _, target in data_loader:
            target = target - 1
            S_1 = S_1.to(args.device)
            S_2 = S_2.to(args.device)
            target = target.to(args.device)
                
            s_base1, s_base2, s_fuse = net(S_1, S_2)
            s_out = super_head(s_fuse)

            test_pred = s_out.data.max(1, keepdim=True)[1]
            # test_pred = torch.argmax(s_out, dim=1)  # 这一行和上面的实现效果是一样的
            test_preds.append(test_pred.cpu())

        hight, width = groundTruth.shape
        test_preds = torch.cat(test_preds, dim=0).numpy()
        predict_labels = test_preds.reshape(hight, width)

        # print(np.unique(predict_labels))
        draw(predict_labels, os.path.join(args.result_dir, "full"))
        predict_labels[groundTruth == 0] = 0
        draw(predict_labels, os.path.join(args.result_dir, "pure"))    

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



