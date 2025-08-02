import os 
import torch
import numpy as np 
import random
import spectral as spl
import matplotlib.pyplot as plt
from loadData import data_reader
from loadData.split_data import sample_gt


# only active in this file
def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data(args):
    data1, data2, GT = data_reader.load_data(args.dataset_name, path_data=args.path_data, type_data="GT")
    data1, data2, train_gt = data_reader.load_data(args.dataset_name, path_data=args.path_data, type_data="TRLabel")
    data1, data2, test_gt = data_reader.load_data(args.dataset_name, path_data=args.path_data, type_data="TSLabel")

    pad_width = (args.patch_size // 2) + 1
    img1 = np.pad(data1, ((pad_width, pad_width), (pad_width, pad_width), (0,0)), 'symmetric')
    img2 = np.pad(data2, ((pad_width, pad_width), (pad_width, pad_width), (0,0)), 'symmetric')
    # img1 = img1[:, :, pad_width:img1.shape[2]-pad_width]
    # img2 = img2[:, :, pad_width:img2.shape[2]-pad_width]
    # print(img1.shape, img2.shape)

    if args.pca:
        print("pca is used")
        img1, pca = data_reader.apply_PCA(img1, num_components=args.components)
        # img12, pca = data_reader.apply_PCA(img12, num_components=args.components)
    else:
        print("pca is not used")
    # print(img11.shape, img12.shape)
    # print(train_gt.shape, test_gt.shape)

    if args.show_gt:
        # data_reader.draw(data_gt, os.path.join(args.result_dir, args.dataset_name + "data_gt"), save_img=True)
        # data_reader.draw(train_gt, os.path.join(args.result_dir, args.dataset_name + "train_gt"), save_img=True)
        # data_reader.draw(test_gt, os.path.join(args.result_dir, args.dataset_name + "test_gt"), save_img=True)
        plt.figure(figsize=(12, 8))
        spl.imshow(classes=GT)
        plt.axis('off')  # 关闭坐标轴（等效于关闭刻度和边框）
        plt.tight_layout(pad=0)  # 去除额外空白边距
        plt.show()
        # spl.imshow(classes=train_gt)
        # spl.imshow(classes=test_gt)


    # 这个不要也能运行，但是会显著的影响精度
    data_gt = np.pad(GT, pad_width=pad_width, mode="constant", constant_values=(0))
    train_gt = np.pad(train_gt, pad_width=pad_width, mode="constant", constant_values=(0))
    test_gt = np.pad(test_gt, pad_width=pad_width, mode="constant", constant_values=(0))

    train_gt, val_gt = sample_gt(train_gt, train_num=args.train_num, 
                            train_ratio=args.train_ratio, mode=args.split_type)
    test_gt, _ = sample_gt(test_gt, train_num=args.train_num, 
                            train_ratio=1, mode=args.split_type)   
    # print("train_gt", train_gt.shape, "test_gt", test_gt.shape)
    
    if args.print_data_info:
        print("print_data_info : ---->")
        data_reader.data_info(train_gt, val_gt, test_gt, start=args.data_info_start)

    return img1, img2, train_gt, val_gt, test_gt, data_gt, GT
