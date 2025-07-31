import torch
import numpy as np
import random


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data11, data12, data2, gt, transform, patch_size=5, remove_zero_labels=True):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data11 = data11
        self.data12 = data12
        self.data2 = data2
        self.label = gt
        self.transform = transform
        self.patch_size = patch_size
        self.ignored_labels = set()
        self.center_pixel = True
        self.remove_zero_labels = remove_zero_labels
    
        # print(supervision)
        mask = np.ones_like(gt)
        # print("mask", mask.shape) 
        
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2

        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x >= p and x < data11.shape[0] - p and y >= p and y < data11.shape[1] - p
            ]
        )

        self.labels = [self.label[x, y] for x, y in self.indices]

        if self.remove_zero_labels:
            self.indices = np.array(self.indices)
            self.labels = np.array(self.labels)

            self.indices = self.indices[self.labels>0]
            self.labels = self.labels[self.labels>0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        '''
            x, y -> index
            x1, y1 = x - 4, y - 4
            x2, y2 = x, y
        '''
        x, y = self.indices[index]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data11 = self.data11[x1:x2, y1:y2].transpose((2, 0, 1))
        data12 = self.data12[x1:x2, y1:y2].transpose((2, 0, 1))
        data2 = self.data2[x1:x2, y1:y2].transpose((2, 0, 1))
        label = self.label[x1:x2, y1:y2]

        # # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        # data11 = np.asarray(np.copy(data11).transpose((2, 0, 1)), dtype="float32")
        # data12 = np.asarray(np.copy(data12).transpose((2, 0, 1)), dtype="float32")
        # data2 = np.asarray(np.copy(data2).transpose((2, 0, 1)), dtype="float32")
        # label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data11 = torch.from_numpy(data11)
        data12 = torch.from_numpy(data12)
        data2 = torch.from_numpy(data2)
        label = torch.from_numpy(label)
        # print(data11.shape)
        # print(data12.shape)
        # print(data2.shape)
        # print(label.shape)

        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        
        if self.transform != None:
            # print("transformed", )
            data11 = self.transform(data11)
            data12 = self.transform(data12)
            data21 = self.transform(data2)
            data22 = self.transform(data2)

            return data11, data12, data21, data22, label
        
        else:
            
            return data11, data12, data2, label


class HyperX2(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data1, data2, gt, transform, patch_size=5, remove_zero_labels=True):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX2, self).__init__()
        self.data1 = data1
        self.data2 = data2
        self.label = gt
        self.transform = transform
        self.patch_size = patch_size
        self.ignored_labels = set()
        self.center_pixel = True
        self.remove_zero_labels = remove_zero_labels
    
        # print(supervision)
        mask = np.ones_like(gt)
        # print("mask", mask.shape) 
        
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2

        self.indices = np.array(
            [
                (x, y) for x, y in zip(x_pos, y_pos)
                if x > p and x < data1.shape[0] - p - 1 and y > p and y < data1.shape[1] - p - 1
            ]
        )

        self.labels = [self.label[x, y] for x, y in self.indices]

        if self.remove_zero_labels:
            self.indices = np.array(self.indices)
            self.labels = np.array(self.labels)

            self.indices = self.indices[self.labels>0]
            self.labels = self.labels[self.labels>0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        '''
            x, y -> index
            x1, y1 = x - 4, y - 4
            x2, y2 = x, y
        '''
        x, y = self.indices[index]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data1 = self.data1[x1:x2, y1:y2].transpose((2, 0, 1))
        data2 = self.data2[x1:x2, y1:y2].transpose((2, 0, 1))
        label = self.label[x1:x2, y1:y2]

        # Load the data into PyTorch tensors
        data1 = torch.from_numpy(data1)
        data2 = torch.from_numpy(data2)
        label = torch.from_numpy(label)

        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]

        # 随机选另一个 index 构造 x_pair
        if self.transform is not None:
            # 随机采样不同 index 构造 x_pair
            rand_idx = random.randint(0, len(self.indices) - 1)
            x_p, y_p = self.indices[rand_idx]
            xp1, yp1 = x_p - self.patch_size // 2, y_p - self.patch_size // 2
            xp2, yp2 = xp1 + self.patch_size, yp1 + self.patch_size
            data_pair = self.data[xp1:xp2, yp1:yp2].transpose((2, 0, 1))
            data_pair = torch.from_numpy(data_pair)
            data11 = self.transform(data1, data_pair)
            data12 = self.transform(data1, data_pair)

            data21 = self.transform(data2)
            data22 = self.transform(data2)

            return data11, data12, data21, data22, label
        else:
            return data1, data2, label
        

# 单模态，多模态，多尺度
class HyperX3(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data1, gt, transform, patch_size=5, data2=None, remove_zero_labels=True):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
            mixture_augmentation  不能用
        """
        super(HyperX3, self).__init__()
        self.data1 = data1
        self.data2 = data2
        self.label = gt
        self.transform = transform
        self.patch_size = patch_size
        self.patch_sizeX2 = patch_size * 2
        self.patch_sizeX3 = patch_size * 3
        self.ignored_labels = set()
        self.center_pixel = True
        self.remove_zero_labels = remove_zero_labels
    
        mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_sizeX3 // 2

        self.indices = np.array(
            [
                (x, y) for x, y in zip(x_pos, y_pos)
                # if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
                if x >= p and x < data1.shape[0] - p and y >= p and y < data1.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]

        # remove zero labels, 这里删除是通过 self.indices 删除的，不是通过 self.labels 删除的
        if self.remove_zero_labels:
            self.indices = np.array(self.indices)
            self.labels = np.array(self.labels)

            self.indices = self.indices[self.labels>0]
            self.labels = self.labels[self.labels>0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        '''
            x, y -> index
            x1, y1 = x - 4, y - 4
            x2, y2 = x, y
        '''
        x, y = self.indices[index]
        # print("self.patch_size", self.patch_size)
        x11, y11 = x - self.patch_size // 2, y - self.patch_size // 2
        x12, y12 = x - self.patch_sizeX2 // 2, y - self.patch_sizeX2 // 2
        x13, y13 = x - self.patch_sizeX3 // 2, y - self.patch_sizeX3 // 2
        x21, y21 = x11 + self.patch_size, y11 + self.patch_size
        x22, y22 = x12 + self.patch_sizeX2, y12 + self.patch_sizeX2
        x23, y23 = x13 + self.patch_sizeX3, y13 + self.patch_sizeX3
        # print("self.patch_size", x11, x21, y11, y21)

        data11 = self.data1[x11:x21, y11:y21].transpose((2, 0, 1))
        data12 = self.data1[x12:x22, y12:y22].transpose((2, 0, 1))
        data13 = self.data1[x13:x23, y13:y23].transpose((2, 0, 1))
        if isinstance(self.data2, np.ndarray):
            data21 = self.data2[x11:x21, y11:y21].transpose((2, 0, 1))
            data22 = self.data2[x12:x22, y12:y22].transpose((2, 0, 1))
            data23 = self.data2[x13:x23, y13:y23].transpose((2, 0, 1))
        label = self.label[x11:x21, y11:y21]


        # Load the data into PyTorch tensors
        data11 = torch.from_numpy(data11)
        data12 = torch.from_numpy(data12)
        data13 = torch.from_numpy(data13)
        if isinstance(self.data2, np.ndarray):
            data21 = torch.from_numpy(data21)
            data22 = torch.from_numpy(data22)
            data23 = torch.from_numpy(data23)
        label = torch.from_numpy(label)
        # label2 = torch.from_numpy(label2)
        # label3 = torch.from_numpy(label3)

        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
            # label2 = label2[self.patch_sizeX2 // 2, self.patch_sizeX2 // 2]
            # label3 = label3[self.patch_sizeX3 // 2, self.patch_sizeX3 // 2]
        
        if self.transform != None:
            # print("transformed", )
            data111 = self.transform(data11)
            data112 = self.transform(data11)
            data121 = self.transform(data12)
            data122 = self.transform(data12)
            data131 = self.transform(data13)
            data132 = self.transform(data13)
            if isinstance(self.data2, np.ndarray):
                data211 = self.transform(data21)
                data212 = self.transform(data21)
                data221 = self.transform(data22)
                data222 = self.transform(data22)
                data231 = self.transform(data23)
                data232 = self.transform(data23)

            if isinstance(self.data2, np.ndarray):
                return data111, data112, data121, data122, data131, data132, data211, data212, data221, data222, data231, data232, label
            else:
                return data111, data112, data121, data122, data131, data132, label
        else:
            if isinstance(self.data2, np.ndarray):
                return data11, data12, data13, data21, data22, data23, label
            else:
                return data11, data12, data13, label
            

def sample_gt(gt, train_num=50, train_ratio=0.1, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    # print("test_gt", test_gt.shape)

    if mode == 'number':
        print("split_type: ", mode, "\ntrain_number: ", train_num)
        sample_num = train_num
        for c in np.unique(gt):
            if c == 0:
              continue
            indices = np.nonzero(gt == c)
            X = list(zip(*indices)) 
            y = gt[indices].ravel()  
            np.random.shuffle(X)

            max_index = np.max(len(y)) + 1
            if sample_num > max_index:
                sample_num = 15
            else:
                sample_num = train_num

            train_indices = X[: sample_num]
            test_indices = X[sample_num:]

            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]

            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

    # elif mode == 'ratio':
    #     print("split_type: ", mode, "\ntrain_ratio: ", train_ratio)
    #     for c in np.unique(gt):
    #         if c == 0:
    #           continue
    #         indices = np.nonzero(gt == c)
    #         X = list(zip(*indices)) 
    #         y = gt[indices].ravel()   
    #         np.random.shuffle(X)

    #         train_num = np.ceil(train_ratio * len(y)).astype('int')
    #         # print(train_num)

    #         train_indices = X[: train_num]
    #         test_indices = X[train_num:]
            
    #         train_indices = [list(t) for t in zip(*train_indices)]
    #         test_indices = [list(t) for t in zip(*test_indices)]
    #         # print("test_indices", test_indices)

    #         train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    #         test_gt[tuple(test_indices)] = gt[tuple(test_indices)]


    elif mode == 'ratio':
            # unique_classes = np.unique(gt)
            # unique_classes = unique_classes[unique_classes != 0]  # skip background (0)

            train_coords = []
            test_coords = []

            for c in np.unique(gt):
                class_coords = list(zip(*np.where(gt == c)))
                n_total = len(class_coords)
                n_train = int(np.round(train_ratio * n_total))
                # random.seed(23)
                # random.shuffle(class_coords)
                train_coords.extend(class_coords[:n_train])
                test_coords.extend(class_coords[n_train:])

            train_indices = [list(t) for t in zip(*train_coords)]
            test_indices = [list(t) for t in zip(*test_coords)]

            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            # train_label = [gt[x, y] for x, y in train_coords]
            # test_label = [gt[x, y] for x, y in test_coords]

            # train_set = np.column_stack((train_indices[0], train_indices[1], train_label))
            # test_set = np.column_stack((test_indices[0], test_indices[1], test_label))

    # elif mode == 'disjoint':
    #     print("split_type: ", mode, "\ntrain_ratio: ", train_ratio)
    #     train_gt = np.copy(gt)
    #     test_gt = np.copy(gt)
    #     for c in np.unique(gt):
    #         mask = gt == c
    #         for x in range(gt.shape[0]):
    #             # numpy.count_nonzero 是用于统计数组中非零元素的个数
    #             first_half_count = np.count_nonzero(mask[:x, :])
    #             second_half_count = np.count_nonzero(mask[x:, :])
    #             try:
    #                 ratio = first_half_count / (first_half_count + second_half_count)
    #                 if ratio >= train_ratio:
    #                     break
    #             except ZeroDivisionError:
    #                 continue
    #         mask[:x, :] = 0
    #         train_gt[mask] = 0
    #     test_gt[train_gt > 0] = 0


    elif mode == 'disjoint':
        print("split_type: ", mode, "\ntrain_ratio: ", train_ratio)
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        
        for c in np.unique(gt):
            if c == 0:
                continue  # 忽略背景类
            mask = gt == c
            total = np.count_nonzero(mask)
            
            if total < 2:
                print(f"[Warning] Class {c} has less than 2 samples. Skipping.")
                train_gt[mask] = 0
                test_gt[mask] = 0
                continue
            
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = total - first_half_count
                try:
                    ratio = first_half_count / total
                    if ratio >= train_ratio:
                        break
                except ZeroDivisionError:
                    continue
            
            # 如果划分后测试集没有样本，调整分割点以保留至少一个测试样本
            if second_half_count == 0:
                x = max(1, x - 1)  # 回退一行以保证测试集不为空
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = total - first_half_count

            # 再检查：如果train或test都为空，就跳过这个类
            if first_half_count == 0 or second_half_count == 0:
                print(f"[Warning] Class {c} cannot be split properly. Skipping.")
                train_gt[mask] = 0
                test_gt[mask] = 0
                continue

            # 应用分割：保留上半部分为训练，其余为测试
            mask[:x, :] = 0  # 下半部分保留
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0  # 删除测试集中与训练集重复的部分

    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))

    return train_gt, test_gt






















































