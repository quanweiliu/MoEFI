models:



resNet1: 单分支版本，来源于 ACNet。
resNet2：多分支版本，来源于 ACNet。将 resNet1 改成多分支版本。
resNet3：多尺度的多分支版本，来源于 MIViT。


main_D3 -- 04-11-11-13-cnn_D3: 第一个正式版本。按照 KnowCL 的模式来的。可以兼容 resNet 和 ms2
DataAugmentationDINO: 带有 resized 的数据增强。
HyperX：分割高光谱图像
resNet2.py
data_pipe.py: 版本 1，参考 KnowCl
split_data.py: 接收 data11, data12, data2
heads.Conv_head
trainer.train_cnn2
tester.linear_test_cnn2


main_D32 -- 04-11-11-19-MS2CANet_D3: 第一个正式版本，可以兼容 resNet2 和 ms2
DataAugmentationDINO
HyperX2：不分割高光谱图像
resNet2.py
data_pipe2.py 版本 2，不要切一半的数据增强了，直接用随机扰动数据增强
split_data2.py：接收 data11, data2, gt，依旧是高光谱图像不分割
heads.MS2_head
trainer.train_ms2ca2
tester.linear_test_ms2ca2


main_D33 
-- 04-11-11-45-MiViT_D3: 将 MIViT 的 多尺度encoder 并入到我们的框架中。
-- 04-11-12-00-mine_combine_D3 将 main_D32 和 main_D33 结合。其实只是将 main_D33 的前半部分的多尺度加上 main_D32 后半部分的注意力。注意这个文件的学习率， batch size 等超参数的变化。
-- 04-11-12-38-mine_gate_D3 gate 版本单分支融合策略
-- 04-11-12-46-mine_combine_gate_D3合并 main_D35 和 main_D34，得到三分支版本。


main_D34
-- 04-11-13-01-mine_D3 加上第三个分支，互信息模块。互信息模块在前半部分。
-- 04-11-13-11-mine2_D3 互信息模块在后半部分。

main_D39 
-- 加上自蒸馏模块。调整训练次数和学习率，集大成。


--------------------------------上面的代码是一直添加组件，下面的代码是一直替换和删除组件。-------------------------------------------------------



main_D310
-- mine_cnn
-- 把 MIVIT 模块换成 Mamba / VIT / CNN 等经典深层的模块。
-- 我要做的是将 main_D32 嫁接到 main_D39 上去。
-- 模型级别的扰动 / dropblock and dropout

main_D311
-- mine_cnn_gate 去掉两个分支上的无用的模块。不要预训练模块精度变好了？？？





main_D89 
-- 更改数据集。





discussion
- 不分割的精度和分割的精度差异好大？？？？ 待定。


discussion
- 对比学习是否真的有效！


下面的任务主要就是怎么提升基础网络的精度了。
discussion
- 改了网络确实提升了精度。





