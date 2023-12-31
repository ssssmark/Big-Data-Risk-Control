反欺诈是金融领域永恒的话题，本次大作业的数据集抽样自企业不同业务时间段的数据，提供了一个全连通的社交网络有向动态图。

在本题目的图数据中，节点代表注册用户，从节点 A 指向节点 B 的有向边代表用户 A 将用户 B 填为他的紧急联系人。图中的边有不同的类型，代表了对紧急联系人的不同分类。图中的边上带有创建日期信息，边的创建日期分别脱敏成从 1 开始的正整数，时间单位为天。

数据集通过 npz 方式储存，有以下内容：

(1) x:节点特征，共 17 个

(2) y:节点共有(0,1,2,3)四类 label，其中测试样本对应的 label 被标为-100

(3) edge_index:有向边信息，其中每一行为(id_a, id_b)，代表用户 id_a 指向用户 id_b 的有向边；

(4) edge_type:边类型；

(5) train_mask：包含训练样本 id 的一维数组；

(6) test_mask：包含测试样本 id 的一维数组。

本题目的预测任务为识别欺诈用户的节点。在数据集中有四类节点，但是预测任务只需要将欺诈用户（Class 1）从正常用户（Class 0）中区分出来；这两类节点被称为前景节点。另外两类用户（Class 2和 Class 3）尽管在数目上占据更大的比例，但是他们的分类与用户是否欺诈无关，因此预测任务不包含这两类节点；这两类节点被称为背景节点。与常规的结构化数据不同，图算法可以通过研究对象之间的复杂关系来提高模型预测效果。而本题除了提供前景节点之间的社交关系，还提供了大量的背景节点。可以充分挖掘各类用户之间的关联和影响力，提出可拓展、高效的图神经网络模型，将隐藏在正常用户中的欺诈用户识别出来。

考察内容
图网络的了解与分析、图分类任务的了解与分析、充分利用所有图节点的信息进行分类、完善代码提高在大规模数据中处理速度。

提交内容
在和鲸平台的课程作业中选择期末大作业 (Links to an external site.)，根据给出的训练集进行模型训练（data.npz中的train_mask），使用测试集（data.npz中的test_mask）预测结果并输出到result.csv，字段为index（整型）, label（浮点型）。将result.csv文件提交到和鲸平台自动评测打分。