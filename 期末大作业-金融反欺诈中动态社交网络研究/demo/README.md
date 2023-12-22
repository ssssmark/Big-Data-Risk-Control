# 数据风控大作业demo
提供使用GCN的demo一个，请在比赛网站上的数据集中下载"data.npz"放到路径'./data/xydata/raw'中。 demo代码中对"data.npz"的train_mask，随机按照6/4的比例将其划分为train/valid dataset。


## Environments
Implementing environment:  
- python = 3.7.6
- numpy = 1.21.2  
- pytorch = 1.6.0  
- torch_geometric = 1.7.2  
- torch_scatter = 2.0.8  
- torch_sparse = 0.6.9  

- GPU: Tesla V100 32G  

## Training

- **GCN**
```bash
python train.py --model gcn  --epochs 200 --device 0
python inference.py --model gcn --device 0
```
