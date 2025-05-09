# 使用 LoRAMoE 微调 Llama2


这是论文 LoRAMoE 的源码实现，论文地址： [LoRAMoE: Revolutionizing Mixture of Experts for Maintaining World Knowledge in Language Model Alignment](https://arxiv.org/abs/2312.09979) 

![Overview of LoRAMoE](image.png)
## 执行

### 安装 python 依赖
您可以使用以下命令快速导出环境：
```bash
conda env create -f environment.yml
```
或者
```bash
conda create -n loramoe python=3.10 -y

pip install -r requirements.txt
```
我们*不安装* `peft` 以避免与本地 `peft` 软件包冲突。

### 配置 CUDA 11.8
检查是否安装 CUDA 11.8:
```bash
ls /usr/local/ | grep cuda
```
如果未找到 `cuda-11.8`，可以前往 [NVIDIA](https://developer.nvidia.com/cuda-11-8-0-download-archive) 下载并安装对应版本。

无需卸载已有的 CUDA 版本。NVIDIA 官方支持多版本 CUDA 并存。只需正确配置环境变量（如 CUDA_HOME 和 PATH），即可灵活切换不同版本的 CUDA，用于编译或运行不同的项目。

### 配置 wandb
1. 注册 wandb 账号：访问 [wandb官网](https://wandb.ai) 注册一个账号。  
2. 登录 wandb：在终端运行以下命令并输入 API 密钥（可以在 [wandb API 密钥页面](https://wandb.ai/authorize) 找到）：  
```bash
wandb login
```

## 用法

### Data Format

We construct a tiny dataset to demonstrate the data format during the training and inference phase and evaluate the correct of code.

```
data/
|--tiny_data/
  |--train/train.json
  |--test.json
```


### Train LoRAMoE on Single Node
```bash
bash run_loramoe.sh
```

### Explanations of Hyper-parameters


| blc weight    | blc alpha | LoRA rank     | LoRA alpha | LoRA trainable |LoRA dropout |LoRA num |
|---------------|---------------|---------------|------------|----------------|---------------| --------|
| the strength of localized balance constraints |degree of imbalance | rank of LoRA experts | LoRA scale  | where the LoRA layers are added | dropout rate in LoRA|number of experts|

### 超参数
* --nnodes：表示分布式训练中使用的节点数（机器数量）。这里设置为 1，表示只使用一台机器进行训练。
* --nproc_per_node： 表示每个节点（机器）上启动的进程数。这里设置为 8，通常对应于一台机器上使用的 GPU 数量，即每个 GPU 启动一个进程。 



## Note: Our main changes to `transformers` and `peft`

In `transformers`, we mainly change `modeling_llama.py` to introduce new para `task_types`.

In `peft`, we replace the original LoRA class with the mixtures of experts architecture.

## How to Evaluate
We use [opencompass](https://github.com/open-compass/opencompass/tree/main) for evaluation. To run LoRAMoE on opencompass:

- In `opencompass/opencompass/models/huggingface.py`, add: 
```python
import sys
sys.path.insert(0, 'path_to_your_current_dir_containing_changed_peft&transformers')
```
- In the config file
```python
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='',
        path="path_to_base_model",
        tokenizer_path='path_to_tokenizer',
        peft_path='path_to_loramoe',
        ...
    )
]
```


## Citation
If you find this useful in your research, please consider citing
```
@misc{dou2024loramoe,
      title={LoRAMoE: Revolutionizing Mixture of Experts for Maintaining World Knowledge in Language Model Alignment}, 
      author={Shihan Dou and Enyu Zhou and Yan Liu and Songyang Gao and Jun Zhao and Wei Shen and Yuhao Zhou and Zhiheng Xi and Xiao Wang and Xiaoran Fan and Shiliang Pu and Jiang Zhu and Rui Zheng and Tao Gui and Qi Zhang and Xuanjing Huang},
      year={2023},
      eprint={2312.09979},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
