# 类案检索实验报告

## 计81 肖光烜 2018011271

## 问题描述

本项目中的类案检索的问题本质上是非对称语义相似度对比，难点在于：

- 输入文本对长度长
- 信息噪音多
- 需要细粒度衡量相似度
- 相似度需要根据语义（甚至逻辑）判断

针对以上难点，我想出了以下解决方案。

## 模型选择

采用`Lawformer`、`RoBERTa-wwm-ext`和`RoBERTa-wwm-ext-large`模型。

使用Lawformer模型，架构为Siamese-BERT（下图中的Bi-Encoder架构），即将query和candidate输入同一个模型，pooling得到sentence embedding后求余弦相似度，回归到标签上。query和candidate均截断前512个token。

![Bi_vs_Cross-Encoder.png](assets/6570a53babc4fe7274569d5b33e41b81f3a20ea7.png)

使用RoBERTa时，架构采用Cross Encoder，即将query和candidate拼接起来，提取[CLS]向量后将数值回归到0-1之间。query和candidate均截断前256个token。

其余组合我试过，效果都不好。

标签处理上，当使用单模型回归时，我将标签转化为浮点数并归一化到0-1区间上（除以三），Bi-Encoder采用MSE Loss，Cross-Encoder采用BCE With Logits Loss。

## 多阶段召回+重排

直接让模型回归得到4个等级的分数是很困难的，我想到可以训练3个模型，做多阶段召回和重排。具体而言：三个模型做的都是二分类任务，第一个模型负责区分0与1、2、3，第二个模型用于区分0、1与2、3，第三个模型用于区分0、1、2与3，三个模型依次擅长粗排序、细排序和精排序。在测试时，先通过第一个模型得到粗排，取其中阳性（score大于0.5）的candidate，通过第二个模型，再取其中阳性的样本通过第三个模型，得到最终的排序结果。

## 集成学习

一种常见的涨点手段是集成多个模型的输出，这里有两种方案：

- 将多个模型给candidate的打分做平均后，再排序。（score ensemble）
- 将多个模型给candidate的排序做平均。（rank ensemble）

## 数据处理

query前加上罪名，比如一个查询是盗窃罪案子，query的输入文本就是：

> 盗窃罪。案件具体情况...

candidate输入文本为：案件名称+罪名+案件基本过程，其中罪名通过正则表达式匹配，

```python
crime_pattern = re.compile(r'已构成(.*?)罪')
```

例如：

```json
{
    "ajId": "3ad97045-7fef-4fc8-8761-c3b95d566ae9",
    "ajName": "赵平、杨某走私、贩卖、运输、制造毒品罪一案",
    "ajjbqk": " 南京市秦淮区人民检察院指控，2018年7月至8月间，被告人赵平在昆明市普吉路明日城市小区等地... ",
}
```

输入文本为：

> 赵平、杨某走私、贩卖、运输、制造毒品罪一案。走私、贩卖、运输、制造毒品罪。南京市秦淮区人民检察院指控，2018年7月至8月间，被告人赵平在昆明市普吉路明日城市小区等地...

## 数据集划分

开发用训练集train_dev与验证集valid比例为188:19，我选取在验证集上效果最好的超参数，随后将同样配置的模型在全部数据上（197条输入）训练，得到提交用的结果。

## 结果汇报

| Model                                  | Test Score         |
| -------------------------------------- | ------------------ |
| roberta-cross (single)                 | 0.9201641113548711 |
| roberta-cross (score ensemble)         | 0.9285080642964105 |
| roberta-cross 3-stage (single)         | 0.8988760595023486 |
| roberta-cross 3-stage (score ensemble) | 0.9128759853084087 |
| roberta-large-cross (single)           | 0.9306393135753759 |
| roberta-large-cross (score ensemble)   | 0.9320346823007378 |
| lawformer-bi (single)                  | 0.914216795605993  |
| lawformer-bi (score ensemble)          | 0.9196956972018997 |
| lawformer-bi 3-stage (single)          | 0.9161835168931821 |
| lawformer-bi 3-stage (rank ensemble)   | 0.9202860293400528 |
| lawformer-bi 3-stage (score ensemble)  | 0.9229117610846013 |

可见在模型方面，有明显的roberta-large > roberta-base > lawformer的关系存在。roberta-large与分数集成法获得了最高的测试分数，roberta-large有最好的单模型表现。

多阶段召回+重排在lawformer上有明显的提升性能效果，然而在roberta上没有提升效果，原因可能在于单阶段的roberta就能够学到较好的相似度排序关系，而将多阶段级联起来反而降低了结果的鲁棒性。

集成方面，我仅在lawformer多阶段的设置下尝试了排序集成法，效果是差于分数集成法的。还可以观察到，单模型效果越好的方法，集成的效果就越小，这是因为集成学习本身就是对多个弱学习器结果进行聚合时有更明显的提升效果，对于本身就较强的学习器提升效果就不大了。

## 实现细节与运行方法

## 文件说明

```bash
./├── baseline
│   ├── main.py
│   ├── run.sh
│   └── stopword.txt
├── config
│   ├── basic-train.yml
│   ├── retriever1-bi-train.yml
│   ├── retriever2-bi-train.yml
│   ├── retriever3-bi-train.yml
│   ├── roberta-256-256-cross.yml
│   ├── roberta-large-256-256-cross.yml
├── data
│   ├── origin
│   │   ├── test
│   │   │   └── processed
│   │   │       ├── candidate_ridxs.pt
│   │   │       ├── edge_graph_ids.pt
│   │   │       ├── edges.pt
│   │   │       ├── inputs.pt
│   │   │       ├── node_graph_ids.pt
│   │   │       └── query_ridxs.pt
│   │   ├── train
│   │   │   └── processed
│   │   │       ├── candidate_ridxs.pt
│   │   │       ├── edge_graph_ids.pt
│   │   │       ├── edges.pt
│   │   │       ├── inputs.pt
│   │   │       ├── labels.pt
│   │   │       ├── node_graph_ids.pt
│   │   │       └── query_ridxs.pt
│   │   ├── train_dev
│   │   │   └── processed
│   │   │       ├── candidate_ridxs.pt
│   │   │       ├── edge_graph_ids.pt
│   │   │       ├── edges.pt
│   │   │       ├── inputs.pt
│   │   │       ├── labels.pt
│   │   │       ├── node_graph_ids.pt
│   │   │       └── query_ridxs.pt
│   │   └── val
│   │       └── processed
│   │           ├── candidate_ridxs.pt
│   │           ├── edge_graph_ids.pt
│   │           ├── edges.pt
│   │           ├── inputs.pt
│   │           ├── labels.pt
│   │           ├── node_graph_ids.pt
│   │           └── query_ridxs.pt
│   ├── startwords.txt
├── environment.yml
├── models
│   ├── chinese-roberta-wwm-ext
│   │   ├── flax_model.msgpack
│   │   ├── pytorch_model.bin
│   │   ├── README.md
│   │   ├── tf_model.h5
│   │   └── vocab.txt
│   ├── chinese-roberta-wwm-ext-large
│   │   ├── flax_model.msgpack
│   │   ├── pytorch_model.bin
│   │   ├── README.md
│   │   ├── tf_model.h5
│   │   └── vocab.txt
│   └── Lawformer
│       ├── pytorch_model.bin
│       ├── README.md
│       └── vocab.txt
├── notebook
│   ├── bm25.ipynb
│   ├── data.ipynb
│   ├── preprocess.ipynb
│   ├── preprocess_summary.ipynb
│   ├── summarization.ipynb
│   ├── summary.py
│   ├── symmetry.ipynb
│   └── tfidf.ipynb
├── predictions
├── README.md
├── scripts
│   ├── data_preprocessing.py
│   └── setup-env.sh
└── src
    ├── agg_results.py
    ├── data.py
    ├── ensemble.py
    ├── evaluator.py
    ├── grid.py
    ├── logger.py
    ├── loss.py
    ├── main.py
    ├── merge_stages.py
    ├── model.py
    ├── parallel.py
```

其中`data`、`models`目录由于过大没有附上，存储的是实验数据和使用的预训练模型参数，模型均可以在huggingface上下载得到。`data`中使用到的实验数据是预处理并tokenize后的二进制格式，用于加速模型训练。我将整个数据集抽象为图模型，每个query和candidate均为一个节点，每条query-candidate对作为一条连边，这本来是为我做基于图的数据增强方法准备的，但那个想法不work，这里就没写。`inputs.pt`为所有query和candidates tokenize后的特征文件，`edges.pt`记录了每个query-candidate对的下表，其余文件的含义与名字相同。

`config`为实验配置文件目录，包含复现实验结果的配置文件。

`notebook`为数据处理使用的jupyter notebook代码目录，可以运行这些代码获得处理后的数据文件。

`src`为实验代码目录，可能会被使用到的有`main.py`、`agg_results.py`、`merge_stages.py`，分别用于训练模型、集成结果与聚合多阶段召回-排序结果。

### 使用方法

#### 实验环境

所有实验均在40GB显存的A100显卡上运行得到，conda环境配置指令如下：

```bash
conda create -f environment.yml
```

#### 数据预处理

将训练数据解压放到`data/origin`下，运行`notebook/preprocess.ipynb`获得`data/origin/*/processed`的所有预处理好的数据。

#### 模型训练

运行

```bash
cd src
python main.py --config ../config/CONFIG_FILE
```

就可以运行CONFIG_FILE对应的实验，比如`python main.py --config ../config/roberta-large-256-256-cross.yml`可训练`roberta-large`模型，在`log/roberta-large-256-256-cross-train`下得到这次实验的日志、结果、模型参数和预测文件。

#### 多阶段召回-排序

运行

```bash
cd src
python merge_stages.py
```

就可以将三个阶段的预测评分分层汇总得到一个整体的结果，保存在`predictions`文件夹中，这个代码里面还包括了分数集成和排序集成两个功能。

#### 结果集成

运行

```
cd src
python agg_results.py --log-dir LOG_DIR
```

就可以将`LOG_DIR`中多个随机种子训练得到的模型预测分数进行平均化后，得到一个集成的预测结果，保存在`LOG_DIR/agg_results.json`中。

## 一个未成功（但我感觉很可能成功）的方法

我最开始看见这个题目感觉一个自然的思路是数据增强，因为专家标注了一些案例之间的**等价性**，这里的如果能够充分利用将可以把数据扩增数倍，但这要求query与candidate本身的对称性。具体来说，**如果query与candidate是完全对称的**，那么对一个等价的query-candidate对$(q_1, c_1)$，$q_1$满足的一切标注$l(q_1, \times)$，均有$l(c_1, \times)$成立。这使得我们可以将数据集扩增$\frac{1}{\lambda}$倍，其中$\lambda$为数据集中完全等价的标注占的比例。

然而这个方法的条件在于**query与candidate完全对称**，在这个数据集中可以观察到，query应该是从candidate中抽取出来的，长度远比candidate短，也只包含案件事实部分，不包含证据、辩护词等部分。因此我做了一些尝试将candidate化简为query的格式，只将案件事实部分抽出进行训练。然而应该是我数据处理的并不好，无法将query和candidate处理到完全对称，由于时间和计算资源的限制，目前我在这个想法上没能得到理想的结果，但我感觉如果数据处理再精准一些，这个扩增数据集的方法将非常有优势。

## 参考资料与声明

本项目所有思路、代码都是我本人思考和编写而成，没有与其他同学交流和参考代码。

[1] [Cross-Encoders -- Sentence-Transformers documentation
