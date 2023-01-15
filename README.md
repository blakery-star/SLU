### 创建环境

如果想使用我们的最佳模型，推荐设置如下。在本branch中同样支持高版本torch（经验证，至少支持至1.13.0版本）。
    
    conda create -n slu python=3.6
    conda activate slu
    pip install torch==1.7.1
    pip install transformers
    
如果想使用（会产生负效果的）文本纠错模块，则另外需要以下环境。
    
    pip install pycorrector
    
特别地，对Ernie4CSC，需以下环境。
    
    pip install paddlepaddle
    pip install paddlenlp
    
    
### 运行
+ 序列标注模块  
    + 在根目录下运行

      + `python scripts/slu_tagging.py --[options]`

    + 参数设置
      + `--dataroot`:数据文件夹位置
      + `--seed`：随机种子
      + `--device`：使用的GPU编号（-1表示使用CPU）
      + `--testing`：仅测试（当前目录下保存有之前训练过的、同样参数组的模型时才有效）

      + `--model`：选择序列标注模型：baseline/bert
      + `--decode`：选择解码方式：baseline/onei/newdecode
      + `--train_data`：选择用于训练序列标注模型的训练数据来源：manu/asr/MacBERT/sound/Ernie。其中MacBERT/sound/Ernie后缀可添加_his表示使用对话历史，这些数据需要提前使用scripts/csc.py进行生成
      + `--dev_data`： 选择用于测试序列标注模型的测试数据来源：manu/asr/MacBERT/sound/Ernie。其中MacBERT/sound/Ernie后缀可添加_his表示使用对话历史，这些数据需要提前使用scripts/csc.py进行生成
      + `--encoder_cell`：选择encoder—cell：LSTM/GRU/RNN
  
      + `--batch_size`：设置batch大小
      + `--lr`：设置学习率
      + `--max_epoch`：设置训练轮次

+ 文本纠错模块（对ASR的文本结果进行降噪）
  + 实现的方式：
    + 我们自己实现的基于torch的Ernie4CSC及对应预训练模型（不推荐）
    + 基于pycorrector实现的MacBERT4CSC
    + 基于pycorrector实现的Ernie4CSC
    + 我们的方法similar-sound-csc：构建混淆音词典，同时进行纠错和对话历史利用
  + 使用方法
    + `python scripts/csc.py --[options]`
  + 相关参数解释
    + `--csc_model`：使用何种模型进行文本纠错/降噪（可选：`Ernie`，`MacBERT`，`sound`）
    + `--csc_pretrained`：使用预训练模型路径（不使用该选项则使用默认的预训练模型（对Ernie4CSC和MacBERT4CSC）或不使用预训练（对similar-sound-csc）
    + `--csc_train`：是否进行训练（对similar-sound-csc）
    + `--csc_save`：是否保存当前模型词表（对similar-sound-csc)
    + `--use_history`：进行纠错时是否使用当前对话中的历史记录

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU
+ `utils/vocab.py`:构建编码输入输出的词表
+ `utils/word2vec.py`:读取词向量
+ `utils/example.py`:读取数据
+ `utils/batch.py`:将数据以批为单位转化为输入
+ `utils/decoder.py`:从序列标注解码得到slot-value
+ `utils/bert2embd.py`:
+ `utils/example_for_csc.py`:
+ `utils/pos_encoding.py`:
+ `utils/vocab_for_onei.py`:构建支持Onei方法的编码词表
+ `model/slu_bert_bertvocab.py`：bert模型
+ `model/slu_baseline_tagging.py`:baseline模型
+ `model/correcting_model.py`：文本纠错模型
+ `scripts/slu_tagging.py`:主程序脚本
+ `scripts/csc.py`:文本纠错脚本

