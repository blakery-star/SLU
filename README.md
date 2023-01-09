### 创建环境

    conda create -n slu python=3.6
    conda activate slu
    pip install torch==1.7.1
    
    以上为推荐设置，在本branch中同样支持高版本torch（经验证，至少支持至1.13.0版本）。
### 运行
    
在根目录下运行

    python scripts/slu_baseline.py

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成
        
        python scripts/slu_baseline.py --<arg> <value>
    其中，`<arg>`为要修改的参数名，`<value>`为修改后的值
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU
+ `utils/vocab.py`:构建编码输入输出的词表
+ `utils/word2vec.py`:读取词向量
+ `utils/example.py`:读取数据
+ `utils/batch.py`:将数据以批为单位转化为输入
+ `model/slu_baseline_tagging.py`:baseline模型
+ `scripts/slu_baseline.py`:主程序脚本

### 有关预训练语言模型

本次代码中没有加入有关预训练语言模型的代码，如需使用预训练语言模型我们推荐使用下面几个预训练模型，若使用预训练语言模型，不要使用large级别的模型
+ Bert: https://huggingface.co/bert-base-chinese
+ Bert-WWM: https://huggingface.co/hfl/chinese-bert-wwm-ext
+ Roberta-WWM: https://huggingface.co/hfl/chinese-roberta-wwm-ext
+ MacBert: https://huggingface.co/hfl/chinese-macbert-base

### 推荐使用的工具库
+ transformers
  + 使用预训练语言模型的工具库: https://huggingface.co/
+ nltk
  + 强力的NLP工具库: https://www.nltk.org/
+ stanza
  + 强力的NLP工具库: https://stanfordnlp.github.io/stanza/
+ jieba
  + 中文分词工具: https://github.com/fxsjy/jieba

### 改动
+ 针对特定运行环境下编码错误的修复
  + 对在\utils文件夹下的文件中的所有json.load(f)函数，在其之前的open函数的输入中，添加了变量设置encoding = 'utf-8'。
+ 增加了多模型选择模块
  + 在scripts文件夹中增加了slu_tagging.py文件，该文件同样为主程序脚本，但支持选择不同的模型进行测试或训练。
  + 修改了utils/args.py文件，添加了"--model"参数。