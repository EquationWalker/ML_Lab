import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
import seaborn as sns
from wordcloud import WordCloud
import time
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchtext import data
from torchtext.vocab import Vectors, GloVe
## 读取训练数据和测试数据
def load_text_data(path):
    ## 获取文件夹的最后一个字段
    text_data = []
    label = []
    for dset in ["pos","neg"]:
        path_dset = os.path.join(path,dset)
        path_list = os.listdir(path_dset)
        ## 读取文件夹下的pos或neg文件
        for fname in path_list:
            if fname.endswith(".txt"):
                filename = os.path.join(path_dset,fname)
                with open(filename) as f:
                    text_data.append(f.read())
            if dset == "pos":
                label.append(1)
            else:
                label.append(0)
    ##  输出读取的文本和对应的标签
    return np.array(text_data),np.array(label)
## 读取训练集和测试集
train_path = "./imdb/train"
train_text,train_label = load_text_data(train_path)
test_path = "./imdb/test"
test_text,test_label = load_text_data(test_path)
print(len(train_text),len(train_label))
print(len(test_text),len(test_label))
train_text[0:2]
string.punctuation[6]
string.punctuation.replace("'","")
## 对文本数据进行预处理
def text_preprocess(text_data):
    text_pre = []
    for text1 in text_data:
        ## 去除指定的字符 <br /><br />
        text1 = re.sub("<br /><br />", " ", text1)
        ## 转化为小写,去除数字,去除标点符号,去除空格
        text1 = text1.lower()
        text1 = re.sub("\d+", "", text1)
        text1 = text1.translate(
            str.maketrans("","", string.punctuation.replace("'","")))
        text1 = text1.strip() 
        text_pre.append(text1)
    return np.array(text_pre)
train_text_pre = text_preprocess(train_text)
test_text_pre = text_preprocess(test_text)
print(train_text[10000])
print("="*10)
print(train_text_pre[10000])
print(train_text[100])
print("="*10)
print(train_text_pre[100])
print(train_text_pre[10000])
## 查看停用词
print(stopwords.words("english"))
print(len(stopwords.words("english")))
## 文本符号化处理,去除停用词，词干化处理
def stop_stem_word(datalist,stop_words,stemer):
    datalist_pre = []
    for text in datalist:
        text_words = word_tokenize(text) 
        ## 去除停用词
        text_words = [word for word in text_words
                      if not word in stop_words]
        ## 删除带“‘”的词语
        text_words = [word for word in text_words
                      if len(re.findall("'",word)) == 0]
        ## 词干化处理
#         text_words = [stemmer.stem(word) for word in text_words]
        datalist_pre.append(text_words)
    return np.array(datalist_pre)
## 文本符号化处理,去除停用词，词干化处理
stop_words = stopwords.words("english")
stop_words = set(stop_words)
stemmer= PorterStemmer()
train_text_pre2 = stop_stem_word(train_text_pre,stop_words,stemmer)
test_text_pre2 = stop_stem_word(test_text_pre,stop_words,stemmer)
print(train_text_pre[10000])
print("="*10)
print(train_text_pre2[10000])
print(train_text_pre2[3])
## 将处理好的文本保存到CSV文件中
texts = [" ".join(words) for words in train_text_pre2]
traindatasave = pd.DataFrame({"text":texts,
                              "label":train_label})
texts = [" ".join(words) for words in test_text_pre2]
testdatasave = pd.DataFrame({"text":texts,
                              "label":test_label})
traindatasave.to_csv("./imdb_train.csv",index=False)
testdatasave.to_csv("./imdb_test.csv",index=False)
print(traindatasave.head())
print(testdatasave.head())
len(texts)
texts[1]
# ### 文本数据可视化
## 将预处理好的文本数据转化为数据表
traindata = pd.DataFrame({"train_text":train_text,
                          "train_word":train_text_pre2,
                          "train_label":train_label})
# testdata = pd.DataFrame({"test_text":test_text,
#                          "test_word":test_text_pre2})
traindata.head()
## 计算每个个影评使用词的数量
train_word_num = [len(text) for text in train_text_pre2]
traindata["train_word_num"] = train_word_num
##可视化影评词语长度的分布
plt.figure(figsize=(8,5))
_ = plt.hist(train_word_num,bins=100)
plt.xlabel("word number")
plt.ylabel("Freq")
plt.show()
traindata.head()
## 使用词云可视化两种情感的词频差异
plt.figure(figsize=(16,10))
for ii in np.unique(train_label):
    ## 准备每种情感的所有词语
    print(ii)
    text = np.array(traindata.train_word[traindata.train_label == ii])
    text = " ".join(np.concatenate(text))
    plt.subplot(1,2,ii+1)
    ## 生成词云
    wordcod = WordCloud(margin=5,width=1800, height=1000,
                        max_words=500, min_font_size=5, 
                        background_color='white',
                        max_font_size=250)
    wordcod.generate_from_text(text)
    plt.imshow(wordcod)
    plt.axis("off")
    if ii == 1:
        plt.title("Positive")
    else:
        plt.title("Negative")
    plt.subplots_adjust(wspace=0.05)
plt.show()
## 可视化正面和负面评论用词分布的差异
sns.boxplot(x=train_label, y=train_word_num,)
# #### 数据准备
# Tokenizer：将句子分成单词列表。如果sequential = False，则不应用标记化
# 
# Field:存储有关预处理方式的信息的类
## 使用torchtext库进行数据准备
# 定义文件中对文本和标签所要做的操作
"""
sequential=True:表明输入的文本时字符，而不是数值字
tokenize="spacy":使用spacy切分词语
use_vocab=True: 创建一个词汇表
batch_first=True: batch悠闲的数据方式
fix_length=200 :每个句子固定长度为200
"""
## 定义文本切分方法，因为前面已经做过处理，所以直接使用空格切分即可
mytokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=mytokenize, 
                  include_lengths=True, use_vocab=True,
                  batch_first=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False, 
                   pad_token=None, unk_token=None)
## 对所要读取的数据集的列进行处理
train_test_fields = [
    ("label", LABEL), # 对标签的操作
    ("text", TEXT) # 对文本的操作
]
## 读取数据
traindata,testdata = data.TabularDataset.splits(
    path="./.", format="csv", 
    train="imdb_train.csv", fields=train_test_fields, 
    test = "imdb_test.csv", skip_header=True
)
len(traindata),len(testdata)
traindata.fields.items()
## TabularDataset是一个包含Example对象的列表。
ex0 = traindata.examples[0]
print(ex0.label)
print(ex0.text)
## 训练集切分为训练集和验证集
train_data, val_data = traindata.split(split_ratio=0.7)
len(train_data),len(val_data)
## 加载预训练的词向量和构建词汇表
## Torchtext使得预训练的词向量的加载变得非常容易。
## 只需提及预训练单词向量的名称（例如glove.6B.50d，fasttext.en.300d等）
vec = Vectors("glove.6B.100d.txt", "./data")
# 将训练集和验证集转化为词项量
## 使用训练集构建单词表，导入预先训练的词嵌入
TEXT.build_vocab(train_data,max_size=20000,vectors = vec)
LABEL.build_vocab(train_data)
## 训练集中的前10个高频词
print(TEXT.vocab.freqs.most_common(n=10))
print("词典的词数:",len(TEXT.vocab.itos))
print("前10个单词:\n",TEXT.vocab.itos[0:10])
## 类别标签的数量和类别
print("类别标签情况:",LABEL.vocab.freqs)
## 查看某个词对应的词项量
TEXT.vocab.vectors[TEXT.vocab.stoi["movie"]]
## 定义一个迭代器，将类似长度的示例一起批处理
BATCH_SIZE = 32
train_iter = data.BucketIterator(train_data,batch_size = BATCH_SIZE)
val_iter = data.BucketIterator(val_data,batch_size = BATCH_SIZE)
test_iter = data.BucketIterator(testdata,batch_size = BATCH_SIZE)
##  获得一个batch的数据，对数据进行内容进行介绍
for step, batch in enumerate(train_iter):  
    if step > 0:
        break
## 针对一个batch 的数据，可以使用batch.label获得数据的类别标签
print("数据的类别标签:\n",batch.label)
## batch.text[0]是文本对应的标签向量
print("数据的尺寸:",batch.text[0].shape)
## batch.text[1] 对应每个batch使用的原始数据中的索引
print("数据样本数:",len(batch.text[1]))
##  获得一个batch的数据，对数据进行内容进行介绍
for step, batch in enumerate(val_iter):  
    if step > 0:
        break
## 针对一个batch 的数据，可以使用batch.label获得数据的类别标签
print("数据的类别标签:\n",batch.label)
## batch.text[0]是文本对应的标签向量
print("数据的尺寸:",batch.text[0].shape)
## batch.text[1] 对应每个batch使用的原始数据中的索引
print("数据样本数:",len(batch.text[1]))
#  ### 构建网络
class CNN_Text(nn.Module):
    def __init__(self,vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        super().__init__()
        """
        vocab_size:词典大小;embedding_dim:词向量维度;
        n_filters:卷积核的个数,filter_sizes:卷积核的尺寸;
        output_dim:输出的维度;pad_idx:填充的索引
        """
        ## 对文本进行词项量
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        ## 卷积操作
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                      kernel_size = (fs, embedding_dim)) for fs in filter_sizes
                                    ])
        ## 全连接层和Dropout层
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text):
        #text = [batch size, sent len]
        embedded = self.embedding(text)    
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]   
        return self.fc(cat)
INPUT_DIM = len(TEXT.vocab) # 词典的数量
EMBEDDING_DIM = 100  # 词向量的维度
N_FILTERS = 100  ## 每个卷积核的个数
FILTER_SIZES = [3,4,5] ## 卷积和的高度
OUTPUT_DIM = 1  
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] # 填充词的索引
model = CNN_Text(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
model
# ### 网络训练和预测
## 将导入的词项量作为embedding.weight的初始值
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
## 将无法识别的词'<unk>', '<pad>'的向量初始化为0
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
## Adam优化,二分类交叉熵作为损失函数
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
## 定义一个对数据集训练一轮的函数
def train_epoch(model, iterator, optimizer, criterion):
    epoch_loss = 0;epoch_acc = 0
    train_corrects = 0;train_num = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        pre = model(batch.text[0]).squeeze(1)
        loss = criterion(pre, batch.label.type(torch.FloatTensor))
        pre_lab = torch.round(torch.sigmoid(pre))
        train_corrects += torch.sum(pre_lab.long() == batch.label)
        train_num += len(batch.label) ## 样本数量
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    ## 所有样本的平均损失和精度
    epoch_loss = epoch_loss / train_num 
    epoch_acc = train_corrects.double().item() / train_num
    return epoch_loss, epoch_acc
## 定义一个对数据集验证一轮的函数
def evaluate(model, iterator, criterion):
    epoch_loss = 0;epoch_acc = 0
    train_corrects = 0;train_num = 0
    model.eval()
    with torch.no_grad(): # 禁止梯度计算
        for batch in iterator:
            pre = model(batch.text[0]).squeeze(1)
            loss = criterion(pre, batch.label.type(torch.FloatTensor))
            pre_lab = torch.round(torch.sigmoid(pre))
            train_corrects += torch.sum(pre_lab.long() == batch.label)
            train_num += len(batch.label) ## 样本数量
            epoch_loss += loss.item()
        ## 所有样本的平均损失和精度
        epoch_loss = epoch_loss / train_num 
        epoch_acc = train_corrects.double().item() / train_num
    return epoch_loss, epoch_acc   
## 使用训练集训练模型，验证集测试模型
EPOCHS = 10
best_val_loss = float("inf")
best_acc  = float(0)
for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_iter, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_iter, criterion)
    end_time = time.time()
    print("Epoch:" ,epoch+1 ,"|" ,"Epoch Time: ",end_time - start_time, "s")
    print("Train Loss:", train_loss, "|" ,"Train Acc: ",train_acc)
    print("Val. Loss: ",val_loss, "|",  "Val. Acc: ",val_acc)
    ## 保存效果较好的模型
    if (val_loss < best_val_loss) & (val_acc > best_acc):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_loss = val_loss
        best_acc = val_acc
# 将最好模型的参数重新赋值给model
model.load_state_dict(best_model_wts)
## 模型的保存和导入
torch.save(model,"./textcnnmodel.pkl")
## 导入保存的模型
model = torch.load("./textcnnmodel.pkl")
model
## 使用evaluate函数对测试集进行预测
test_loss, test_acc = evaluate(model, test_iter, criterion)
print("在测试集上的预测精度为:", test_acc)
# ## 训练好模型的重复使用
import dill
## 保存Field实例
with open("./TEXT.Field","wb")as f:
     dill.dump(TEXT,f)      
with open("./LABEL.Field","wb")as f:
     dill.dump(LABEL,f)
## 导入保存后的Field实例
with open("./TEXT.Field","rb")as f:
     TEXT=dill.load(f)
with open("./LABEL.Field","rb")as f:
     LABEL=dill.load(f)
print(TEXT.vocab.freqs.most_common(n=10))
print("词典的词数:",len(TEXT.vocab.itos))
print("前10个单词:\n",TEXT.vocab.itos[0:10])
## 类别标签的数量和类别
print("类别标签情况:",LABEL.vocab.freqs)
## 对所要读取的数据集的列进行处理
train_test_fields2 = [
    ("label", LABEL), # 对标签的操作
    ("text", TEXT) # 对文本的操作
]
## 读取数据
testdata2 = data.TabularDataset.splits(
    path="./.", format="csv", 
    fields=train_test_fields2, 
    test = "imdb_test.csv", skip_header=True
)
test_iter = data.BucketIterator(testdata,batch_size = 32)
## 使用evaluate函数对测试集进行预测
test_loss, test_acc = evaluate(model, test_iter, criterion)
print("在测试集上的预测精度为:", test_acc)