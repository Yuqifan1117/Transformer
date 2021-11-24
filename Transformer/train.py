import torch
import torch.nn as nn
from models.model.transformer import Transformer
import time
import numpy as np
from torch.autograd import Variable
from models.model.transformer import Generator
from models.utils import SimpleLossCompute, NoamOpt
from preprocess import PrepareData, LabelSmoothing, subsequent_mask

PAD = 0                             # padding占位符的索引
UNK = 1                             # 未登录词标识符的索引
BATCH_SIZE = 128                    # 批次大小

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_model(src_vocab, tgt_vocab, N_layers=6, Max_len=5000, d_model=512, d_ff=2048, n_head=8, dropout=0.1):
    # c = copy.deepcopy
    # 实例化Transformer模型对象
    model = Transformer(
        enc_voc_size=src_vocab,
        dec_voc_size=tgt_vocab,
        d_model=d_model,
        n_head=n_head,
        max_len=Max_len,
        ffn_hidden=d_ff,
        n_layers=N_layers,
        drop_prob=dropout,
        generator=Generator(d_model, tgt_vocab),
        device=DEVICE)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)


def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.

    for i, batch in enumerate(data):
        # print(batch.src.size(), batch.src_mask.size())
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch %d Batch: %d Loss: %f Tokens per Sec: %fs" % (
            epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens

def train(data, model, criterion, optimizer, save_file, epoch=20):
    """
        训练模型并进行评估保存模型
    """
    best_dev_loss = 1e5

    for epoch in range(epoch):
        model.train()
        run_epoch(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch)
        model.eval()

        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch)
        print('<<<<< Evaluate loss: %f' % dev_loss)

        if dev_loss < best_dev_loss:
            torch.save(model.state_dict(), save_file)
            best_dev_loss = dev_loss
            print('****** Save model done... ******')
    
    print('-----train already-----')

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    传入一个训练好的模型，对指定数据进行预测
    """
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def evaluate(data, model):
    """
    在data上用训练好的模型进行预测，打印模型翻译结果
    """
    # 梯度清零
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for i in range(len(data.dev_en)):
            # 打印待翻译的英文语句
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)

            # 打印对应的中文语句答案
            cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print("".join(cn_sent))

            # 将当前以单词id表示的英文语句数据转为tensor，并存储到DEVICE中
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(DEVICE)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])
            # 初始化一个用于存放模型翻译结果语句单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = data.cn_index_dict[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前语句的翻译结果列表
                if sym != 'EOS':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            # 打印模型翻译输出的中文语句结果
            print("translation: %s" % " ".join(translation))


if __name__ == '__main__':
    TRAIN_FILE = 'data/train.txt'  # 训练集
    DEV_FILE = "data/valid.txt"      # 验证集
    SAVE_FILE = 'save/model.pt'         # 模型保存路径
    BATCH_SIZE = 128                    # 批次大小
    EPOCHS = 50                         # 训练轮数
    LAYERS = 6                          # transformer中encoder、decoder层数
    H_NUM = 8                           # 多头注意力个数
    D_MODEL = 256                       # 输入、输出词向量维数
    D_FF = 1024                         # feed forward全连接层维数
    DROPOUT = 0.1                       # dropout比例
    MAX_LEN = 5000                   # 训练语句最大长度
    MAX_LENGTH = 60                     # 语句最大长度
    data = PrepareData(TRAIN_FILE, DEV_FILE, BATCH_SIZE)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print("src_vocab %d" % src_vocab)
    print("tgt_vocab %d" % tgt_vocab)
    # 初始化模型
    model = make_model(
                        src_vocab,
                        tgt_vocab,
                        LAYERS,
                        MAX_LEN,
                        D_MODEL,
                        D_FF,
                        H_NUM,
                        DROPOUT
                        
                    )

    # 训练
    print(">>>>>>> start train")
    train_start = time.time()
    criterion = LabelSmoothing(tgt_vocab, padding_idx = 0, smoothing= 0.0)
    optimizer = NoamOpt(D_MODEL, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))

    train(data, model, criterion, optimizer, SAVE_FILE, EPOCHS)
    print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")

    # 预测
    # 加载模型
    model.load_state_dict(torch.load(SAVE_FILE))
    # 开始预测
    print(">>>>>>> start evaluate")
    evaluate_start  = time.time()
    evaluate(data, model)
    print(f"<<<<<<< finished evaluate, cost {time.time()-evaluate_start:.4f} seconds")