from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import pandas as pd
import os
import random
import math
from bert_model import BertEncoder


def get_data():
    all_data = pd.read_csv(os.path.join("..","data","task2.csv"))

    # all_data = sklearn.utils.shuffle(all_data)

    text1 = all_data["text1"].tolist()
    text2 = all_data["text2"].tolist()
    label = all_data["label"].tolist()

    return text1,text2,label


class BDataset(Dataset):
    def __init__(self,all_text1,all_text2,all_lable,max_len,word_2_index):
        assert len(all_text1) == len(all_text2) == len(all_lable),"数据长度都不一样，复现个冒险啊！"
        self.all_text1 = all_text1
        self.all_text2 = all_text2
        self.all_lable = all_lable
        self.max_len = max_len
        self.word_2_index = word_2_index


    def __getitem__(self, index):
        text1 = self.all_text1[index]
        text2 = self.all_text2[index]

        lable = self.all_lable[index]

        text1_idx = [word_2_index.get(i,self.word_2_index["[UNK]"]) for i in text1][:62]
        text2_idx = [word_2_index.get(i,self.word_2_index["[UNK]"]) for i in text2][:62]



        mask_val = [0] * self.max_len

        text_idx = [self.word_2_index["[CLS]"]] + text1_idx + [self.word_2_index["[SEP]"]] + text2_idx + [self.word_2_index["[SEP]"]]
        seg_idx = [0] + [0] * len(text1_idx) + [0] + [1] * len(text2_idx) + [1] + [2] * (self.max_len - len(text_idx))

        for i,v in enumerate(text_idx):
            if v in [self.word_2_index["[CLS]"],self.word_2_index["[SEP]"],self.word_2_index["[UNK]"]] :
                continue

            if random.random() < 0.15:
                r = random.random()
                if  r < 0.8:
                    text_idx[i] = self.word_2_index["[MASK]"]

                    mask_val[i] = v

                elif r > 0.9:
                    other_idx = random.randint(6,len(self.word_2_index)-1)
                    text_idx[i] = other_idx
                    mask_val[i] = v


        text_idx = text_idx + [self.word_2_index["[PAD]"] ]* (self.max_len - len(text_idx))


        return torch.tensor(text_idx) , torch.tensor(lable) ,torch.tensor(mask_val),torch.tensor(seg_idx)



    def __len__(self):
        return len(self.all_lable)


class BertEmbeddding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.word_embeddings.weight.requires_grad = True

        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.position_embeddings.weight.requires_grad = True

        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["hidden_size"])
        self.token_type_embeddings.weight.requires_grad = True

        self.LayerNorm = nn.LayerNorm(config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])


    def forward(self,batch_idx,batch_seg_idx):
        word_emb = self.word_embeddings(batch_idx)

        pos_idx = torch.arange(0,self.position_embeddings.weight.data.shape[0],device=batch_idx.device)
        pos_idx = pos_idx.repeat(batch_idx.shape[0],1)
        pos_emb = self.position_embeddings(pos_idx)

        token_emb = self.token_type_embeddings(batch_seg_idx)

        emb = word_emb + pos_emb + token_emb

        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)

        return emb


class BertPooler(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.activation = nn.Tanh()


    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]

        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embedding = BertEmbeddding(config)
        # self.bert_layer = nn.Linear(config["hidden_size"],config["hidden_size"]) #


        self.bert_layer = nn.Sequential(*[BertEncoder(config["hidden_size"],config["feed_num"],config["head_num"]) for i in range(config["layer_num"])])

        self.pool = BertPooler(config)

    def forward(self,batch_idx,batch_seg_idx):
        x = self.embedding(batch_idx,batch_seg_idx)

        x = self.bert_layer.forward(x)


        bertout2 = self.pool(x)

        return x,bertout2


class Model(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.bert = BertModel(config) #  3*128*768 @  768 * 768 = 3*128*768 @ 768 * 10 = 3 * 128 * 10,             3*768  @ 768 * 2  = 3 * 2

        self.cls_mask = nn.Linear(config["hidden_size"],config["vocab_size"])
        self.cls_next = nn.Linear(config["hidden_size"],2)

        self.loss_fun_mask = nn.CrossEntropyLoss(ignore_index=0)
        self.loss_fun_next = nn.CrossEntropyLoss()


    def forward(self,batch_idx,batch_seg_idx,batch_mask_val=None,batch_label=None):
        bertout1,bertout2 = self.bert(batch_idx,batch_seg_idx)

        pre_mask = self.cls_mask(bertout1)
        pre_next = self.cls_next(bertout2)


        if batch_mask_val is not None and batch_label is not None:
            loss_mask = self.loss_fun_mask(pre_mask.reshape(-1,pre_mask.shape[-1]),batch_mask_val.reshape(-1))
            loss_next = self.loss_fun_next(pre_next,batch_label)

            loss = loss_mask + loss_next

            return loss
        else:
            return torch.argmax(pre_mask,dim=-1),  torch.argmax(pre_next,dim=-1)



if __name__ == "__main__":
    # all_text = read_data()
    # build_task2_dataset(all_text)
    # word_2_index = build_word_2_index(all_text)

    all_text1,all_text2,all_label = get_data()

    with open("index_2_word.txt",encoding="utf-8") as f:
        index_2_word = f.read().split("\n")
        word_2_index = {w:idx for idx,w in enumerate(index_2_word)}

    epoch = 10
    batch_size = 40
    max_len = 128       #
    lr = 0.001


    config = {
        "epoch" :epoch,
        "batch_size":batch_size,
        "max_len" : max_len,
        "vocab_size" : len(word_2_index),
        "hidden_size":768,
        "max_position_embeddings":128,
        "head_num":4,
        "feed_num":1024,
        "type_vocab_size":3,
        "hidden_dropout_prob":0.2,
        "layer_num":3,
        "device":"cuda:0" if torch.cuda.is_available() else "cpu"
    }

    dev_size = 400

    train_dataset = BDataset(all_text1[:-dev_size], all_text2[:-dev_size], all_label[:-dev_size], max_len, word_2_index)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    dev_dataset = BDataset(all_text1[-dev_size:], all_text2[-dev_size:], all_label[-dev_size:], max_len, word_2_index)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    model = Model(config).to(config["device"])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for e in range(epoch):
        print(f"epoch : {e}")


        for i, (batch_idx, batch_label, batch_mask_val, batch_seg_idx) in enumerate(train_dataloader):
            model.train()
            batch_idx = batch_idx.to(config["device"])
            batch_label = batch_label.to(config["device"])
            batch_mask_val = batch_mask_val.to(config["device"])
            batch_seg_idx = batch_seg_idx.to(config["device"])

            loss = model.forward(batch_idx, batch_seg_idx, batch_mask_val, batch_label)
            loss.backward()

            opt.step()
            opt.zero_grad()

            mask_right_num = 0
            mask_all_num = 0

            next_right_num = 0

            if i % 200 == 0:
                model.eval()
                print(f"loss:{loss:.2f}")

                for i, (batch_idx, batch_label, batch_mask_val, batch_seg_idx) in enumerate(dev_dataloader):
                    batch_idx = batch_idx.to(config["device"])
                    batch_label = batch_label.to(config["device"])
                    batch_mask_val = batch_mask_val.to(config["device"])
                    batch_seg_idx = batch_seg_idx.to(config["device"])

                    pre_mask, pre_next = model.forward(batch_idx, batch_seg_idx)

                    mask_right_num += int(
                        torch.sum(pre_mask[[batch_mask_val != 0]] == batch_mask_val[batch_mask_val != 0]))
                    mask_all_num += len(pre_mask[[batch_mask_val != 0]])

                    next_right_num += int(torch.sum(pre_next == batch_label))

                acc_mask = mask_right_num / mask_all_num * 100
                acc_next = next_right_num / len(dev_dataset) * 100

                print("*" * 100)
                print(f"acc_mask:{acc_mask:.3f}%, acc_next:{acc_next:.3f}%")




