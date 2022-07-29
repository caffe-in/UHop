import pickle
from collections import defaultdict
import itertools
import operator
from torch.utils.data import DataLoader, Dataset
import torch
import json
from functools import reduce
from itertools import accumulate
import random
import numpy as np
import pytorch_lightning as pl
from transformers import BertTokenizer
from itertools import accumulate
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

def gen_ques_classifier(data_path,dump_path):
    datafile = open(data_path,'r',encoding='utf8')
    dumpfile = open(dump_path,'w',encoding='utf8')
    lines = datafile.readlines()
    for line in lines:
        line = line.strip().split("\t")
        ques = line[0]
        if len(line)==4:
            dumpfile.write(ques+'\t'+str(0)+'\n')
        elif len(line)==5:
            dumpfile.write(ques+'\t'+str(1)+'\n')
def process_score(score_path,dump_path):
    score_file = open(score_path,'r')
    dump_file = open(dump_path,'w',encoding="utf8")
    lines = score_file.readline().strip()
    lines = eval(lines)
    for line in lines:
        dump_file.write(str(line)+'\n')

def preprae_train_data(data_path,dump_path):
    '''
    translate the train file
    ques
    head rel tail
    head rel tail
    to
    ques ans entity1#rel1#entity2#...
    '''
    filein = open(data_path,'r',encoding='utf8')
    fileout = open(dump_path,'w',encoding='utf8')
    lines = filein.readlines()
    count = 0
    for line in lines:
        if line=='\n':
            ans = tail
            rel_str+=tail+"#<end>#"+ans
            s = ques+'\t'+ans+'\t'+rel_str+'\n'
            fileout.write(s)
            count=0
        else:
            if count==0:
                ques = line.strip().split('\t')[0]
                rel_str = ""
            else:
                
                line = line.strip().split()
                head = line[0]
                rel = line[1]
                tail = line[2]
                rel_str+=head+"#"+rel+"#"
                # if count==1:
                #     ques.replace(head.replace("_"," "),"NER")
            count+=1


def standard_train(train_flie):
    '''
    replace the " " of entity of trian file with "_"
    '''
    file = open(train_flie,'r+',encoding='utf8')
    lines = file.readlines()
    for idx,line in enumerate(lines):
        line = line.strip().split("\t")
        rel_list = line[1:]
        rel_list = [rel.replace(" ","_") for rel in rel_list]
        rel_list = [line[0]]+rel_list
        line = "\t".join(rel_list)
        line+='\n'
        lines[idx]=line
    for i in range(0,5):
        print(lines[i])
    file.seek(0)
    file.truncate()
    file.writelines(lines)    
    file.close()

def split_hop23(data_path,dump2path,dump3_path):
    '''
    split the trian_data by the hop num to
    dump2path,dump3path
    '''
    filein = open(data_path,'r',encoding='utf8')
    fout2 = open(dump2path,'w',encoding='utf8')
    fout3 = open(dump3_path,'w',encoding='utf8')
    lines = filein.readlines()
    for line in lines:
        raw_line = line
        line = line.strip().split('\t')
        rel_list = line[2]
        num = rel_list.count('#')
        if(num==6):               # ques entity rel1 rel2
            fout2.write(raw_line)
        else:
            fout3.write(raw_line) 

def merge_file(data2path,data3path,dataAllpath,rela:bool):
    if not rela:
        with open(data2path,'r') as f2,open(data3path,'r') as f3,open(dataAllpath,"w") as fAll:
            lines2 = f2.readlines()
            lines3 = f3.readlines()
            linesAll = lines2+lines3
            fAll.writelines(linesAll)
    else:
        with open(data2path,'r') as f2,open(data3path,'r') as f3,open(dataAllpath,"w") as fAll:
            dic2 = json.load(f2)
            dic3 = json.load(f3)
            dicAll = dic2.copy()
            length = len(dic2.keys())
            for key in dic3.keys():
                if key not in dicAll.keys():
                    dicAll[key]=length
                    length+=1
            json.dump(dicAll,fAll)

def merge_dictory(hop2path,hop3path,Allpath):
    rela2id2_path = hop2path+"//rela2id.json"
    rela2id3_path = hop3path+"//rela2id.json"
    rela2idAll_path = Allpath+"//rela2id.json"
    merge_file(rela2id2_path,rela2id3_path,rela2idAll_path,rela=True)

    mode_list = ["test","train","valid"]
    for mode in mode_list:
        data2_path = hop2path+f"//{mode}_data.txt"
        data3_path = hop3path+f"//{mode}_data.txt"
        dataAll_path = Allpath+f"//{mode}_data.txt"
        merge_file(data2_path,data3_path,dataAll_path,rela=False)

def valid_ques_gen(valid_data_path,train_datap_path,dump_path):
    fvalid  = open(valid_data_path,'r')
    ftrainp = open(train_datap_path,'r')
    fdump = open(dump_path,'w',encoding='utf8')
    lines_valid = fvalid.readlines()
    lines_trainp = ftrainp.readlines()
    for line in lines_valid:
        line = eval(line)
        idx = line[0]
        sample = lines_trainp[idx]
        sample = sample.strip().split('\t')
        ques = sample[0]
        rel_list = sample[2].split('#')
        topic_entity = rel_list[0]
        rel = []
        for i,obj in enumerate(rel_list):
            if obj=="<end>":break
            if i%2==1:
                rel.append(obj)
        rel_str = "\t".join(rel)
        s = ques+"\t"+topic_entity+"\t"+rel_str+'\n'
        fdump.write(s)
class PerQuestionDataset(Dataset):
    def __init__(self,lines,batch_size,word2id, rela2id,mode,q_representation=False,pretrain_model="."):
        super(PerQuestionDataset, self).__init__()
        self.batch_size = batch_size
        self.q_representation = q_representation
        if self.q_representation:   
            self.bert_tokenizer = BertTokenizer.from_pretrained(pretrain_model)
        self.data_objs = self._get_data(lines, word2id, rela2id,mode)
    def _get_data(self,lines, word2id, rela2id,mode):
        data_objs = []
        for i, line in enumerate(lines):
            data = json.loads(line)
            data = self._numericalize(data, word2id, rela2id, self.q_representation,mode)
            data_objs.append(data)
        return data_objs
    def _numericalize(self, data, word2id, rela2id, q_representation,mode):
        index, ques, step_list = data[0], data[1], data[2]
        if q_representation == "bert":
            ques = self._bert_numericalize_str(ques)
        else:
            ques = self._numericalize_str(ques, word2id, [' '])
        if mode=="predict":
            new_step_list = []
        else:
            new_step_list=[]
            for step in step_list:
                new_step = []
                for t in step:
                    num_rela = self._numericalize_str(t[0], rela2id, ['.'])
                    num_rela_text = self._numericalize_str(t[0], word2id, ['.', '_'])
                    num_prev = [self._numericalize_str(prev, rela2id, ['.']) for prev in t[1]]
                    num_prev_text = [self._numericalize_str(prev, word2id, ['.', '_']) for prev in t[1]]
                    new_step.append((num_rela, num_rela_text, num_prev, num_prev_text, t[2]))
                new_step_list.append(new_step)
        return index, ques, new_step_list
    def _numericalize_str(self, string, map2id, dilemeter):
        if len(dilemeter) == 2:
            string = string.replace(dilemeter[1], dilemeter[0])
        dilemeter = dilemeter[0]
        tokens = string.strip().split(dilemeter)
        tokens = [map2id[x] if x in map2id else map2id['<unk>'] for x in tokens]
        return tokens
    def _bert_numericalize_str(self, seq):

        tokens = self.bert_tokenizer.tokenize(seq)
        tokens = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        return tokens
    def __len__(self):
        return len(self.data_objs)
    def __getitem__(self, index):
        return self.data_objs[index]

class PerQuestionDataModule(pl.LightningDataModule):
    def __init__(self,data_path,word2id, rela2id,mode,q_representation=False,pretrain_model="."):
        super().__init__()
        self.data_path = data_path
        self.word2id = word2id
        self.rela2id = rela2id
        self.mode  = mode
        self.q_representation = q_representation
        self.pretrain_model = pretrain_model
    def setup(self,stage):
        if stage in ("fit",None):
            with open(self.data_path,'r') as f:
                lines = f.readlines()
            rest_data,test_data= train_test_split(lines,test_size=0.1,shuffle=True)
            train_data,valid_data = train_test_split(rest_data,test_size=0.1,shuffle=False)

            self.train_dataset = PerQuestionDataset(train_data,self.word2id,self.rela2id,mode="train",
                                                    q_representation=self.q_representation,pretrain_model=self.pretrain_model)
            self.valid_dataset = PerQuestionDataset(valid_data,self.word2id,self.rela2id,mode="train",
                                                    q_representation=self.q_representation,pretrain_model=self.pretrain_model)
            self.test_dataset = PerQuestionDataset(test_data,self.word2id,self.rela2id,mode="train",
                                                    q_representation=self.q_representation,pretrain_model=self.pretrain_model)
        elif stage in("predict",None):
            with open(self.data_path,'r') as f:
                lines = f.readlines()
            self.predict_dataset = PerQuestionDataset(lines,self.word2id,self.rela2id,mode="predict",
                                                    q_representation=self.q_representation,pretrain_model=self.pretrain_model)
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=4)
    def val_dataloader(self):
        return DataLoader(self.valid_dataset,batch_size=self.batch_size,num_workers=4)
    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size,num_workers=4)
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,batch_size=self.batch_size,num_workers=4)
            
class Parser():
    def __init__(self, file_name, kb_name, hop_num):
        # read and split file 
        with open(file_name, 'r') as f:
            raw_data = [line.split('\t') for line in f.read().splitlines()]
        with open(kb_name, 'r') as f:
            kb = [line.split('\t') for line in f.read().splitlines()]
        self.kb = kb
        self.hop = hop_num
        self.gold = []
        self.data, self.rela, self.concat_rela, self.baseline, self.ent = [], [], [], [], []
        for idx, data in enumerate(raw_data[:]):
            if idx%100==0:
                print(idx)
            # if idx>20:
            #     break
            # extract candidates of UHop
            ques, steplist, gold_rela, rela_list = self.make_UHop(data)
            self.rela.append(rela_list)
            self.data.append([idx, ques, steplist])
            self.gold.append(gold_rela)
 #           print(steplist)

            # extract candidates of baseline
            neg, concat_rela = self.make_baseline(data, gold_rela)
            self.baseline.append([gold_rela, neg, ques])
            self.concat_rela.append(concat_rela)

    def make_UHop(self, data):
        path = data[2].split('#<end>')[0].split('#')
        steplist = []
        prev_step = []
        prev_ent = [[path[0]]]
        for i in range(0,self.hop*2,2):
            rela = list(set([k[1] for k in self.kb if k[0] in prev_ent[-1]]))
            #rela = list(set([k[1] for k in kb if k[0] == path[i]]))
            cands = [[self._format(r), prev_step[:], int(len(path)>i+1 and r==path[i+1])] for r in rela]
            steplist.append(cands)
            if len(path)>i+1:
                prev_step.append(self._format(path[i+1]))
                prev_ent.append(list(set([k[2] for k in self.kb if (k[1]==path[i+1] and (k[0] in prev_ent[-1]))])))
        ques = data[0].replace(path[0],'TOPIC_ENTITY')
        gold_rela = '..'.join([max(step, key=lambda x:x[2])[0] for step in steplist[:-1]])
        rela_list = sum([self._format(path[i]).split('.') for i in range(1,len(path),2)],[])
        return ques, steplist, gold_rela, rela_list

    def make_baseline(self, data, gold_rela):
        path = data[2].split('#<end>')[0].split('#')
        root = path[0]
        candidates = self.kb_dfs(root, 3)
        if gold_rela not in candidates:
            print(gold_rela)
        candidates = list(set(candidates))
        neg = [rela for rela in candidates if rela!=gold_rela]
        return neg, candidates

    def kb_dfs(self, entity, hop):
        candidates = []
        for k in self.kb:
            if k[0]==entity:
                if hop > 1:
                    candidates += [(self._format(k[1]) + '..' + candidate) for candidate in self.kb_dfs(k[2], hop-1)]
                candidates += [self._format(k[1])]
        return candidates

    def baseline_format(self, data, concat_rela2id):
        baseline = []
        for d in data:
            pos = str(concat_rela2id[d[0]])
            neg = ' '.join([str(concat_rela2id[rela]) for rela in d[1]]) if len(d[1]) > 0 else 'noNegativeAnswer'
            ques = d[2]
            baseline.append('\t'.join([pos, neg, ques]))
        return baseline

    def _split(self, data):
        random.shuffle(data)
        # split data into 8:1:1 and dump
        a, b = int(len(data)*0.8), int(len(data)*0.9)
        print(a, b-a, len(data)-b)
        return data[:a], data[a:b], data[b:]

    def std_dump(self, dir_name, baseline=False):
        train, valid, test = self._split(self.data) if not baseline else self._split(self.baseline)
        rela = self.rela
        
        if not baseline:
            self.dump(dir_name, train, valid, test, rela)
        else:
            concat_rela = self.concat_rela
            self.dump_baseline(dir_name, train, valid, test, rela, concat_rela)

    def merge_dump(self, parser2, dir_name, baseline=False):
        train1, valid1, test1 = self._split(self.data) if not baseline else self._split(self.baseline)
        train2, valid2, test2 = self._split(parser2.data) if not baseline else self._split(parser2.baseline)

        train = train1 + train2
        valid = valid1 + valid2
        test  = test1 + test2
        rela = self.rela + parser2.rela

        if not baseline:
            self.dump(dir_name, train, valid, test, rela)
        else:
            concat_rela = self.concat_rela + parser2.concat_rela
            self.dump_baseline(dir_name, train, valid, test, rela, concat_rela)

    def mixed_dump(self, parser2, train_size, dir_name, baseline_dir_name):
        data1, data2 = self.data, parser2.data
        tv1, tv2 = data1[:train_size*100], data2[:train_size*100]
        test1, test2 = data1[train_size*100:], data2[train_size*100:]

        data1_b, data2_b = self.baseline, parser2.baseline
        tv1_b, tv2_b = data1_b[:train_size*100], data2_b[:train_size*100]
        test1_b, test2_b = data1_b[train_size*100:], data2_b[train_size*100:]

        rela = self.rela + parser2.rela
        concat_rel = self.concat_rela + parser2.concat_rela

        idx1, idx2 = [i for i in range(train_size*100)], [i for i in range(train_size*100)]
        random.shuffle(idx1)
        random.shuffle(idx2)

        # mixed 2 and 3
        for w in range(0,11):
            a1, b1 = train_size*w*9, train_size*w*10
            a2, b2 = train_size*(10-w)*9, train_size*(10-w)*10
            train = tv1[idx[:a1]] + tv2[idx[:a2]]
            valid = tv1[idx[a1:b1]] + tv2[idx[a2:b2]]

            if not baseline:
                self.dump(dir_name+'/'+str(w)+'_'+str(10-w), train, valid, test1, rela, test2)
#                self.dump(dir_name+'1/'+str(w)+'_'+str(10-w), train, valid, test1, rela)
#                self.dump(dir_name+'2/'+str(w)+'_'+str(10-w), train, valid, test2, rela)
            else:
                self.dump_baseline(dir_name+'/'+str(w)+'_'+str(10-w), train, valid, test1, rela, concat_rel, test2)
#                self.dump_baseline(dir_name+'1/'+str(w)+'_'+str(10-w), train, valid, test1, rela, concat_rela)
#                self.dump_baseline(dir_name+'2/'+str(w)+'_'+str(10-w), train, valid, test2, rela, concat_rela)

    def dump(self, dir_name, train, valid, test, rela, test2=None):
        self._check_dir(dir_name)
        self._write(train, dir_name+'/train_data.txt') 
        self._write(valid, dir_name+'/valid_data.txt') 
        self._write(test, dir_name+'/test_data.txt')
        self._rela2id(rela, dir_name+'/rela2id.json')
        if test2:
            self._write(test2, dir_name+'/test2_data.txt')

    def dump_baseline(self, dir_name, train, valid, test, rela, concat_rela, test2=None):
        self._check_dir(dir_name)
        concat_rela2id = self._rela2id(concat_rela, dir_name+'/concat_rela2id.json')
        train = self.baseline_format(train, concat_rela2id)
        valid = self.baseline_format(valid, concat_rela2id)
        test = self.baseline_format(test, concat_rela2id)
        self._write2(train, dir_name+'/train_data.txt') 
        self._write2(valid, dir_name+'/valid_data.txt') 
        self._write2(test, dir_name+'/test_data.txt')
        self._rela2id(rela, dir_name+'/rela2id.json')
        if test2:
            test2 = self.baseline_format(test2, concat_rela2id)
            self._write2(test2, dir_name+'/test2_data.txt')
   
    def _format(self, rela):
        if rela[:2]=='__':
            rela = rela[2:]
        return rela.replace('__','.')

    def _write(self, data, output_name):
        with open(output_name, 'w') as f:
            for line in data:
                json.dump(line, f, ensure_ascii=False)
                f.write('\n')

    def _write2(self, data, output_name):
        with open(output_name, 'w') as f:
            for line in data:
                f.write(line+'\n')

    def _rela2id(self, relations, rela_name):
        rela = ["PADDING", "<unk>"] + list(set([i for j in relations for i in j]))
        rela = dict([(r,idx) for idx, r in enumerate(rela)])
        with open(rela_name, 'w') as f:
            json.dump(rela, f, ensure_ascii=False)
        return rela
                
    def _check_dir(self, dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def all_rela(self):
        return list(set([i for j in self.rela for i in j]))

    def count_rela(self):
        return list(set([k[1] for k in self.kb]))

    def count_candidate(self):
        counts = [[len(d[2][i]) for d in self.data] for i in range(self.hop)]
        return [[step.count(i) for i in range(max(step))] for step in counts]

    def count_ent(self):
        counts = [[len(e[i]) for e in self.ent] for i in range(self.hop)]
        return [[step.count(i) for i in range(max(step)+1)] for step in counts]
if __name__ == '__main__':
    #1. translate the train_data.txt to train_datap.txt
    base_path = "..//data"
    data_path = base_path+"//train_data.txt"
    dump_path = base_path+"//train_datap.txt"
    preprae_train_data(data_path,dump_path)

    #2. split the train_datap.txt into hop2 and hop3
    base_path = "..//data"
    data_path = base_path+"//train_datap.txt"
    dump2_path = base_path+"//ccks2hop//train_datap.txt"
    dump3_path = base_path+"//ccks3hop//train_datap.txt"
    split_hop23(data_path,dump2_path,dump3_path)

    #3. generate the medium data for hop2 and hop3, then merge them
    base_path = "..//data"
    kb_path = "..//data//triples.txt"
    data2_path = base_path+"//ccks2hop//train_datap.txt"
    data3_path = base_path+"//ccks3hop//train_datap.txt"
    dump2_dic = base_path+"//ccks2hop"
    dump3_dic = base_path+"//ccks3hop"
    dumpAll_dic = base_path+"//ccksAll"
    ccks2hop = Parser(data2_path,kb_path,3)
    ccks3hop = Parser(data3_path,kb_path,4)

    ccks2hop.std_dump(dump2_dic)
    ccks3hop.std_dump(dump3_dic)

    merge_dictory(dump2_dic,dump3_dic,dumpAll_dic)

    #4. generate the valid_ques for hop2 and hop3, then merge them
    base_path = "..//data"

    valid2_path = base_path+"//ccks2hop//valid_data.txt"
    train_datap2_path = base_path+"//ccks2hop//train_datap.txt"
    dump2_path = base_path+"//ccks2hop//valid_ques.txt"
    valid_ques_gen(valid2_path,train_datap2_path,dump2_path)

    valid3_path = base_path+"//ccks3hop//valid_data.txt"
    train_datap3_path = base_path+"//ccks3hop//train_datap.txt"
    dump3_path = base_path+"//ccks3hop//valid_ques.txt"
    valid_ques_gen(valid3_path,train_datap3_path,dump3_path)

    dump_all_path = "..//data//valid_ques.txt"
    merge_file(dump2_path,dump3_path,dump_all_path,rela=False)

    data_path = "..//data//train_ques.txt"
    dump_path = "..//data//hop_ques.txt"

    gen_ques_classifier(data_path,dump_path)

    # #5. train the model and test the model ...

    # 6. process the socres.json in saved_model and generate the hop_ques for classifier
    # base_path = "..//saved_model//HR_BiLSTM_plus_4"
    # score_path = base_path+"//scores_82.99.json"
    # dump_path = base_path+"//scores.txt"
    # process_score(score_path,dump_path)




    # #7. predict the valid_ques and run the valid.py...
    

    # #8. predict the  predict_ques
    pass
