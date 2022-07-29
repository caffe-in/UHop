import pytorch_lightning as pl
import transformers
import json
import torch
class UhopModel(pl.LightningModule):
    def __init__(self,args,word2id,rela2id,model):
        super().__init__()
        self.args = args
        self.word2id = word2id
        self.rela2id = rela2id
        self.id2rela = {v:k for k,v in rela2id.items()}
        self.model = model
        self.total_loss, self.total_acc = 0.0, 0.0
        self.loss_count, self.acc_count = 0, 0
        self.total_rc_acc, self.total_td_acc = 0.0, 0.0
        self.rc_count, self.td_count = 0, 0
        self.labels = []
        self.scores = []
    def _eval_metric(self, scores):
        pos_scores = scores[0].repeat(len(scores)-1)
        neg_scores = scores[1:]
        ones = torch.ones(len(neg_scores)).cuda()
        loss = self.loss_function(pos_scores, neg_scores, ones)
        acc = 1 if all([x > y for x, y in zip(pos_scores, neg_scores)]) else 0
        return loss, acc

    def _loss_weight(self, current_len, total_len, acc, task):
        hop_weight = self.args.hop_weight**(current_len)
        task_weight = self.args.task_weight if task=='TD' else 1
        acc_weight = self.args.acc_weight if acc==1 else 1
        return acc_weight / (hop_weight * task_weight)
    def _padding_cuda(self, seqs, maxlen, pad_type, padding, start_position=None):
        pad_seq, mask, position = [], [], []
        for seq in seqs:
            if pad_type == 'append':
                pad_seq.append(seq + [padding]*(maxlen-len(seq)))
                mask.append([1]*len(seq) + [0]*(maxlen-len(seq)))
                if start_position != None:
                    position.append([i+start_position for i in range(len(seq))] + [0]*(maxlen-len(seq)))
            elif pad_type == 'prepend':
                pad_seq.append([padding]*(maxlen-len(seq)) + seq)
                mask.append([0]*(maxlen-len(seq)) + [1]*len(seq))
                if start_position != None:
                    position.append([0]*(maxlen-len(seq)) + [i+start_position for i in range(len(seq))])
        if start_position == None:
            return torch.LongTensor(pad_seq).cuda(), torch.LongTensor(mask).cuda()
        return torch.LongTensor(pad_seq).cuda(), torch.LongTensor(mask).cuda(), torch.LongTensor(position).cuda()
    def _single_UHop_step(self, model, ques, pos_tuples, neg_tuples):
        pos_rela, pos_rela_text, pos_prev, pos_prev_text, _ = zip(*pos_tuples)
        neg_rela, neg_rela_text, neg_prev, neg_prev_text, _ = zip(*neg_tuples)
        # input of question
        if self.args.q_representation == 'bert':
            ques, ques_mask = self._padding_cuda([ques]*(len(pos_rela)+len(neg_rela)), max(5, len(ques)), 'append', 0)
            ques = torch.cat([ques, ques_mask], dim=-1)
        else:
            ques, _ = self._padding_cuda([ques]*(len(pos_rela)+len(neg_rela)), max(5, len(ques)), 'append', self.word2id['PADDING'])
        # input of relation and previous
        if self.framework == 'baseline' or self.args.dynamic == 'none':
            # concat all previous and relation
            pos_relas = [sum(prev+[rela], []) for prev, rela in zip(pos_prev, pos_rela)]
            neg_relas = [sum(prev+[rela], []) for prev, rela in zip(neg_prev, neg_rela)]
            maxlen = max([len(rela) for rela in pos_relas+neg_relas])
            relas, _ = self._padding_cuda(pos_relas+neg_relas, maxlen, 'append', self.rela2id['PADDING'])
            pos_relas_text = [sum(prev+[rela], []) for prev, rela in zip(pos_prev_text, pos_rela_text)]
            neg_relas_text = [sum(prev+[rela], []) for prev, rela in zip(neg_prev_text, neg_rela_text)]
            maxlen = max([len(rela) for rela in pos_relas_text+neg_relas_text])
            relas_text, _ = self._padding_cuda(pos_relas_text+neg_relas_text, maxlen, 'append', self.word2id['PADDING'])
            prevs, prevs_text = [], []
        else:
            maxlen = max([len(rela) for rela in pos_rela+neg_rela])
            relas, _ = self._padding_cuda(pos_rela+neg_rela, maxlen, 'append', self.rela2id['PADDING'])
            maxlen = max([len(rela) for rela in pos_rela_text+neg_rela_text])
            relas_text, _ = self._padding_cuda(pos_rela_text+neg_rela_text, maxlen, 'append', self.word2id['PADDING'])
            if self.args.dynamic == 'flatten':
                # concat all previous
                prevs = [sum(prev, []) for prev in pos_prev+neg_prev]
                maxlen = max([len(prev) for prev in prevs])
                if maxlen > 0:
                    prevs, _ = self._padding_cuda(prevs, maxlen, 'append', self.rela2id['PADDING'])
                    prevs = [prevs]
                else:
                    prevs = []
                prevs_text = [sum(prev, []) for prev in pos_prev+neg_prev]
                maxlen = max([len(prev) for prev in prevs_text])
                if maxlen > 0:
                    prevs_text, _ = self._padding_cuda(prevs_text, maxlen, 'append', self.word2id['PADDING'])
                    prevs_text = [prevs_text]
                else:
                    prevs_text = []
            elif self.args.dynamic == 'recurrent':
                # make every candidates have same steps of previous
                maxlen = max([len(prev) for prev in pos_prev+neg_prev])
                prevs = [prev+[[]]*(maxlen-len(prev)) for prev in pos_prev+neg_prev]
                prevs_text = [prev+[[]]*(maxlen-len(prev)) for prev in pos_prev_text+neg_prev_text]
                # pad every previous respectively
                prevs = [self._padding_cuda(prev, max([len(p) for p in prev]), 'append', self.rela2id['PADDING'])[0] for prev in zip(*prevs)]
                prevs_text = [self._padding_cuda(prev, max([len(p) for p in prev]), 'append', self.word2id['PADDING'])[0] for prev in zip(*prevs_text)]
        score = model(ques, relas_text, relas, prevs_text, prevs)
        loss, acc = self._eval_metric(score)
        # readible format for score of all candidates : [(score, [token1, token2 ... ]) ... ]
        #rev_rela = [self.tokenizer.convert_ids_to_tokens(rela) for rela in pos_concat_rela+neg_concat_rela]
        rev_rela = [[self.id2rela[r] for r in rela] for rela in pos_rela+neg_rela]
        rela_score=list(zip(score.detach().cpu().numpy().tolist()[:], rev_rela[:]))
        return loss, acc, rela_score
    def _termination_decision(self, model, ques, tuples, next_tuples, movement):
        if movement == 'continue':
            pos_tuples = [t for t in next_tuples if t[-1] == 1]
            neg_tuples = [t for t in tuples if t[-1] == 1]
        elif movement == 'terminate':
            pos_tuples = [t for t in tuples if t[-1] == 1]
            neg_tuples = [t for t in next_tuples if t[-1] == 0]
        else:
            raise ValueError(f'Unknown movement:{movement} in UHop._termination_decision')
        # special case
        if len(pos_tuples) == 0 or len(neg_tuples) == 0:
            return 0, 1, 'noNegativeInTD'
        if len(pos_tuples) > 1:
            print('mutiple positive tuples!')
        if len(neg_tuples) > self.args.neg_sample:
            neg_tuples = neg_tuples[:self.args.neg_sample]
        # run model
        loss, acc, rela_score = self._single_UHop_step(model, ques, pos_tuples, neg_tuples)
        return loss, acc, rela_score
    def _single_step_rela_choose(self, model, ques, tuples):
        # make +/- pairs
        pos_tuples = [t for t in tuples if t[-1] == 1]
        neg_tuples = [t for t in tuples if t[-1] == 0]
        # special case
        if len(pos_tuples) == 0 or len(neg_tuples) == 0:
            return 0, 1, 'noNegativeInRC'
        if len(pos_tuples) > 1:
            print('mutiple positive tuples!')
        if len(neg_tuples) > self.args.neg_sample:
            neg_tuples = random.sample(neg_tuples, self.args.neg_sample)#neg_tuples[:self.args.neg_sample]
        # run model
        loss, acc, rela_score = self._single_UHop_step(model, ques, pos_tuples, neg_tuples)
        return loss, acc, rela_score
    def training_step(self,batch,batch_idx):
        index,ques,step_list = batch
        loss = 0
        step_count,loss_count=0,0
        rc_acc, rc_count = 0, 0
        td_acc, td_count = 0, 0
        labels, scores = [], []
        acc_list = []
        for i in range(len(step_list)-1):
            # TD if not the first nor the last step : continue
            if i > 0:
                step_loss, acc, score = self._termination_decision(self.model, ques, step_list[i-1], step_list[i], 'continue')
                if step_loss != 0:
                    step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'TD')
                    loss = loss + step_loss
                    step_count += 1
                loss_count += 1
                acc_list.append(acc)
                td_acc += acc; td_count += 1
                labels.append('<C>' if acc else '<T>')
                scores.append(score)
                if self.args.stop_when_err and acc != 1:
                    break
            # single_step RC
            step_loss, acc, score = self._single_step_rela_choose(self.model, ques, step_list[i])
            if step_loss != 0:
                step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'RC')
                loss = loss + step_loss
                step_count += 1
            loss_count += 1
            acc_list.append(acc)
            rc_acc += acc; rc_count += 1
            labels.append('<CR>' if acc else '<WR>')
            scores.append(score)
            if self.args.stop_when_err and acc != 1:
                break
        # last TD : terminate
        if (not self.args.stop_when_err) or all([x==1 for x in acc_list]):
            step_loss, acc, score = self._termination_decision(self.model, ques, step_list[-2], step_list[-1], 'terminate')
            if step_loss != 0:
                step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'TD')
                loss = loss + step_loss
                step_count += 1
            loss_count += 1
            acc_list.append(acc)
            td_acc += acc; td_count += 1
            labels.append('<T>' if acc else '<C>')
            scores.append(score)
        # step if not step_every_step
        acc = 1 if all([x==1 for x in acc_list]) else 0
        if not self.args.step_every_step:
            loss /= (step_count if step_count > 0 else 1)
        return {"loss":loss,"model":self.model, "loss_count":loss_count,"acc":acc, "scores":scores, "labels":'\t'.join(labels), "rc":(rc_acc, rc_count), "td":(td_acc, td_count)}

    def training_step_end(self,outputs):
        self.model = outputs["model"]
        loss = outputs["loss"]
        loss_count = outputs["loss_count"]
        acc = outputs["acc"]
        rc_acc = outputs["rc"]
        td_acc = outputs["td"]
        label = outputs["lables"]
        score = outputs["scores"]
        self.total_loss += loss; self.loss_count += loss_count
        self.total_acc += acc; self.acc_count += 1
        self.total_rc_acc += rc_acc[0]; self.rc_count += rc_acc[1]
        self.total_td_acc += td_acc[0]; self.td_count += td_acc[1]
        self.labels.append(label)
        self.scores.append(score)
    def on_validation_epoch_start(self):
        self.total_loss, self.total_acc = 0.0, 0.0
        self.loss_count, self.acc_count = 0, 0
        self.total_rc_acc, self.total_td_acc = 0.0, 0.0
        self.rc_count, self.td_count = 0, 0
        self.labels = []
        self.scores = []
    def on_validation_end(self):
        with open(f'{self.args.path}/scores_{100*total_acc/acc_count:.2f}.json', 'w') as f:
            json.dump(self.scores, f)
        with open(f'{self.args.path}/prediction.txt', 'w') as f:
            f.write('\n'.join(self.labels))

        
    def validation_step(self,batch,batch_idx):
        index,ques,step_list = batch
        loss = 0
        step_count,loss_count=0,0
        rc_acc, rc_count = 0, 0
        td_acc, td_count = 0, 0
        labels, scores = [], []
        acc_list = []
        for i in range(len(step_list)-1):
            # TD if not the first nor the last step : continue
            if i > 0:
                step_loss, acc, score = self._termination_decision(self.model, ques, step_list[i-1], step_list[i], 'continue')
                if step_loss != 0:
                    step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'TD')
                    loss = loss + step_loss
                    step_count += 1
                loss_count += 1
                acc_list.append(acc)
                td_acc += acc; td_count += 1
                labels.append('<C>' if acc else '<T>')
                scores.append(score)
                if self.args.stop_when_err and acc != 1:
                    break
            # single_step RC
            step_loss, acc, score = self._single_step_rela_choose(self.model, ques, step_list[i])
            if step_loss != 0:
                step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'RC')
                loss = loss + step_loss
                step_count += 1
            loss_count += 1
            acc_list.append(acc)
            rc_acc += acc; rc_count += 1
            labels.append('<CR>' if acc else '<WR>')
            scores.append(score)
            if self.args.stop_when_err and acc != 1:
                break
        # last TD : terminate
        if (not self.args.stop_when_err) or all([x==1 for x in acc_list]):
            step_loss, acc, score = self._termination_decision(self.model, ques, step_list[-2], step_list[-1], 'terminate')
            if step_loss != 0:
                step_loss *= self._loss_weight(i, len(step_list)-2, acc, 'TD')
                loss = loss + step_loss
                step_count += 1
            loss_count += 1
            acc_list.append(acc)
            td_acc += acc; td_count += 1
            labels.append('<T>' if acc else '<C>')
            scores.append(score)
        # step if not step_every_step
        acc = 1 if all([x==1 for x in acc_list]) else 0
        if not self.args.step_every_step:
            loss /= (step_count if step_count > 0 else 1)
        return {"loss":loss,"model":self.model, "loss_count":loss_count,"acc":acc, "scores":scores, "labels":'\t'.join(labels), "rc":(rc_acc, rc_count), "td":(td_acc, td_count)}
    def validation_step_end(self,outputs):
        self.model = outputs["model"]
        loss = outputs["loss"]
        loss_count = outputs["loss_count"]
        acc = outputs["acc"]
        label = outputs["label"]
        rc_acc = outputs["rc"]
        td_acc = outputs["td"]
        self.total_loss += loss; self.loss_count += loss_count
        self.total_acc += acc; self.acc_count += 1
        self.total_rc_acc += rc_acc[0]; self.rc_count += rc_acc[1]
        self.total_td_acc += td_acc[0]; self.td_count += td_acc[1]
        self.labels.append(label)
        self.scores.append(score)
    def configure_optimizers(self):
        return torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),
        lr=self.args.learning_rate, weight_decay=self.args.l2_norm, amsgrad=True)