import numpy as np
from tools_adam import loadEntityId, loadRelationId, loadTriplet
import logging
import scipy.sparse as sp
import random
import os
import sys
import time
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("output/log.txt")
handler.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

logger.info("Start print log")

"""
不出意外， 加了"_"的变量都表示entity_id(int) or relation_id(int)
"""
class transE(object):
    def __init__(self, triplet: list, relationId: dict, entityId: dict,
                 batch_size=64, learning_rate=0.01, dim=20, norm='L2', margin=2, seed=52,
                 validTriplet=None, testTriplet=None, batch_test=False, optimizer='AdaGrad', evaluator='Relation_Evaluator'):
        self.learning_rate = learning_rate  # 论文中{0.001, 0.01, 0.1}
        self.batch_size = batch_size
        self.norm = norm
        self.margin = margin  # 论文中{1, 2, 10}
        self.dim = dim  # 论文中{20, 50}
        self.raw_data = triplet
        self.data = []  # 储存id的triplet
        self.relationId = relationId  # {entity: id, ...}
        self.relationId_reverse = {v:k for k, v in relationId.items()}
        self.entityId = entityId
        self.entityList = list(entityId.values())
        self.relationList = list(relationId.values())
        self.entityIdReverse = {v:k for k, v in entityId.items()}
        self.entityMat = np.zeros((len(entityId), dim))
        self.relationMat = np.zeros((len(relationId), dim))

        self.data = encode2id(self.raw_data, self.entityId, self.relationId)

        mod = sys.modules[__name__]
        opt = getattr(mod, optimizer)
        self.optimizers = {'relationMat': opt(self.relationMat, learning_rate),
                           'entityMat': opt(self.entityMat, learning_rate)}
        if validTriplet and testTriplet:
            eval = getattr(mod, evaluator)
            self.evaluator = eval(encode2id(validTriplet, self.entityId, self.relationId),
                                  encode2id(validTriplet+triplet+testTriplet, self.entityId, self.relationId))
        else:
            self.evaluator = None

        self.loss = 0
        self.epoch = 0
        self.violations = 0
        self.batch_loss = 0
        self.seed = seed
        self.batch_test = batch_test

    def initialize(self):
        # 初始化向量
        # bnd = 6 / (self.dim ** 0.5)
        bnd = np.sqrt(6) / np.sqrt(self.entityMat.shape[0]+self.entityMat.shape[1])
        for e_ in self.entityList:
            self.entityMat[e_] = \
                np.random.uniform(-bnd, bnd, self.dim)
        # self.entityMat = normalize(self.entityMat)
        logger.info(f"entityVector初始化完成，维度是{self.entityMat.shape}")

        bnd = np.sqrt(6) / np.sqrt(self.relationMat.shape[0]+self.relationMat.shape[1])
        for p_ in self.relationList:
            self.relationMat[p_] = \
                np.random.uniform(-bnd, bnd, self.dim)
        # self.relationMat = normalize(self.relationMat)
        logger.info(f"relationVectorList初始化完成，维度是{self.relationMat.shape}")

    def train(self, epoch=1000):
        logger.info('训练开始')
        # np.random.seed(self.seed)
        self.initialize()
        for self.epoch in range(epoch):
            self.loss = 0
            self.violations = 0
            # 只标准化entity Vector
            # self.entityMat = L2_normalize(self.entityMat)
            random.shuffle(self.data)
            count = 0
            for j in range(0, len(self.data), self.batch_size):
                count += 1
                self.batch_loss = 0
                Sbatch = self.data[j: j + self.batch_size]
                pxs, nxs = [], []  # 初始化positive sample和负样本negative sample
                for h_, l_, t_ in Sbatch:
                    new_h_ = self.getRandomEntity(h_)
                    new_t_ = self.getRandomEntity(t_)
                    pxs.append((h_, l_, t_))
                    pxs.append((h_, l_, t_))
                    nxs.append((new_h_, l_, t_))
                    nxs.append((h_, l_, new_t_))
                self.transE_update(pxs, nxs)

                # if count % 1000 == 0:
                logger.info(f"完成{count}个minibatch，损失函数为{self.batch_loss}")

            self.x = logger.info(f"完成{self.epoch}轮训练，损失函数为{self.loss/count}, 超过margin的样本数量为{self.violations}")
            if self.evaluator:
                self.evaluator(self)

    def transE_update(self, pxs, nxs):
        """pxs: [(head_id, label_id, tail_id)]"""
        h_p, l_p, t_p = np.array(list(zip(*pxs)))
        h_n, l_n, t_n = np.array(list(zip(*nxs)))
        # distance_p.shape = (batch_size,)
        distance_p = self.distance(h_p, l_p, t_p)
        distance_n = self.distance(h_n, l_n, t_n)
        # 需要更新的sample的index array
        loss = np.maximum(self.margin + distance_p - distance_n, 0)

        loss_ = np.mean(loss)
        self.loss += loss_
        self.batch_loss = loss_
        ind = np.where(loss > 0)[0]

        self.violations += len(ind)
        if len(ind) == 0:  # 若没有样本需要更新向量，则返回
            return

        h_p2, l_p2, t_p2 = list(h_p[ind]), list(l_p[ind]), list(t_p[ind])
        h_n2, l_n2, t_n2 = list(h_n[ind]), list(l_n[ind]), list(t_n[ind])

        # step 1 : 计算d = (head + label - tail), gradient_p = (len(ind), dim)
        gradient_p = self.entityMat[h_p2] + self.relationMat[l_p2] - self.entityMat[t_p2]
        gradient_n = self.entityMat[h_n2] + self.relationMat[l_n2] - self.entityMat[t_n2]

        if self.norm == 'L1':
            # L1正则化的次梯度
            gradient_p = np.sign(gradient_p)
            gradient_n = np.sign(gradient_n)
        else:
            gradient_p = gradient_p*2
            gradient_n = gradient_n*2

        # 所有需要更新的entity_id list
        tot_entity = h_p2 + t_p2 + h_n2 + t_n2
        # step 2 : 计算一个中间矩阵M，方便整合positive和negative sample的所有entity（有重复的）到一个unique_entity
        unique_idx_e, M_e, tot_update_time_e = grad_sum_matrix(tot_entity)
        # step 3 : 计算每个entity的梯度
        # M.shape = (num_of_unique_entities, num_of_samples),
        # M2.shape = (num_of_samples, dim)
        # gradient.shape = (num_of_unique_entities, dim)除以n表示一个batch中的平均梯度，例如gradient.shape = (4, 3)
        # M = [[1,1,0,1],
        #      [0,0,1,0],
        #      [1,1,1,1]]
        # M2 = [[0.1, 0.3, 0.1],
        #       [-0.1, -0.3, -0.1],
        #       [0.18, 0.11, 0.43],
        #       [-0.18, -0.11, -0.43]]
        # gradient[0,0] = (0.1-0.1-0.18) / 3, 即0号entity在所有的正(负)样本的head(tail)中出现了3次（应该是偶数，我只是举例)
        M2_e = np.vstack((gradient_p, # 正样本的head的梯度
                        -gradient_p, # 正样本的tail的梯度
                        -gradient_n, # 负样本的head的梯度
                        gradient_n)) # 负样本的tail的梯度
        gradient_e = M_e.dot(M2_e) / tot_update_time_e

        # step 4 : 计算每个relation的梯度
        tot_relations = l_p2 + l_n2
        unique_idx_r, M_r, tot_update_time_r = grad_sum_matrix(tot_relations)
        M2_r = np.vstack((gradient_p, # 正样本的relation的梯度
                        -gradient_n) # 负样本的relation的梯度
                       )
        gradient_r = M_r.dot(M2_r) / tot_update_time_r
        self.optimizers['entityMat']._update(gradient_e, unique_idx_e)
        self.optimizers['relationMat']._update(gradient_r, unique_idx_r)

        normalize(self.entityMat, unique_idx_e)

    def getRandomEntity(self, entity_):
        random_entity_ = entity_
        while (random_entity_ == entity_):
            random_entity_ = random.sample(self.entityList, 1)[0]
        return random_entity_

    def distance(self, ss, ps, os):
        score = self.entityMat[ss] + self.relationMat[ps] - self.entityMat[os]
        if self.norm == 'L1':
            score = np.abs(score)
        else:
            score = score ** 2
        return np.sum(score, axis=1)

    def save(self, directory="output"):
        logger.info("保存Entity Vector")
        dir = os.path.join(directory, "entityVector.txt")
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(dir, 'w') as f:
            for e_ in self.entityList:
                f.write(self.entityIdReverse[e_] + "\t")
                f.write(str(self.entityMat[e_].tolist()))
                f.write("\n")
        logger.info("保存Relation Vector")
        dir = os.path.join(directory, "relationVector.txt")
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(dir, 'w') as f:
            for e_ in self.relationList:
                f.write(self.relationId_reverse[e_] + "\t")
                f.write(str(self.relationMat[e_].tolist()))
                f.write("\n")

    def _scores_r(self, h_, t_, l_):
        # 给定一个三元组，求替换了label的
        score = self.entityMat[h_] + self.relationMat - self.entityMat[t_]
        if self.norm == 'L1':
            score = np.abs(score)
        else:
            score = score ** 2

        return np.sum(score, axis=1)

    def _scores_e(self, h_, t_, l_, change_head=True):
        # 给定一个三元组，求替换了head or tail的
        if change_head:
            score = self.entityMat + self.relationMat[l_] - self.entityMat[l_]
        else:
            score = self.entityMat[h_] + self.relationMat[l_] - self.entityMat

        if self.norm == 'L1':
            score = np.abs(score)
        else:
            score = score ** 2

        return np.sum(score, axis=1)

    def loadModel(self, entity_path, replation_path):
        entityVectors = loadVectors(path=entity_path)
        relationVectors = loadVectors(path=replation_path)
        for e, v in entityVectors.items():
            self.entityMat[self.entityId[e]] = v
        for r, v in relationVectors.items():
            self.relationMat[self.relationId[r]] = v


class SGD(object):
    """
    SGD updates on a parameter
    """
    def __init__(self, param, learning_rate):
        self.param = param
        self.learning_rate = learning_rate

    def _update(self, g, idx):
        self.param[idx] -= self.learning_rate * g

class AdaGrad(object):

    def __init__(self, param, learning_rate):
        self.param = param
        self.learning_rate = learning_rate
        self.p2 = np.zeros_like(param)

    def _update(self, g, idx=None):
        self.p2[idx] += g * g
        H = np.maximum(np.sqrt(self.p2[idx]), 1e-7)
        self.param[idx] -= self.learning_rate * g / H

    def reset(self):
        self.p2 = np.zeros_like(self.p2)

class Relation_Evaluator(object):
    def __init__(self, test_triplet, tot_triplet):
        self.xs = test_triplet
        self.tt_h_t = self.convert_triple_into_dict(tot_triplet)

    def positions(self, mdl):
        pos = {}  # Raw Positions
        fpos = {}

        for s, p, o in self.xs:
            pos[p] = []
            fpos[p] = []

            scores_r = mdl._scores_r(s, o, p).flatten()
            sortidx_r = np.argsort(np.argsort(scores_r))
            pos[p].append(sortidx_r[p]+1)

            rm_idx = self.tt_h_t[s][o]
            rm_idx = [i for i in rm_idx if i != p]
            scores_r[rm_idx] = np.Inf
            sortidx_r = np.argsort(np.argsort(scores_r))
            fpos[p].append(sortidx_r[p]+1)
        return pos, fpos

    def __call__(self, model):
        pos_v, fpos_v = self.positions(model)
        mrr_valid = self.p_ranking_scores(pos_v, fpos_v, model.epoch, 'VALID')

    def p_ranking_scores(self, pos, fpos, epoch, txt):
        rpos = [p for k in pos.keys() for p in pos[k]]
        frpos = [p for k in fpos.keys() for p in fpos[k]]
        fmrr = self._print_pos(
            np.array(rpos),
            np.array(frpos),
            epoch, txt)
        return fmrr

    def _print_pos(self, pos, fpos, epoch, txt):
        mrr, mean_pos, hits = self.compute_scores(pos)
        fmrr, fmean_pos, fhits = self.compute_scores(fpos)
        logger.info(
            f"[{epoch: 3d}] {txt}: MRR = {mrr:.2f}/{fmrr:.2f}, "
            f"Mean Rank = {mean_pos:.2f}/{fmean_pos:.2f}, "
            f"Hits@1 = {hits[0]:.2f}/{fhits[0]:.2f}, "
            f"Hits@3 = {hits[1]:.2f}/{fhits[1]:.2f}, "
            f"Hits@10 = {hits[2]:.2f}/{fhits[2]:.2f}"
        )
        return fmrr

    def compute_scores(self, pos, hits=None):
        if hits is None:
            hits = [1,3,10]
        mrr = np.mean(1.0 / pos)
        mean_pos = np.mean(pos)
        hits_results = []
        for h in range(0, len(hits)):
            k = np.mean(pos <= hits[h])
            k2 = k.sum()
            hits_results.append(k2 * 100)
        return mrr, mean_pos, hits_results

    def convert_triple_into_dict(self, triplet):
        triple_dict = {}
        for h, l, t in triplet:
            if h in triple_dict.keys():
                if t in triple_dict[h].keys():
                    triple_dict[h][t].append(l)
                else:
                    triple_dict[h][t] = [l]
            else:
                triple_dict[h] = {t: [l]}

        return triple_dict

class Entity_Evaluator(object):
    def __init__(self, test_triplet, tot_triplet):
        # hh, ll, tt = list(zip(*test_triplet))
        self.test_triplet = test_triplet
        self.tt_h_l, self.tt_l_t = self.convert_triple_into_dict(tot_triplet)  # {head: {tail: [relation1, relation2, ...]}}

    def positions(self, model):
        pos = []  # 存每一个e的rank
        fpos = [] # Filtered Position

        for head, label, tail in self.test_triplet:
            # *************  换head的损失函数  ********************
            scores_e_h = model._scores_e(head, tail, label, change_head=True).flatten()
            sortidx_e_h = np.argsort(np.argsort(scores_e_h))
            pos.append(sortidx_e_h[head] + 1)

            rm_idx = self.tt_l_t[label][tail]
            rm_idx = [i for i in rm_idx if i != head]
            scores_e_h[rm_idx] = np.Inf
            sortidx_e_h = np.argsort(np.argsort(scores_e_h))
            fpos.append(sortidx_e_h[head] + 1)

            # *************  换tail的损失函数  ********************
            scores_e_t = model._scores_e(head, tail, label, change_head=False).flatten()
            sortidx_e_t = np.argsort(np.argsort(scores_e_t))
            pos.append(sortidx_e_t[tail] + 1)

            rm_idx = self.tt_h_l[head][label]
            rm_idx = [i for i in rm_idx if i != tail]
            scores_e_t[rm_idx] = np.Inf
            sortidx_e_t = np.argsort(np.argsort(scores_e_t))
            fpos.append(sortidx_e_t[tail] + 1)

        return pos, fpos

    def __call__(self, model):
        pos_v, fpos_v = self.positions(model)
        mrr_valid = self.p_ranking_scores(pos_v, fpos_v, model.epoch, 'VALID')

    def p_ranking_scores(self, pos, fpos, epoch, txt):
        rpos = pos
        frpos = fpos
        fmrr = self._print_pos(
            np.array(rpos),
            np.array(frpos),
            epoch, txt)
        return fmrr

    def _print_pos(self, pos, fpos, epoch, txt):
        mrr, mean_pos, hits = self.compute_scores(pos)
        fmrr, fmean_pos, fhits = self.compute_scores(fpos)
        logger.info(
            "[%3d] %s: MRR = %.2f/%.2f, Mean Rank = %.2f/%.2f, Hits@1 = %.2f/%.2f, Hits@3 = %.2f/%.2f, Hits@10 = %.2f/%.2f" %
            (epoch, txt, mrr, fmrr, mean_pos, fmean_pos, hits[0], fhits[0], hits[1], fhits[1], hits[2], fhits[2])
        )
        return fmrr

    def compute_scores(self, pos, hits=[1, 3, 10]):
        mrr = np.mean(1.0 / pos)
        mean_pos = np.mean(pos)
        hits_results = []
        for h in range(0, len(hits)):
            hits_results.append(np.mean(pos <= hits[h]).sum() * 100)
        return mrr, mean_pos, hits_results

    def convert_triple_into_dict(self, triplet):
        h_l_dict = {}
        l_t_dict = {}
        for head, label, tail in triplet:
            if head in h_l_dict.keys():
                if label in h_l_dict[head].keys():
                    h_l_dict[head][label].append(tail)
                else:
                    h_l_dict[head][label] = [tail]
            else:
                h_l_dict[head] = {label: [tail]}

            if label in l_t_dict.keys():
                if tail in l_t_dict[label].keys():
                    l_t_dict[label][tail].append(head)
                else:
                    l_t_dict[label][tail] = [head]
            else:
                l_t_dict[label] = {tail: [head]}

        return h_l_dict, l_t_dict

def grad_sum_matrix(idx):
    # unique_idx: unique entity_id; idx_inverse: index for each entity in idx in the unique_idx
    unique_idx, idx_inverse = np.unique(idx, return_inverse=True)
    # 需要更新的entity的数量(包括重复的，比如一个一个batch里面的多个sample包含同一个entity)
    sz = len(idx_inverse)
    # 生成一个系数矩阵 M.shape = (num_of_unique_entities, num_of_samples)
    # M = [[1,1,0,1,0],
    #      [0,0,1,0,1],
    #      [1,1,1,1,1]]
    # 表示第一个sample需要更新第0,2号的entity; 第三个sample需要更新第1,2号entity.
    M = sp.coo_matrix((np.ones(sz), (idx_inverse, np.arange(sz)))).tocsr()  # M.shape = (num_of_unique_idx, tot_sample)
    # normalize summation matrix so that each row sums to one
    tot_update_time = np.array(M.sum(axis=1))  # shape = (num_of_unique_entities, ) 比如M中第0号entity需要更新3次
    return unique_idx, M, tot_update_time

def normalize(M, idx=None):
    if idx is None:
        M /= np.sqrt(np.sum(M ** 2, axis=1))[:, np.newaxis]
    else:
        nrm = np.sqrt(np.sum(M[idx, :] ** 2, axis=1))[:, np.newaxis]
        M[idx, :] /= nrm
    return M

def loadVectors(path="output/entityVector.txt"):
    vectorDict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            k, v = line.strip().split("\t")
            v = np.array(eval(v))
            vectorDict[k] = v
    return vectorDict

def encode2id(triplet, entityId, relationId):
    data = []
    for h, l, t in triplet:
        data.append((entityId[h], relationId[l], entityId[t]))

    return data

def init_nunif(sz):
    """
    Normalized uniform initialization

    See Glorot X., Bengio Y.: "Understanding the difficulty of training
    deep feedforward neural networks". AISTATS, 2010
    """
    bnd = np.sqrt(6) / np.sqrt(sz[0] + sz[1])
    p = np.random.uniform(low=-bnd, high=bnd, size=sz)
    return np.squeeze(p)

def init_nunif2(sz):

    bnd = np.sqrt(6) / np.sqrt(sz[0] + sz[1])
    p = np.random.uniform(low=-bnd, high=bnd, size=sz)
    return np.squeeze(p)


if __name__ == "__main__":
    # 训练模型
    entity2Id = loadEntityId()
    relation2Id = loadRelationId()
    triplet = loadTriplet("data/freebase_mtr100_mte100-train.txt")
    valid_triplet = loadTriplet("data/freebase_mtr100_mte100-valid.txt")
    test_triplet = loadTriplet("data/freebase_mtr100_mte100-test.txt")
    model = transE(triplet, relation2Id, entity2Id,
                   learning_rate=0.01, dim=100, batch_size=20000,
                   margin=1, norm='L1', optimizer='AdaGrad',
                   validTriplet=valid_triplet,
                   testTriplet=test_triplet,
                   evaluator='Relation_Evaluator')
    model.train(150)  # 论文使用的1000次，early_stopping using the mean predicted ranks on the validation set.
    model.save()

    # # 评估模型
    entityVectors = loadVectors(path="output/entityVector.txt")
    relationVectors = loadVectors(path="output/relationVector.txt")
    testTriplet = loadTriplet("data/freebase_mtr100_mte100-test.txt")
    # model.loadModel("output/entityVector.txt", "output/relationVector.txt")
    print(f"测试集一共有个{len(testTriplet)}个")
    testTriplet_ = encode2id(testTriplet, model.entityId, model.relationId)
    t1 = time.time()
    eval = Relation_Evaluator(testTriplet_, model.data+testTriplet_)
    eval(model)



    
    


