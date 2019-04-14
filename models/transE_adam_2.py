import numpy as np
from tools_adam import loadEntityId, loadRelationId, loadTriplet
import scipy.sparse as sp
import random
import os
import sys

"""
不出意外， 加了"_"的变量都表示entity_id(int) or relation_id(int)
"""
class transE(object):
    def __init__(self, triplet: list, relationId: dict, entityId: dict,
                 batch_size=64, learning_rate=0.01, dim=20, norm='L2', margin=2, seed=52,
                 evaluation_metric=None, validTriplet=None, batch_test=False, optimizer='AdaGrad'):
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
        mod = sys.modules[__name__]
        opt = getattr(mod, optimizer)
        self.optimizers = {'relationMat': opt(self.relationMat, learning_rate),
                           'entityMat': opt(self.entityMat, learning_rate)}
        self.loss = 0
        self.violations = 0
        self.batch_loss = 0
        self.seed = seed
        self.metric = evaluation_metric
        self.batch_test = batch_test
        if self.metric:
            assert validTriplet is not None
            self.validTriplet = validTriplet

    def initialize(self):
        # 把三元组转换成id形式
        for h, l, t in self.raw_data:
            self.data.append((self.entityId[h], self.relationId[l], self.entityId[t]))

        # 初始化向量
        for e_ in self.entityList:
            self.entityMat[e_] = \
                np.random.uniform(-6/(self.dim**0.5), 6/(self.dim**0.5), self.dim)
        self.entityMat = normalize(self.entityMat)
        print(f"entityVector初始化完成，维度是{self.entityMat.shape}")

        for p_ in self.relationList:
            self.relationMat[p_] = \
                np.random.uniform(-6 / (self.dim ** 0.5), 6 / (self.dim ** 0.5), self.dim)
        self.relationMat = normalize(self.relationMat)
        print(f"relationVectorList初始化完成，维度是{self.relationMat.shape}")

    def train(self, epoch=1000):
        print('训练开始')
        np.random.seed(self.seed)
        self.initialize()
        for i in range(epoch):
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
                #     print(f"完成{count}个minibatch，损失函数为{self.batch_loss/self.batch_size}")

            print(f"完成{i}轮训练，损失函数为{self.loss / len(self.data)}, 超过margin的样本数量为{self.violations}")

    def transE_update(self, pxs, nxs):
        """pxs: [(head_id, label_id, tail_id)]"""
        h_p, l_p, t_p = np.array(list(zip(*pxs)))
        h_n, l_n, t_n = np.array(list(zip(*nxs)))
        # distance_p.shape = (batch_size,)
        distance_p = self.distance(h_p, l_p, t_p)
        distance_n = self.distance(h_n, l_n, t_n)
        # 需要更新的sample的index array
        loss = np.maximum(self.margin + distance_p - distance_n, 0)

        loss_ = np.sum(loss)
        self.loss += loss_
        self.batch_loss = loss_
        ind = np.where(loss > 0)[0]

        self.violations += len(ind)
        if len(ind) == 0:  # 若没有样本需要更新向量，则返回
            return

        h_p2, l_p2, t_p2 = list(h_p[ind]), list(l_p[ind]), list(t_p[ind])
        h_n2, l_n2, t_n2 = list(h_n[ind]), list(l_n[ind]), list(t_n[ind])

        # step 1 : 计算d = (head + label - tail), gradient_p = (len(ind), dim)
        gradient_p = self.entityMat[t_p2] - self.relationMat[l_p2] - self.entityMat[h_p2]
        gradient_n = self.entityMat[t_n2] - self.relationMat[l_n2] - self.entityMat[h_n2]
        if self.norm == 'L1':
            # L1正则化的次梯度
            gradient_p = np.sign(-gradient_p)
            gradient_n = -np.sign(-gradient_n)
        else:
            gradient_p = -gradient_p*2
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
                        gradient_n, # 负样本的head的梯度
                        -gradient_n)) # 负样本的tail的梯度
        gradient_e = M_e.dot(M2_e) / tot_update_time_e

        # step 4 : 计算每个relation的梯度
        tot_relations = l_p2 + l_n2
        unique_idx_r, M_r, tot_update_time_r = grad_sum_matrix(tot_relations)
        M2_r = np.vstack((gradient_p, # 正样本的relation的梯度
                        gradient_n) # 负样本的relation的梯度
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

def L2_normalize(vector, axis=1):
    if np.ndim(vector) == 1:
        norm = np.linalg.norm(vector, ord=2)
    else:
        norm = np.linalg.norm(vector, ord=2, axis=axis)[:,np.newaxis]
    return vector / norm

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

if __name__ == "__main__":
    # 训练模型
    entity2Id = loadEntityId()
    relation2Id = loadRelationId()
    triplet = loadTriplet("data/freebase_mtr100_mte100-train.txt")
    valid_triplet = loadTriplet("data/freebase_mtr100_mte100-valid.txt")
    model = transE(triplet, relation2Id, entity2Id,
                   learning_rate=0.01, dim=100, batch_size=1000,
                   margin=1, norm='L2', optimizer='AdaGrad')
    model.train(200)  # 论文使用的1000次，early_stopping using the mean predicted ranks on the validation set.
    # model.save()

    # 评估模型
    # entityVectors = loadVectors(path="output/entityVector.txt")
    # relationVectors = loadVectors(path="output/relationVector.txt")
    # testTriplet = loadTriplet("data/freebase_mtr100_mte100-test.txt")
    # import time
    # t1 = time.time()
    # print(f"测试集一共有个{len(testTriplet)}个")
    # mean_rank, hit10 = evaluate2(testTriplet, entityVectors, relationVectors, hits=10, limit=1000)
    # print(f"time: {time.time()-t1}")
    # print(f"mean rank: {mean_rank}, HITs10: {hit10}")

    
    


