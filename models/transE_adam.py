import numpy as np
from tools_adam import loadEntityId, loadRelationId, loadTriplet
import random
import os
class transE(object):
    def __init__(self, triplet: list, relationId: dict, entityId: dict,
                 batch_size=64, learning_rate=0.01, dim=20, norm='L2', margin=2, seed=52):
        self.learning_rate = learning_rate  # 论文中{0.001, 0.01, 0.1}
        self.batch_size = batch_size
        self.norm = norm
        self.margin = margin  # 论文中{1, 2, 10}
        self.dim = dim  # 论文中{20, 50}
        self.data = triplet
        self.relationId = relationId  # {entity: id, ...}
        self.relationId_reverse = {v:k for k, v in relationId.items()}
        self.entityId = entityId
        self.entityList = list(entityId.keys())
        self.relationList = list(relationId.keys())
        self.entityIdReverse = {v:k for k, v in entityId.items()}
        self.entityMat = np.zeros((len(entityId), dim))  #
        self.relationMat = np.zeros((len(relationId), dim))

        self.loss = 0
        self.batch_loss = 0
        self.seed = seed

    def initialize(self):
        # 初始化向量
        for e in self.entityId.keys():
            self.entityMat[self.entityId[e],:] = \
                np.random.uniform(-6/(self.dim**0.5), 6/(self.dim**0.5), self.dim)
                

        # self.entityMat = L2_normalize(self.entityMat)  
        print(f"entityVector初始化完成，维度是{self.entityMat.shape}")

        for e in self.relationId.keys():
            self.relationMat[self.relationId[e], :] = \
                np.random.uniform(-6 / (self.dim ** 0.5), 6 / (self.dim ** 0.5), self.dim)
        self.relationMat = L2_normalize(self.relationMat)       # 首先只标准化relationMat
        print(f"relationVectorList初始化完成，维度是{self.relationMat.shape}")

    def train(self, epoch=1000):
        print('训练开始')
        np.random.seed(self.seed)
        self.initialize()
        # 第一循环是数据的遍历次数，training_times
        for i in range(epoch):
            self.loss = 0
            # 只标准化entity Vector
            self.entityMat = L2_normalize(self.entityMat)
            # 打乱数据的顺序
            random.shuffle(self.data)
            count = 0
            # 第二重循环是每一次里面的mini_batch
            for j in range(0, len(self.data), self.batch_size):
                count += 1
                self.batch_loss = 0
                Sbatch = self.data[j:j+self.batch_size]
                Tbatch = set() # 数据形式： {(三元组, 负样本，是否换的是head的flag变量}
                # 负采样
                for h, l, t in Sbatch:  # (h, l, t)对应原论文的(h, l, t) -> (head, label, tail)
                    p = random.uniform(0, 1)
                    if p > 0.5:  # 换head
                        new_head = self.getRandomEntity(h)
                        Tbatch.add(((h, l, t), new_head, 1))
                    else:
                        new_tail = self.getRandomEntity(t)
                        Tbatch.add(((h, l, t), new_tail, 0))
                # 更新权重
                for S, new_entity, isHeadCurrupted in Tbatch:
                    # 正样本向量
                    h = self.entityMat[self.entityId[S[0]], :]
                    t = self.entityMat[self.entityId[S[2]], :]
                    l = self.relationMat[self.relationId[S[1]], :]
                    # 负样本entity向量
                    swap = self.entityMat[self.entityId[new_entity], :]
                    # 计算损失函数
                    d1 = cal_distance(vector1=h+l, vector2=t, norm=self.norm)
                    if isHeadCurrupted:
                        d2 = cal_distance(vector1=swap+l, vector2=t, norm=self.norm)
                    else:
                        d2 = cal_distance(vector1=h+l, vector2=swap, norm=self.norm)
                    loss = max(0, self.margin+d1-d2)
                    self.batch_loss += loss
                    self.loss += loss
                    # 计算梯度
                    if loss > 0:
                        if self.norm == 'L2':
                            if isHeadCurrupted:
                                dh = 2*h + 2*l - 2*t
                                dl = 2*h - 2*swap
                                dt = -2*h + 2*swap
                                dswap = -2*swap - 2*l + 2*t
                            else:
                                dh = -2*t + 2*swap
                                dl = -2*t + 2*swap
                                dt = -2*l - 2*h + 2*t
                                dswap = 2*l + 2*h - 2*swap

                        elif self.norm == 'L1':
                            pass
                        # 更新向量, 
                        # version 1: 该版本是对于每一个batch里面的sample，都会更新向量，需要学习率很小来控制收敛
                        self.entityMat[self.entityId[S[0]], :] -= self.learning_rate * dh
                        self.entityMat[self.entityId[S[2]], :] -= self.learning_rate * dt
                        self.relationMat[self.relationId[S[1]], :] -= self.learning_rate * dl
                        self.entityMat[self.entityId[new_entity], :] -= self.learning_rate * dswap
                    else:
                        # hinge loss在小于0的时候梯度为0，所以不用更新向量
                        pass

                # if count % 1000 == 0:
                #     print(f"完成{count}个minibatch，损失函数为{self.batch_loss/self.batch_size}")
        
            print(f"完成{i}轮训练，损失函数为{self.loss/len(self.data)}")


    def getRandomEntity(self, entity):
        random_entity = entity
        while(random_entity == entity):
            random_entity = random.sample(self.entityList, 1)
        return random_entity[0]

    def save(self, directory="output"):
        print("保存Entity Vector")
        dir = os.path.join(directory, "entityVector.txt")
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(dir, 'w') as f:
            for entity in self.entityList:
                f.write(entity+"\t") 
                f.write(str(self.entityMat[self.entityId[entity], :].tolist()))
                f.write("\n")
        
        dir = os.path.join(directory, "relationVector.txt")
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(dir, 'w') as f:
            for entity in self.relationList:
                f.write(entity+"\t") 
                f.write(str(self.relationMat[self.relationId[entity], :].tolist()))
                f.write("\n")



def L2_normalize(vector):
    if np.ndim(vector) == 1:
        norm = np.linalg.norm(vector, ord=2)
    else:
        norm = np.linalg.norm(vector, ord=2, axis=1)[:,np.newaxis]
    return vector / norm

def cal_distance(vector1, vector2, norm='L2'):
    if norm == 'L2':
        return np.linalg.norm(vector1-vector2, ord=2)
    if norm == 'L1':
        return np.linalg.norm(vector1-vector2, ord=1)

def loadVectors(path="output/entityVector.txt"):
    vectorDict = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            k, v = line.strip().split("\t")
            v = np.array(eval(v))
            vectorDict[k] = v
    return vectorDict

def evaluate(test_triplet, entity_vectors, relation_vectors, hits=10):
    entity_list = list(entity_vectors.keys())
    head_rank = []
    tail_rank = []
    # For each test triplet, the head is removed and replaced by each of the entities of the dictionary in turn
    i = 0
    for (h, l, t) in test_triplet:
        i += 1
        if i % 100 == 0:
            print(f"已经评估完了{i/len(test_triplet): .2f}%个样本")
        # 转换成向量
        h_vec, l_vec, t_vec = entity_vectors[h], relation_vectors[l], entity_vectors[t]
        # 替换head部分
        tot_head = [(e, cal_distance(entity_vectors[e]+l_vec, t_vec)) for e in entity_list]  
        tot_head = sorted(tot_head, key=lambda x: x[1], reverse=True)  # 按照距离降序排列
        rank = 0
        for e, d in tot_head:
            rank += 1
            if h == e:
                break
        head_rank.append(rank)

        # 替换tail部分
        tot_tail = [(e, cal_distance(h_vec+l_vec, entity_vectors[e])) for e in entity_list]  
        tot_tail = sorted(tot_tail, key=lambda x: x[1], reverse=True)  # 按照距离降序排列
        rank=0
        for e, d in tot_tail:
            rank += 1
            if t == e:
                break
        tail_rank.append(rank)

    tot_rank = np.array(tail_rank + head_rank)
    mean_rank = np.sum(tot_rank) / len(tot_rank)
    hit = np.sum(tot_rank < hits) / len(tot_rank)
    return mean_rank, hit

def evaluate2(test_triplet, entity_vectors, relation_vectors, hits=10):
    # 使用向量化加快运算速度
    entity_list = list(entity_vectors.keys())
    entity_id = {e:i for i, e in enumerate(entity_list)}  # 储存entity位置
    v = [entity_vectors[e] for e in entity_list]
    entityMat = np.stack(v, axis=1)  # 每列是一个e的embedding
    outputMat = np.zeros((len(test_triplet), len(entity_list)))  # output[i, j]则是第i个样本和第j个entity的distance
    right_index = []
    i = 0
    for (h, l, t) in test_triplet:
        vec = entity_vectors[h] + relation_vectors[l]
        mat = np.repeat(vec, len(entity_list), axis=1)  # 将embedding横向扩充
        mat -= entityMat  # 替换tail, 就是h+l-t
        d = np.diag(np.matmul(mat.T, mat))  # 对角线即为L2-norm距离
        outputMat[i, :] = d 
        right_index.append(entity_id[t])  #记录正确的entity的位置
        i += 1
    # 替换head的形式还没有进行计算33

    rank_mat = np.argsort(np.argsort(outputMat, axis=1), axis=1)
    rank_array = rank_mat[:, right_index]
    mean_rank = np.mean(rank_array)
    hit10 = np.mean(rank_array<hist)
    return mean_rank, hit10

if __name__ == "__main__":
    # # 训练模型
    # entity2Id = loadEntityId()
    # relation2Id = loadRelationId()
    # triplet = loadTriplet("data/freebase_mtr100_mte100-train.txt")
    # model = transE(triplet, relation2Id, entity2Id, learning_rate=0.01, dim=50, batch_size=100)
    # model.train(50)  # 论文使用的1000次，early_stopping using the mean predicted ranks on the validation set.
    # model.save()

    # 评估模型
    entityVectors = loadVectors(path="output/entityVector.txt")
    relationVectors = loadVectors(path="output/relationVector.txt")
    testTriplet = loadTriplet("data/freebase_mtr100_mte100-test.txt")
    mean_rank, hit10 = evaluate2(testTriplet, entityVectors, relationVectors, hits=10)
    print(f"mean rank: {mean_rank}, HITs10: {hit10}")
    
    


