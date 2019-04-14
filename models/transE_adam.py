import numpy as np
from tools_adam import loadEntityId, loadRelationId, loadTriplet
import random
import os
class transE(object):
    def __init__(self, triplet: list, relationId: dict, entityId: dict,
                 batch_size=64, learning_rate=0.01, dim=20, norm='L2', margin=2, seed=52,
                 evaluation_metric=None, validTriplet=None, batch_test=False):
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
        self.metric = evaluation_metric
        self.batch_test = batch_test
        if self.metric:
            assert validTriplet is not None
            self.validTriplet = validTriplet

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
                Sbatch = self.data[j: j+self.batch_size]
                Tbatch = set() # 数据形式： {(三元组, 负样本，是否换的是head的flag变量}
                # 负采样
                for h, l, t in Sbatch:  # (h, l, t)对应原论文的(h, l, t) -> (head, label, tail)
                    p = random.uniform(0, 1)
                    new = self.getRandomEntity(h)
                    if p > 0.5:  # 换head
                        Tbatch.add(((h, l, t), (new, l, t)))
                    else:
                        Tbatch.add(((h, l, t), (h,l,new)))
                # 更新权重
                for S1, S2 in Tbatch:
                    # 正样本向量
                    h1 = self.entityMat[self.entityId[S1[0]], :]
                    t1 = self.entityMat[self.entityId[S1[2]], :]
                    l1 = self.relationMat[self.relationId[S1[1]], :]
                    # 负样本entity向量
                    h2 = self.entityMat[self.entityId[S2[0]], :]
                    t2 = self.entityMat[self.entityId[S2[2]], :]
                    l2 = self.relationMat[self.relationId[S2[1]], :]
                    # 计算损失函数
                    d1 = cal_distance(vector1=h1+l1, vector2=t1, norm=self.norm)
                    d2 = cal_distance(vector1=h2+l2, vector2=t2, norm=self.norm)
                    loss = max(0, self.margin+d1-d2)
                    self.batch_loss += loss
                    self.loss += loss
                    # 计算梯度
                    if loss > 0:
                        if self.norm == 'L2':
                            gradient1 = 2 * (h1 + l1 - t1)
                            gradient2 = -2 * (h2 + l2 - t2)
                            dh1 = gradient1
                            dl1 = gradient1
                            dt1 = -gradient1

                            dh2 = gradient2
                            dl2 = gradient2
                            dt2 = -gradient2

                        elif self.norm == 'L1':
                            pass
                        # 更新正样本向量
                        self.entityMat[self.entityId[S1[0]], :] -= self.learning_rate * dh1
                        self.entityMat[self.entityId[S1[2]], :] -= self.learning_rate * dt1
                        self.relationMat[self.relationId[S1[1]], :] -= self.learning_rate * dl1
                        # 更新负样本向量
                        self.entityMat[self.entityId[S2[0]], :] -= self.learning_rate * dh2
                        self.entityMat[self.entityId[S2[2]], :] -= self.learning_rate * dt2
                        self.relationMat[self.relationId[S2[1]], :] -= self.learning_rate * dl2
                    else:
                        # hinge loss在小于0的时候梯度为0，所以不用更新向量
                        pass

                # if count % 1000 == 0:
                #     print(f"完成{count}个minibatch，损失函数为{self.batch_loss/self.batch_size}")
        
            if self.metric:
                m_valid = self.evaluate(self.validTriplet)
                m_train = self.evaluate(self.data)
                print(f"完成{i}轮训练，损失函数为{self.loss/len(self.data)}, {self.metric} for train: {m_train}, {self.metric} for valid: {m_valid}")
            else:
                print(f"完成{i}轮训练，损失函数为{self.loss / len(self.data)}")

    def evaluate(self, test_triplet):
        entity_list = self.entityList
        entity_id = self.entityId
        entityMat = self.entityMat.T  # 每列是一个e的embedding
        outputMat1 = np.zeros((len(test_triplet), len(entity_list)))  # 换tail: output[i, j]则是第i个样本和第j个entity的distance
        outputMat2 = np.zeros((len(test_triplet), len(entity_list)))  # 换head: output[i, j]则是第i个样本和第j个entity的distance
        right_tail_index = []
        right_head_index = []
        i = 0
        for (h, l, t) in test_triplet:
            # 转换成向量
            h_vec, l_vec, t_vec = self.entityMat[self.entityId[h], :], \
                                  self.relationMat[self.relationId[l], :], \
                                  self.entityMat[self.entityId[t], :]

            right_tail_index.append(entity_id[t])  # 记录正确的tail的位置
            right_head_index.append(entity_id[h])  # 记录正确的head的位置
            vec1 = h_vec + l_vec
            vec2 = l_vec - t_vec
            newMat1 = vec1[:, np.newaxis] - entityMat  # 换tail, 利用broadcast机制, 每列是一个h+l-t的向量,
            newMat2 = entityMat + vec2[:, np.newaxis]  # 换head, 利用broadcast机制, 每列是一个h+l-t的向量
            d1 = np.linalg.norm(newMat1, ord=2, axis=0)
            d2 = np.linalg.norm(newMat2, ord=2, axis=0)
            outputMat1[i, :] = d1
            outputMat2[i, :] = d2
            i += 1

        rank_mat1 = np.argsort(np.argsort(outputMat1, axis=1), axis=1)
        rank_mat2 = np.argsort(np.argsort(outputMat2, axis=1), axis=1)
        rank_array1 = [rank_mat1[i, ind] for i, ind in
                       enumerate(right_tail_index)]  # 储存每一个样本i的rank值（包括换head, tial，所以样本值是要翻倍的）
        rank_array2 = [rank_mat2[i, ind] for i, ind in
                       enumerate(right_head_index)]  # 储存每一个样本i的rank值（包括换head, tial，所以样本值是要翻倍的）
        if self.metric == "mean-rank":
            return np.mean(rank_array1 + rank_array2)
        if self.metric == "hit10":
            return np.mean([1 if i < 10 else 0 for i in rank_array1 + rank_array2])

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


def L2_normalize(vector, axis=1):
    if np.ndim(vector) == 1:
        norm = np.linalg.norm(vector, ord=2)
    else:
        norm = np.linalg.norm(vector, ord=2, axis=axis)[:,np.newaxis]
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

def evaluate1(test_triplet, entity_vectors, relation_vectors, hits=10, limit=None):
    if limit:
        # testTriplet = random.sample(test_triplet, limit)
        test_triplet = test_triplet[:limit]

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
        tot_head = sorted(tot_head, key=lambda x: x[1], reverse=False)  # 按照距离降序排列
        rank = 0
        for e, d in tot_head:
            rank += 1
            if h == e:
                break
        head_rank.append(rank)

        # 替换tail部分
        tot_tail = [(e, cal_distance(h_vec+l_vec, entity_vectors[e])) for e in entity_list]  
        tot_tail = sorted(tot_tail, key=lambda x: x[1], reverse=False)  # 按照距离降序排列
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

def evaluate2(test_triplet, entity_vectors, relation_vectors, hits=10, limit=None):
    # 使用向量化加快运算速度

    if limit:
        # test_triplet = random.sample(test_triplet, limit)
        test_triplet = test_triplet[:limit]

    # entity_list = list(entity_list)
    entity_list = list(entity_vectors.keys())
    entity_id = {e:i for i, e in enumerate(entity_list)}  # 储存entity位置
    entityMat = np.stack([entity_vectors[e] for e in entity_list], axis=1)  # 每列是一个e的embedding
    outputMat1 = np.zeros((len(test_triplet), len(entity_list)))  # 换tail: output[i, j]则是第i个样本和第j个entity的distance
    outputMat2 = np.zeros((len(test_triplet), len(entity_list)))  # 换head: output[i, j]则是第i个样本和第j个entity的distance
    right_tail_index = []
    right_head_index = []
    i = 0
    for (h, l, t) in test_triplet:
        right_tail_index.append(entity_id[t])  #记录正确的tail的位置
        right_head_index.append(entity_id[h])  #记录正确的head的位置
        vec1 = entity_vectors[h] + relation_vectors[l]
        vec2 = relation_vectors[l] - entity_vectors[t]
        newMat1 = vec1[:, np.newaxis] - entityMat  # 换tail, 利用broadcast机制, 每列是一个h+l-t的向量,
        newMat2 = entityMat + vec2[:, np.newaxis]  # 换head, 利用broadcast机制, 每列是一个h+l-t的向量
        d1 = np.linalg.norm(newMat1, ord=2, axis=0)
        d2 = np.linalg.norm(newMat2, ord=2, axis=0)
        outputMat1[i, :] = d1
        outputMat2[i, :] = d2
        i += 1
        if i % 1000==0:
            print(f"已经完成{100*i/len(test_triplet):.2f}%样本的评估")

    rank_mat1 = np.argsort(np.argsort(outputMat1, axis=1), axis=1)
    rank_mat2 = np.argsort(np.argsort(outputMat2, axis=1), axis=1)
    rank_array1 = [rank_mat1[i, ind] for i, ind in enumerate(right_tail_index)]  # 储存每一个样本i的rank值（包括换head, tial，所以样本值是要翻倍的）
    rank_array2 = [rank_mat2[i, ind] for i, ind in enumerate(right_head_index)]  # 储存每一个样本i的rank值（包括换head, tial，所以样本值是要翻倍的）
    mean_rank = np.mean(rank_array1+rank_array2)
    hit10 = np.mean([1 if i < hits else 0 for i in rank_array1+rank_array2])
    return mean_rank, hit10

if __name__ == "__main__":
    # 训练模型
    # entity2Id = loadEntityId()
    # relation2Id = loadRelationId()
    # triplet = loadTriplet("data/freebase_mtr100_mte100-train.txt")
    # valid_triplet = loadTriplet("data/freebase_mtr100_mte100-valid.txt")
    # model = transE(triplet, relation2Id, entity2Id,
    #                learning_rate=0.001, dim=50, batch_size=100,
    #                margin=1)
    # model.train(500)  # 论文使用的1000次，early_stopping using the mean predicted ranks on the validation set.
    # model.save()

    # 评估模型
    entityVectors = loadVectors(path="output/entityVector.txt")
    relationVectors = loadVectors(path="output/relationVector.txt")
    testTriplet = loadTriplet("data/freebase_mtr100_mte100-test.txt")
    import time
    t1 = time.time()
    print(f"测试集一共有个{len(testTriplet)}个")
    mean_rank, hit10 = evaluate2(testTriplet, entityVectors, relationVectors, hits=10, limit=1000)
    print(f"time: {time.time()-t1}")
    print(f"mean rank: {mean_rank}, HITs10: {hit10}")
    # t1 = time.time()
    # mean_rank, hit10 = evaluate1(testTriplet, entityVectors, relationVectors, hits=10, limit=100)
    # print(f"time: {time.time()-t1}")
    # print(f"mean rank: {mean_rank}, HITs10: {hit10}")
    
    


