from random import uniform, sample
from numpy import *
from copy import deepcopy

class TransE:
    def __init__(self, entityList, relationList, tripleList, margin = 1, learingRate = 0.1, dim = 10, L1 = True):
        self.margin = margin            #TODO:?
        self.learingRate = learingRate  #学习率
        self.dim = dim                  #向量维度
        self.entityList = entityList    #一开始，entityList是entity的list；初始化后，变为字典，key是entity，values是其向量（使用narray）。
        self.relationList = relationList   #理由同上
        self.tripleList = tripleList       #理由同上
        self.loss = 0
        self.L1 = L1

    def initialize(self):
        '''
        初始化向量   就是给实体和关系的向量（列表） 随机填充一个范围的数字，之后通过训练在调整里面的参数。
        '''
        entityVectorList = {}
        relationVectorList = {}

        for entity in self.entityList:
            entityVector = norm(random.uniform(-6/(self.dim**0.5), 6/(self.dim**0.5), self.dim))
            entityVectorList[entity] = entityVector

        print(f"entityVector初始化完成，数量是{len(entityVectorList)}")
        for relation in self. relationList:
            relationVector = norm(norm(random.uniform(-6/(self.dim**0.5), 6/(self.dim**0.5), self.dim)))   #归一化
            relationVectorList[relation] = relationVector  #这是定义字典 类似于dict1['a']=b  {'a':b} 就是一个键值对 为每一个关系定义一个

        print("relationVectorList初始化完成，数量是%d"%len(relationVectorList))
        self.entityList = entityVectorList   #用实体向量链表初始化  字典
        self.relationList = relationVectorList

    def transE(self, cI = 20):
        print("训练开始")
        for cycleIndex in range(cI): #
            self.loss = 0
            Sbatch = sample(self.tripleList, 150)  #150表示的是采样150个三元组 这是个列表 [(),(),(),,,,]  列表里面有元组
            Tbatch = set()     #元组对（原三元组，打碎的三元组）的列表 ：[((h,r,t),(h',r,t'))] Tbatch是个列表哦
            for sbatch in Sbatch: #原三元组  sbatch这是个元组  里面的圆括号
                tripletWithCorruptedTriplet = (sbatch, self.getCorruptedTriplet(sbatch)) #是个元组  把这随机采样的150个三元组打乱  结构是这样的（（没替换的三元组），（替换完后的三元组））
                Tbatch.add(tripletWithCorruptedTriplet)  #这是个列表 把上面的三元组加入到列表里面。tripletWithCorruptedTriplet是包含两个三元组的元组，Tbatch是列表

            self.update(Tbatch)  #这部分是重点
            if cycleIndex % 50 == 0:
                print("第%d次循环, Loss: %.2f"%(cycleIndex, self.loss/150))

    def getCorruptedTriplet(self, triplet):
        '''
        training triplets with either the head or tail replaced by a random entity (but not both at the same time)
        训练三元组，头部或尾部由随机实体替换（但不能同时替换）
        :param triplet:
        :return corruptedTriplet:
        '''
        i = uniform(-1, 1)        #生成随机数
        if i < 0:                         #小于0，打坏三元组的第一项 从序列a中随机抽取n个元素，并将n个元素生以list形式返回。
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]  #sample(序列a，n) 实体链表（是个字典）的值，即"实体向量"
                if entityTemp != triplet[0]:
                    break  #break 用来终止循环语句，如果使用嵌套循环，break将停止执行最深层的循环并开始执行下一行的代码。
            corruptedTriplet = (entityTemp, triplet[1], triplet[2])     #entityTemp 是随机采样的一个实体？  这里是用来替换<头实体>,triplet[1][2]分别是未替换的 尾实体和关系 这个三元组定义的顺序是（head,tail,relation）头-尾-关系
        else:#大于等于0，打坏三元组的第二项 即尾实体
            while True:
                entityTemp = sample(self.entityList.keys(), 1)[0]
                if entityTemp != triplet[1]:
                    break
            corruptedTriplet = (triplet[0], entityTemp, triplet[2])  #entityTemp是用来替换的实体，替换第二项，即尾实体， 头和关系不变
        return corruptedTriplet  #返回打乱的三元组

    def update(self, Tbatch): #[((),()),,,,,,,] 这是个列表
        copyEntityList = deepcopy(self.entityList)   # 深复制 字典  同时复制值及其包含的所有值  浅复制的值会被改变 深复制的值不会变 entityList是字典
        copyRelationList = deepcopy(self.relationList)
        
        for tripletWithCorruptedTriplet in Tbatch:         #[(h,t,r),(h',t',r)] Tbatch是个列表哦   a = [('a','b','c'),('1','2','3')]  a[1][2] = '3' 这是例子
            headEntityVector = copyEntityList[tripletWithCorruptedTriplet[0][0]]#tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple 00表示头实体
            tailEntityVector = copyEntityList[tripletWithCorruptedTriplet[0][1]]  #01表示尾实体 字典是键 对应向量 表示取该实体对应的向量
            relationVector = copyRelationList[tripletWithCorruptedTriplet[0][2]]    #02表示关系  前面三个是正样本的
             #d = {'names':'zyj','school':'swufe'}  d['names']  这就是例子

            headEntityVectorWithCorruptedTriplet = copyEntityList[tripletWithCorruptedTriplet[1][0]]   #负样本的头实体
            tailEntityVectorWithCorruptedTriplet = copyEntityList[tripletWithCorruptedTriplet[1][1]]   #负样本的尾实体


            headEntityVectorBeforeBatch = self.entityList[tripletWithCorruptedTriplet[0][0]]   #tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple
            tailEntityVectorBeforeBatch = self.entityList[tripletWithCorruptedTriplet[0][1]]     #entityList是字典  实体和向量键值对
            relationVectorBeforeBatch = self.relationList[tripletWithCorruptedTriplet[0][2]]

            headEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][0]]    #打乱的头实体在批量之前
            tailEntityVectorWithCorruptedTripletBeforeBatch = self.entityList[tripletWithCorruptedTriplet[1][1]]
            
            if self.L1:
                distTriplet = distanceL1(headEntityVectorBeforeBatch, tailEntityVectorBeforeBatch, relationVectorBeforeBatch)    #顺序是头->尾->关系  这个应该是没有打乱原来的三元组
                distCorruptedTriplet = distanceL1(headEntityVectorWithCorruptedTripletBeforeBatch, tailEntityVectorWithCorruptedTripletBeforeBatch ,  relationVectorBeforeBatch)
            else:   #上面这个应该是打乱后的三元组
                distTriplet = distanceL2(headEntityVectorBeforeBatch, tailEntityVectorBeforeBatch, relationVectorBeforeBatch)
                distCorruptedTriplet = distanceL2(headEntityVectorWithCorruptedTripletBeforeBatch, tailEntityVectorWithCorruptedTripletBeforeBatch ,  relationVectorBeforeBatch)

            eg = self.margin + distTriplet - distCorruptedTriplet   #（最大间隔一般设为1）+（正样本的距离）----（负样本的距离）
            if eg > 0:                 # [function]+ 是一个取正值的函数
                self.loss += eg
                if self.L1:                     #（h+r-t）^2 - (h'+r-t')^2    想让正样本的d-pos 尽可能小，负样本的d-nag 尽可能大  平方里面可以换成 d-pos/d-nag=（t-h-r） t↓h↑r↑ 整个一项就减小
                    tempPositive = tailEntityVectorBeforeBatch - headEntityVectorBeforeBatch - relationVectorBeforeBatch  #这是没有替换的三元组
                    tempCorrupted = tailEntityVectorWithCorruptedTripletBeforeBatch - headEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch #这是替换了的三元组

                    tempPositive=array([1 if tempPositive[i]>0 else -1 for i in range(tempPositive.__len__())])*self.learingRate
                    tempCorrupted = array([1 if tempCorrupted[i] > 0 else -1 for i in range(tempCorrupted.__len__())])*self.learingRate

                else:
                    tempPositive = 2 * self.learingRate * (tailEntityVectorBeforeBatch - headEntityVectorBeforeBatch - relationVectorBeforeBatch)
                    tempCorrupted = 2 * self.learingRate * (tailEntityVectorWithCorruptedTripletBeforeBatch - headEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch)

                headEntityVector = headEntityVector + tempPositive   # t-r-h 使正样本距离小
                tailEntityVector = tailEntityVector - tempPositive
                relationVector = relationVector + tempPositive - tempCorrupted

                headEntityVectorWithCorruptedTriplet = headEntityVectorWithCorruptedTriplet - tempCorrupted   #t'-r-h' 使负样本距离
                tailEntityVectorWithCorruptedTriplet = tailEntityVectorWithCorruptedTriplet + tempCorrupted


               #只归一化这几个刚更新的向量，而不是按原论文那些一口气全更新了
                copyEntityList[tripletWithCorruptedTriplet[0][0]] = norm(headEntityVector)  #把向量归一化
                copyEntityList[tripletWithCorruptedTriplet[0][1]] = norm(tailEntityVector)
                copyRelationList[tripletWithCorruptedTriplet[0][2]] = norm(relationVector)
                copyEntityList[tripletWithCorruptedTriplet[1][0]] = norm(headEntityVectorWithCorruptedTriplet)
                copyEntityList[tripletWithCorruptedTriplet[1][1]] = norm(tailEntityVectorWithCorruptedTriplet)
                # 一个列表 [(h,r,t),(h',r,t)]

        self.entityList = copyEntityList
        self.relationList = copyRelationList
        
    def writeEntilyVector(self, dir):
        print("写入实体")
        entityVectorFile = open(dir, 'w')
        for entity in self.entityList.keys():
            entityVectorFile.write(entity+"\t")  #写入实体
            entityVectorFile.write(str(self.entityList[entity].tolist())) #写入实体对应的向量
            entityVectorFile.write("\n")
        entityVectorFile.close()

    def writeRelationVector(self, dir):
        print("写入关系")
        relationVectorFile = open(dir, 'w')  #写入
        for relation in self.relationList.keys():  #关系字典的 键，即关系的集合
            relationVectorFile.write(relation + "\t")
            relationVectorFile.write(str(self.relationList[relation].tolist()))
            relationVectorFile.write("\n")
        relationVectorFile.close()

def init(dim):
    return uniform(-6/(dim**0.5), 6/(dim**0.5))

def distanceL1(h, t ,r):
    s = h + r - t #都是{ndarray} [,,,,,\n,,,,,\n]这种类型的
    sum = fabs(s).sum()   #向量元素的绝对值，求和
    return sum

def distanceL2(h, t, r):  #注意这个顺序
    s = h + r - t    #注意这个顺序
    sum = (s*s).sum() #没开根号
    return sum
 

def norm(list):
    '''
    归一化
    :param 向量
    :return: 向量的平方和的开方后的向量
    '''
    var = linalg.norm(list)  # linalg=linear（线性）+algebra（代数），norm则表示范数。#是一个确定的数
    i = 0            # 这是求范数的 相当于那个模长 作为分母
    while i < len(list):
        list[i] = list[i]/var    # list[i] 为每一维的向量建立一个列表，分子是向量，分母是母长 list是一百个向量组成的列表
        i += 1
    return array(list)

def openDetailsAndId(dir,sp="\t"):
    idNum = 0
    list = []   #每经过一次循环 list[]列表里面增加一个实体
    with open(dir) as file:
        lines = file.readlines()  #lines ->列表
        for line in lines:  #line是单个的实体。
            DetailsAndId = line.strip().split(sp)   # DetailsAndId = ['/m/06rf7','0'] 例子
            list.append(DetailsAndId[0]) #这一项表示去实体或关系的列表
            idNum += 1  #标量 用来标记有多少个实体或关系
    return idNum, list

def openTrain(dir,sp="\t"):
    num = 0 #计数
    list = []  #[(),(),,,,]  里面是很多三元组
    with open(dir) as file:
        lines = file.readlines()  #读取整个文件，line: '/m/027rn\t/m/06cx9\t/location/country/form_of_government\n'line就是一个字符串 lines是一个列表 很多line组成的
        for line in lines:    #triple 是列表
            triple = line.strip().split(sp) # 方法用于移除字符串头尾指定的字符（默认为空格或换行符，也可以填其他的）或字符序列。 再通过制表符隔开成三部分
                                 #注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。 triple = ['/m/027rn','/m/06cx9','/location/country/form_of_government']
            if(len(triple)<3):  #这是为了验证三元组的完整性，可能有缺项的，就直接跳过
                continue #如果元组缺了一项，就跳到下一行
            list.append(tuple(triple)) #list是列表 tuple将列表/序列\字典(返回的是键)转换为元组。 []->() triple 是list  ，listappend是列表  意思是把元组作为列表的元素加入
            num += 1
    return num, list

if __name__ == '__main__':
    dirEntity = "../data/FB15k/entity2id.txt"
    entityIdNum, entityList = openDetailsAndId(dirEntity)           #entityIdNum=14951，entityList是个列表，放的是实体的集合
    dirRelation = "../data/FB15k/relation2id.txt"
    relationIdNum, relationList = openDetailsAndId(dirRelation)   #relationIdNum = 1345；relationList是个列表，放的是关系的集合。
    dirTrain = "../data/FB15k/train.txt"
    tripleNum, tripleList = openTrain(dirTrain)
    print("打开TransE")
    transE = TransE(entityList,relationList,tripleList, margin=1, dim = 100, L1=False)
    print("TranE初始化")
    transE.initialize()                #随机填充向量值
    transE.transE(15000)               #把元组对加入到【（），（），，】
    transE.writeRelationVector("../output/relationVector.txt")
    transE.writeEntilyVector("../output/entityVector.txt")

