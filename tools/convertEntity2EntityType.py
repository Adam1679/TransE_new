#!/usr/bin/env python
# encoding: utf-8
# # Task
# 1. 通过把实体替换为实体类型的方式，构造实体类型三元组如：<left entity type, relation, right entity type>。（数量约为：N*M^2个）
# 2. 构造的数据中有大量重复的数据，所以需统计数据的数据的频次

#%%
from collections import Counter
from collections import defaultdict
from copy import deepcopy
import pandas as pd
import os
import json
trainTriplePath = "../data/freebase_mtr100_mte100-train.txt"
trainTypePath = "../data/FB15k_Entity_Type_train.txt"
savingTypeRelationTypePath = "../output/type-relation-type.txt"
savingTypeRelationTypeUniquePath = "../output/type-relation-type-unique-with-frequency.txt"

#%%

triplesEntityRelationEntity = []
with open(trainTriplePath, "r") as f:
    for line in f.readlines():
        triplesEntityRelationEntity.append(tuple(line.strip().split("\t")))
print(f"In total, there is {len(triplesEntityRelationEntity)} samples")


#%%


entityEntityTypeDict = defaultdict(list)
with open(trainTypePath, "r") as f:
    for line in f.readlines():
        entityId, entityType = line.strip().split("\t")
        entityEntityTypeDict[entityId].append(entityType)
totalTypes = sum(map(lambda x: len(x), entityEntityTypeDict.values()))
totalEntity = len(entityEntityTypeDict.keys())
print(f"In total, there is {totalEntity} entity and {totalTypes} types. \nOn average, every entigy has {totalTypes/totalEntity: .2f} types")


#%%
# entityRelationEntityDf = pd.DataFrame(triplesEntityRelationEntity, columns=['leftEntity', "Relation", 'rightEntity'])
# entityEntityTypeDf = pd.DataFrame(dict(entityEntityTypeDict, columns=['Entity', "EntityType"])

# %%


triplesTypeRelationType = []
entityEntityTypeDict1 = deepcopy(entityEntityTypeDict)
entityEntityTypeDict2 = deepcopy(entityEntityTypeDict)
count = 0
for spo in triplesEntityRelationEntity:
    leftEntity, relation, rightEntity = spo
    while(entityEntityTypeDict1[leftEntity]):
        leftType = entityEntityTypeDict1[leftEntity].pop()
        tmp = deepcopy(entityEntityTypeDict2[rightEntity])
        while(tmp):
            rightType = tmp.pop()
            triplesTypeRelationType.append(tuple([leftType, relation, rightType]))
            count += 1
            
print(f"In total, there is {len(triplesTypeRelationType)} type2type samples")

with open(savingTypeRelationTypePath, 'w') as f:
    for line in triplesTypeRelationType:
        f.write("\t".join(line)+"\n")
#%% Generate Unique <leftEntity Relation rightEntity> file and count there frequencies

count = dict(Counter(triplesTypeRelationType))
with open(savingTypeRelationTypeUniquePath, "w") as f:
    for key, freq in count.items():
        s1 = "\t".join(key)+"\t"+str(freq)+"\n"
        f.write(s1)

print(f"In total, there is {len(count)} unique samples")







