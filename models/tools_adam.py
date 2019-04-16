from collections import OrderedDict

def loadEntityId(path="data/entity2id.txt"):
    d = OrderedDict()
    with open(path, 'r') as f:
        for line in f.readlines():
            e, id = line.strip().split("\t")
            d[e] = int(id)
    return d

def loadRelationId(path="data/relation2id.txt"):
    d = OrderedDict()
    with open(path, 'r') as f:
        for line in f.readlines():
            r, id = line.strip().split("\t")
            d[r] = int(id)
    return d

def loadTriplet(path="data/freebase_mtr100_mte100-train.txt"):
    triplet = []
    with open(path, 'r') as f:
        for line in f.readlines():
            h, l, t = line.strip().split("\t")
            triplet.append((h,l,t)) 
    return triplet



if __name__ == "__main__":
    trainEntitySet = set()
    trainRelationSet = set()
    trainDataPath = "data/freebase_mtr100_mte100-train.txt"
    with open(trainDataPath, "r") as f:
        for line in f.readlines():
            e1, r, e2 = line.strip().split("\t")
            trainEntitySet.add(e1)
            trainEntitySet.add(e2)
            trainRelationSet.add(r)

    with open("data/entity2id.txt", 'w') as f:
        count = 0
        for e in trainEntitySet:
            f.write("\t".join([e, str(count)]))
            f.write("\n")
            count += 1

    with open("data/relation2id.txt", 'w') as f:
        count = 0
        for e in trainRelationSet:
            f.write("\t".join([e, str(count)]))
            f.write("\n")
            count += 1


