from math import log
from scipy import io
from numpy  import *
import operator
import sys
import arff


class TreeNode:
    def __init__(self):
        self.sons = {}
        self.label = ''
        self.classification = ''
        self.bestMidPoint = 0
        self.numberEachClass = {}
        return
    
    def isLeaf(self):
        return self.label == ''

    def getSizeOfEachClass(self):
        ans = " ["
        self.c = 0
        for key in labels[-1][1]:
            if self.c == 0:
                self.c = 2 
                ans += str(self.numberEachClass[key])
            else:
                ans += " "+str(self.numberEachClass[key])
        ans = ans + "] "
        return ans


def isNominal(label):
    return label[1] == 'REAL' or label[1]=='NUMERIC'

def isLabelNominal(label):
    for l in labels:
        if l[0] == label:
            return l[1] == 'REAL' or l[1] == 'NUMERIC'
    return False

def getMidPointList(dataSet, featureAxis, labels):
    feature_class = [[data[featureAxis],data[-1]] for data in dataSet]
    feature_class.sort(key=operator.itemgetter(0,1))
    midPointList = []
    for i in range(1,len(feature_class)):
        if feature_class[i][1] != feature_class[i-1][1]:
            if feature_class[i][0] == feature_class[i-1][0]:
                currentFeatureVal = feature_class[i][0]
                while True:
                    if i < 0:
                        break;
                    i -= 1
                    if feature_class[i][0] != currentFeatureVal:
                        midPoint = float(feature_class[i][0] + currentFeatureVal) / 2.0
                        midPointList.append(midPoint)
                        break;
                while True:
                    if i == len(feature_class) - 1:
                        break
                    i += 1
                    if feature_class[i][0] != currentFeatureVal:
                        midPoint = float(feature_class[i][0] + currentFeatureVal) / 2.0
                        midPointList.append(midPoint)
                        break
            else:
                midPoint = float(feature_class[i][0] + feature_class[i-1][0]) / 2.0
                midPointList.append(midPoint)
    return midPointList


def getClassification(classificationList):
    if (len(classificationList) == 0):
        return labels[-1][1][0]
    classes = {}
    for vote in classificationList:
        if vote in classes.keys():
            classes[vote] += 1
        else:
            classes[vote] = 1

    def func(x,y):
        axis = label2Axis['class']
        featureVals = labels[axis][1]
        px = len(featureVals)
        py = len(featureVals)
        for i in range(len(featureVals)):
            if x == featureVals[i]:
                px = i
            if y == featureVals[i]:
                py = i
        return px - py
    sortedClass = sorted(classes.iteritems(), cmp=func, key=operator.itemgetter(0))
    sortedClass = sorted(sortedClass, key=operator.itemgetter(1), reverse=True)
    return sortedClass[0][0]

def calcEntropy(dataSet):
    labels = {}
    dataNum = shape(dataSet)[0]
    
    for i in range(dataNum):
        data = dataSet[i]
        label = data[-1]
        if label in labels.keys():
            labels[label] += 1
        else:
            labels[label] = 1
    
    entropy = 0.0
    for i in labels:
        prob = float(labels[i]) / dataNum
        entropy -= prob * log2(prob)
        
    return entropy

def splitDataSet(dataSet, axis, feature):
    subDataSet = []
    for data in dataSet:
        if data[axis] == feature:
            neoData = data[:axis]
            neoData.extend(data[axis+1:])
            subDataSet.append(neoData)
    return subDataSet

def splitNominalDataSet(dataSet, axis, midpoint):
    oriSubDataSet1 = [data for data in dataSet if data[axis] <= midpoint]
    oriSubDataSet2 = [data for data in dataSet if data[axis] > midpoint]

    subData1 = []
    for data in oriSubDataSet1:
        temp = data[:]
        subData1.append(temp)

    subData2 = []
    for data in oriSubDataSet2:
        temp = data[:]
        subData2.append(temp)
    return subData1, subData2


    
def findBestSplitFeature(dataSet, labels):
    dataNum = shape(dataSet)[0]
    featureNum = shape(dataSet)[1] - 1
    
    baseEntropy = calcEntropy(dataSet)
    bestFeatureAxis = 0.0
    bestInfoGain = 0.0
    bestMidPoint = 0
    
    for featureAxis in range(featureNum):
        if isNominal(labels[featureAxis]):
            midPointList = getMidPointList(dataSet, featureAxis,labels)
            for midPoint in midPointList:
                subDataSet1, subDataSet2 = splitNominalDataSet(dataSet, featureAxis, midPoint)
                prob1 = 1.0 * len(subDataSet1) / float(dataNum)
                prob2 = 1.0 * len(subDataSet2) / float(dataNum)
                neoEntropy = prob1 * calcEntropy(subDataSet1) + prob2 * calcEntropy(subDataSet2)
                if baseEntropy - neoEntropy > bestInfoGain:
                    bestFeatureAxis = featureAxis
                    bestInfoGain = baseEntropy - neoEntropy
                    bestMidPoint = midPoint
        else:
            featureList = [data[featureAxis] for data in dataSet]
            featureVal = set(featureList)
            neoEntropy = 0.0
            for feature in featureVal:
                subDataSet = splitDataSet(dataSet, featureAxis, feature)
                prob = 1.0 * len(subDataSet) / dataNum
                neoEntropy += prob * calcEntropy(subDataSet)
            if baseEntropy - neoEntropy > bestInfoGain:
                bestInfoGain = baseEntropy - neoEntropy
                bestFeatureAxis = featureAxis
                
    return bestFeatureAxis, bestMidPoint
    
def getNumberOfEachClassification(node, classificationList):
    classificationDict = {}
    for vote in classes:
        classificationDict[vote] = 0
    for vote in classificationList:
        classificationDict[vote] += 1
    node.numberEachClass = classificationDict
    
    
def createTree(dataSet, labels, m):
    node = TreeNode()

    classificationList = [data[-1] for data in dataSet]
    
    
    # get the number of instance for each classification, just for output
    getNumberOfEachClassification(node, classificationList)
    
    # only one classification
    classificationSet = set(classificationList)
    if len(classificationSet) == 1:
        node.classification = dataSet[0][-1]
        return node
    
    # fewer than m training instances
    instanceNum = shape(dataSet)[0]
    if instanceNum < m:
        node.classification = getClassification(classificationList)
        return node
    
    # no information has positive information gain ?
    
    # no more training candidate splits at the node
    # Is this the same as no more attributes?
    featureNum = shape(dataSet)[1] - 1
    if featureNum == 0:
        node.classification = getClassification(classificationList)
        return node
        
    bestFeatureAxis, bestMidPoint = findBestSplitFeature(dataSet,labels)
    node.label = labels[bestFeatureAxis][0]
    node.bestMidPoint = bestMidPoint

    if isNominal(labels[bestFeatureAxis]) :
        subDataSet1, subDataSet2 = splitNominalDataSet(dataSet, bestFeatureAxis, bestMidPoint)
        feature1 = " <= " + str(bestMidPoint)
        feature2 = " > " + str(bestMidPoint)
        node.sons[feature1] = createTree(subDataSet1, labels[:], m)
        node.sons[feature2] = createTree(subDataSet2, labels[:], m)
    else:
        featureSet = set([data[bestFeatureAxis] for data in dataSet])
        featureSet = set(labels[bestFeatureAxis][1])
        fullBestFeatureSet = labels[bestFeatureAxis][1]
        
        subLabel = labels[:bestFeatureAxis]
        subLabel.extend(labels[bestFeatureAxis+1:])
        
        for feature in featureSet:
            subDataSet = splitDataSet(dataSet, bestFeatureAxis, feature)
            node.sons[feature] = createTree(subDataSet, subLabel, m)

    return node
    
def classify(data,root):
    classification = ''
    while True:
        if root.isLeaf():
            classification = root.classification
            break
        else:
            label = root.label
            featureAxis = label2Axis[label]
            featureValue = data[featureAxis]
            if isLabelNominal(label):
                if featureValue <= root.bestMidPoint:
                    root = root.sons[" <= " + str(root.bestMidPoint)]
                else:
                    root = root.sons[" > " + str(root.bestMidPoint)]
            else:
                root = root.sons[featureValue]
    return classification


def dfs(root, string):
    label = root.label
    
    def comparator(x,y):
        if isLabelNominal(label):
            return cmp(x,y)
        else:
            axis = label2Axis[label]
            featureVals = labels[axis][1]
            px = len(featureVals)
            py = len(featureVals)
            for i in range(len(featureVals)):
                if x == featureVals[i]:
                    px = i
                if y == featureVals[i]:
                    py = i
            return px - py

    keyList = sorted(root.sons.iteritems(),cmp=comparator,key=lambda d:d[0])
    for keyItem in keyList:
        key = keyItem[0]
        if root.sons[key].isLeaf():
            if isLabelNominal(label):
                print string + str(label) + str(key) + \
                      root.sons[key].getSizeOfEachClass() + \
                      " : " + root.sons[key].classification
            else:
                print string + str(label) +' = '+ str(key) + \
                      root.sons[key].getSizeOfEachClass() + \
                      " : " + root.sons[key].classification
        else:
            if isLabelNominal(label):
                print string + str(label)+str(key)+ root.sons[key].getSizeOfEachClass()
            else:
                print string + str(label)+' = '+str(key)+root.sons[key].getSizeOfEachClass()                
            dfs(root.sons[key], string+'|\t')
    return



#train_file = sys.argv[1]
#test_file = sys.argv[2]
#m = int(sys.argv[3])

train_file = 'diabetes_train.arff'
test_file = 'diabetes_test.arff'
m = 20


originalDataSet = arff.load(open(train_file, 'rb'))
dataSet = originalDataSet['data']
classes = set([data[-1] for data in dataSet])
labels = originalDataSet['attributes']
label2Axis = {}
for i in range(len(labels)):
    label2Axis[labels[i][0]] = i


root = createTree(dataSet,labels,m)

dfs(root,"")



print "<Predictions for the Test Set Instances>"
originalTestDataSet = arff.load(open(test_file,'rb'))
dataSet = originalTestDataSet['data']
count = 0
for i in range(len(dataSet)):
    data = dataSet[i]
    realClassification = data[-1]
    resClassification = classify(data,root)
    print "%3d: Actual: %s  Predicted: %s" %((i+1),realClassification, resClassification)
    if realClassification == resClassification:
        count += 1
accuracy = 1.0 * count / len(dataSet)
print "Number of correctly classified: %d  Total number of test instances: %d" \
      %(count,len(dataSet))
