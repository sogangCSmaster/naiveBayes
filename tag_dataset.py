import os
import errno
import glob
import random
import sys
from functools import reduce

def load_data(foldername):
    print("Data loading Start!")
    columns = ['section', 'class', 'subclass', 'abstract']
    selected = ['section', 'abstract']

    file_list = []
    for path, dirs, files in os.walk(foldername):
        if files:
            for filename in files:
                fullname = os.path.join(path, filename)
                file_list.append(fullname)
    
    data = []
    temp = list()
    for filename in file_list:
        with open(filename, 'r') as fp:
            for line in fp:
                temp.append(line)
    print("Data loading finished")
    return temp


def load_neg_word(negativewords):
    fp = open(negativewords, 'r')
    wordArray = fp.read()
    wordArray = wordArray.split('\n')
    wordArray.pop()
    return wordArray

def rearrange():
    dataset = './dataset'
    negativewords = 'neg.txt'
    negwords = load_neg_word(negativewords)
    datas = load_data(dataset)

    i = 0
    countPos = 0
    countNeg = 0
    for data in datas:
        if True in map(lambda x: x in data, negwords):
            fp = open('./redataset/neg/' + str(i) + '.txt','w')
            fp.write(data)
            sys.stdout.write("\rData Proccessed : %d" %(i))
            sys.stdout.flush()
            fp.close()
            i = i + 1
            countNeg = countNeg + 1
        else:             
            fp = open('./redataset/pos/' + str(i) + '.txt','w')
            fp.write(data)
            sys.stdout.write("\rData Proccessed : %d" %(i))
            sys.stdout.flush()
            fp.close()
            i = i +1
            countPos = countPos + 1
    print("\nPositive : " + str(countPos))
    print("Negative : " + str(countNeg))
if __name__=='__main__':
    rearrange()
    print()
