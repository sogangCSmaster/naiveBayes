from textblob.classifiers import NaiveBayesClassifier
import glob
from konlpy.tag import Twitter
import re
import pickle
#train = [('text', 'pos'), ('text2', 'neg')]
#classifier = nltk.NaiveBayesClassifier.train(train)



def preprocess(text):
    target_list = ["\t", "…", "·", "●", "○", "◎", "△", "▲", "◇", "■", "□", "☎", "☏", "※", "▶", "▷", "ℓ", "→", "↓", "↑", "┌", "┬", "┐", "├", "┤", "┼", "─", "│", "└", "┴", "┘"]

    for target in target_list:
        text = text.replace(target, " ")
        
    gija_str1 = r"[^ ]*[ ]?기자[ ]?[a-zA-Z0-9]*@[a-zA-Z0-9]*\.co[^ ]*"
    gija_str2 = r"[^ ]*[ ]?기자[ ]?[a-zA-Z0-9]*@"
    part1 = re.compile(gija_str1)
    part2 = re.compile(gija_str2)

    text = re.sub(part1, "", text)
    text = re.sub(part2, "", text)
    strip_text = text.strip()
    return strip_text

def training():

    train = []
    twitter = Twitter()
    listofPos = glob.glob('./redataset/pos/*.txt')
    listofNeg = glob.glob('./redataset/neg/*.txt')
    temp = tuple()
    for filename in listofNeg:
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                line = preprocess(line)
                iine = twitter.pos(line, norm=True, stem=True)
                line = ' '.join(map(lambda x: x[0], line))
                temp = list(temp)
                temp.append(line)
                temp.append('pos')
                temp = tuple(temp)
                train.append(temp)
        fp.close()

    temp2 = tuple()
    for filename in listofPos:
        with open(filename, 'r', encoding='utf-8') as fp2:
            for line in fp2:
                line = preprocess(line)
                iine = twitter.pos(line, norm=True, stem=True)
                line = ' '.join(map(lambda x: x[0], line))
                temp2 = list(temp2)
                temp2.append(line)
                temp2.append('pos')
                temp2 = tuple(temp2)
                train.append(temp2)
        fp2.close()


    classifier = NaiveBayesClassifier(train)
    f = open('./model/NaiveBayes.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()

if __name__=='__main__':
    print("training start!")
    training()
    print("training done")
