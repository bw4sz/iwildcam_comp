#evaluation scripts
from sklearn.metrics import f1_score

#Fscore
def fscore(groundtruth, prediction):
    score = f1_score(groundtruth, prediction, average='macro')  
    return score
