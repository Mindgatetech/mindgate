from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def my_model():
    clf = LDA()
    return ('LDA', clf)