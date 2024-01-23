from sklearn.metrics import make_scorer, accuracy_score

def my_metric():
    metric = {
        'accuracy' : 'accuracy',
        'my_metric' : make_scorer(accuracy_score)
    }
    return metric