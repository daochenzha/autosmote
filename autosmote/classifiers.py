from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

name2f = {
    "knn": KNeighborsClassifier(),
    "svm": LinearSVC(random_state=0),
    "dt": DecisionTreeClassifier(),
    "adaboost": AdaBoostClassifier(random_state=0),
}

def get_clf(name):
    return name2f[name]

