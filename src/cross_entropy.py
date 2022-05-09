from math import log2
def cross_entropy_single(p, q):
    return -sum([p[i]*log2(q[i]) for i in range(len(p))])


def cross_entropy(y_test,y_pred):
    result = 0
    for i in range(len(y_pred)):
        result += cross_entropy_single(1, y_pred[i][y_test[i]])

    result = result / len(y_pred)

    return result
