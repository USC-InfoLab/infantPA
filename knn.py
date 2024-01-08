from itertools import chain
from pyts.classification import KNeighborsClassifier

def train_test_split(motifs, labels):
  X_train = []
  y_train = []
  X_test = []
  y_test = []
  if type(motifs) == list and type(labels) == list:
    N = len(motifs)
    for i in range(N):
      X_train.append(list(chain.from_iterable(motifs[:i]+motifs[i+1:])))
      y_train.append(list(chain.from_iterable(labels[:i]+labels[i+1:])))
      X_test.append(motifs[i])
      y_test.append(labels[i])
  else:
    for key, y in labels.items():
      motif_array = motifs[key]
      del motifs[key]
      X_test.append(motif_array)
      y_test.append(y)
      xtemp = []
      ltemp = []
      for k, motif in motifs.items():
        xtemp.extend(motif)
        ltemp.extend(labels[k])
      y_train.append(ltemp)
      X_train.append(xtemp)
      motifs[key] = motif_array
  return (X_train, X_test, y_train, y_test)

def knn(X_train, X_test, y_train, y_test):
  clf = KNeighborsClassifier()
  TP = 0
  TN = 0
  FP = 0
  FN = 0
  N = len(X_train)
  for i in range(N):
    clf.fit(X_train[i], y_train[i])
    res = clf.predict(X_test[i])
    if res.mean() >= 0.5:
      y_pred = 1
    else:
      y_pred = 0
    y_true = y_test[i][0]
    if y_true == 0:
      if y_pred == 0:
        TN += 1
      else:
        FP += 1
    else:
      if y_pred == 0:
        FN += 1
      else:
        TP += 1
  return (TP, TN, FP, FN)
