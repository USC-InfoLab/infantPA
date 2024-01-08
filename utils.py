import pickle
from knn import train_test_split, knn
from stats import stats
import numpy as np
from scipy.stats import skew, kurtosis, zscore

AR_good = {'AR_03', 'AR_04', 'AR_05', 'AR_06', 'AR_08', 'AR_09', 'AR_10', 'AR_13', 'AR_24', 'AR_26'}
AR_poor = {'AR_02', 'AR_11', 'AR_12', 'AR_14', 'AR_16', 'AR_17', 'AR_19', 'AR_22', 'AR_25'}

def zscore_(data):
  # mean = np.mean(data)
  # std = np.std(data)

  # # Calculate Z-scores
  # z_scores = (data - mean) / std
  # min_z_score = np.min(z_scores)
  # positive_z_scores = z_scores - min_z_score
  # return positive_z_scores
  return data

def extract_motifs(L, R, LI, RI, m, y, n_motifs):
  Mi = []
  yi = []
  D = []
  for k, i in enumerate(LI):
    j = i[0]
    motif = L[j : j+m]
    Mi.append(zscore_(motif))
    yi.append(y)
    D.append(L)
    if k == n_motifs-1:
      break
  for k,i in enumerate(RI):
    j = i[0]
    motif = R[j : j+m]
    Mi.append(zscore_(motif))
    yi.append(y)
    D.append(R)
    if k == n_motifs-1:
      break
  return (Mi, yi, D)

def parse_files(task, data, motifs, labels, n_motifs, dataset=[]):
  params = data['params']
  mp = data['result']
  m = params['window']
  ##
  for key in mp:
    if task == 'ARgood vs ARpoor' and key[:2] == 'TD':
        # print("Skipping.. ", key)
        continue
    # ith time series for left and right leg
    L = mp[key]['Left']
    R = mp[key]['Right']
    # motif indexes
    LI = mp[key]['Left_idx']
    RI = mp[key]['Right_idx']
    # T label
    yi = mp[key]['y']
    if task == 'ARgood vs ARpoor':
      if key[:5] in AR_good:
        yi = 0
      elif key[:5] in AR_poor:
        yi = 1
      else:
        continue
    M, y, D = extract_motifs(L, R, LI, RI, m, yi, n_motifs)
    if type(motifs) == dict and type(labels) == dict:
      if key[:5] not in motifs:
        motifs[key[:5]] = M
        labels[key[:5]] = y
      else:
        motifs[key[:5]] += M
        labels[key[:5]] += y
    else:
      motifs.append(M)
      labels.append(y)
      dataset.append(D)
    
def preprocess_all(task, storage_dir, motifs_dir, sub_files, motif_len, n_motifs):
  labels = []
  motifs = []
  dataset = []
  for sfile in sub_files:
    with open(storage_dir + motifs_dir + sfile + str(motif_len) + '.pickle', 'rb') as file:
      # print(file)
      data = pickle.load(file)
      parse_files(task, data, motifs, labels, n_motifs, dataset)
  stats(dataset, motifs, labels)
  X_train, X_test, y_train, y_test = train_test_split(motifs, labels)
  # print(labels)
  res = knn(X_train, X_test, y_train, y_test)
  print(res)

def preprocess_age_group(task, storage_dir, motifs_dir, sub_files, motif_len, n_motifs):
  labels = []
  motifs = []
  for sfile in sub_files:
    print(sfile)
    with open(storage_dir + motifs_dir + sfile + str(motif_len) + '.pickle', 'rb') as file:
      # print(file)
      data = pickle.load(file)
      parse_files(task, data, motifs, labels, n_motifs)
    X_train, X_test, y_train, y_test = train_test_split(motifs, labels)
    res = knn(X_train, X_test, y_train, y_test)
    print(res)

def preprocess_vgroup(task, storage_dir, motifs_dir, sub_files, motif_len, n_motifs):
  labels = {}
  motifs = {}
  for sfile in sub_files:
    with open(storage_dir + motifs_dir + sfile + str(motif_len) + '.pickle', 'rb') as file:
      # print(file)
      data = pickle.load(file)
      parse_files(task, data, motifs, labels, n_motifs)
  X_train, X_test, y_train, y_test = train_test_split(motifs, labels)
  res = knn(X_train, X_test, y_train, y_test)
  print(res)

