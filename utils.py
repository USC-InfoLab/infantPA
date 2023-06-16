import pickle
from knn import train_test_split, knn

AR_good = {'AR_03', 'AR_04', 'AR_05', 'AR_06', 'AR_08', 'AR_09', 'AR_10', 'AR_13', 'AR_24', 'AR_26'}
AR_poor = {'AR_02', 'AR_11', 'AR_12', 'AR_14', 'AR_16', 'AR_17', 'AR_19', 'AR_22', 'AR_25'}

def extract_motifs(L, R, LI, RI, m, y, n_motifs):
  Mi = []
  yi = []
  for k, i in enumerate(LI):
    j = i[0]
    motif = L[j : j+m]
    Mi.append(motif)
    yi.append(y)
    if k == n_motifs-1:
      break
  for k,i in enumerate(RI):
    j = i[0]
    motif = R[j : j+m]
    Mi.append(motif)
    yi.append(y)
    if k == n_motifs-1:
      break
  return (Mi, yi)

def parse_files(task, data, motifs, labels, n_motifs):
  params = data['params']
  mp = data['result']
  m = params['window']
  ##
  for key in mp:
    if task == 'ARgood vs ARpoor' and key[:2] == 'TD':
        print("Skipping.. ", key)
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
    M, y = extract_motifs(L, R, LI, RI, m, yi, n_motifs)
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
    
def preprocess_all(task, storage_dir, motifs_dir, sub_files, motif_len, n_motifs):
  labels = []
  motifs = []
  for sfile in sub_files:
    with open(storage_dir + motifs_dir + sfile + str(motif_len) + '.pickle', 'rb') as file:
      # print(file)
      data = pickle.load(file)
      parse_files(task, data, motifs, labels, n_motifs)
  X_train, X_test, y_train, y_test = train_test_split(motifs, labels)
  # print(labels)
  res = knn(X_train, X_test, y_train, y_test)
  print(res)

def preprocess_age_group(task, storage_dir, motifs_dir, sfile, motif_len, n_motifs):
  labels = []
  motifs = []
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
      