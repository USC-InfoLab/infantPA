import json
from utils import preprocess_all, preprocess_age_group, preprocess_vgroup

if __name__ == "__main__":
  with open('config.json', 'r') as config_file:
    config = json.load(config_file)
  n_motifs = config['n_motifs']
  task = config['task']
  subtask = config['subtask']
  storage_dir = config['storage_dir']
  motifs_dir = config['motifs_dir']
  sub_files = config['age_group']
  motif_len = config['motif_len']
  
  if subtask == "all":
    preprocess_all(task, storage_dir, motifs_dir, sub_files, motif_len, n_motifs)
  elif subtask == "age_group":
    preprocess_age_group(task, storage_dir, motifs_dir, sub_files, motif_len, n_motifs)
  elif subtask == "visit_group":
    preprocess_vgroup(task, storage_dir, motifs_dir, sub_files, motif_len, n_motifs)
  else:
    print("Error, wrong task")
    exit()
  