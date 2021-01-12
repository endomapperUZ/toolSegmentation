import glob
import random

def get_split(root, fold, train_files, val_files):
    
  train_file_names = []
  val_file_names = []
  for file in train_files:
    train_path = glob.glob(root + "/data/" + fold.replace('raw','masks') + "/"+file+"*.png") 
    train_path.sort()
    train_file_names.append(train_path)

  for file in val_files:
    val_path = glob.glob(root + "/data/" + fold.replace('raw','masks') + "/"+file+"*.png") 
    val_path.sort()
    val_file_names.append(val_path)

  random.shuffle(train_file_names)
  random.shuffle(val_file_names)
  print(len(train_file_names))
  print(len(val_file_names))
  #file_name = "train_val_dataset"+fold.replace('train_data/','').replace('_raw','')+".txt"
  #f = open(file_name, "a")
  #f.write("Train files : \n")
  #f.write("\n".join(train_file_names))
  #f.write("\n Val files : \n")
  #f.write("\n".join(val_file_names))
  #f.close()
  return train_file_names, val_file_names
