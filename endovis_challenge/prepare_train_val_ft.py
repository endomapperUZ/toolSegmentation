import glob
import random

def get_split(root, fold, train_size ):
    
  train_path = glob.glob(root + "/data/" + fold.replace('raw','masks') + "/*.png") 
  train_file_names = []
  val_file_names = []
  #train_path.sort()
  num_files = len(train_path)
  print(num_files)
  file_ind = [k for k in range(num_files)]
  random.shuffle(file_ind)
  for i in range(int(num_files*train_size)):
    train_file_names.append(train_path[file_ind[i]])
  for m in range(int(num_files*train_size), num_files):
    val_file_names.append(train_path[file_ind[m]])
  #train_file_names = train_path[:int(num_files*train_size)]
  #val_file_names = train_path[int(num_files*train_size):]
 # print(val_file_names)
  random.shuffle(train_file_names)
  random.shuffle(val_file_names)
  #print(val_file_names) 
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
