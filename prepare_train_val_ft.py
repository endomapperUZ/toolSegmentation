import glob

def get_split(root, fold, train_size ):
    
  train_path = glob.glob(root + "/data/Training/OP"+ str(fold) + "/Raw/*.png") 

  train_file_names = []
  val_file_names = []
  num_files = len(train_path)
  train_file_names = train_path[:int(num_files*train_size)]
  val_file_names = train_path[int(num_files*train_size):]
  return train_file_names, val_file_names
