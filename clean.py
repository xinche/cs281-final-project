import os
import shutil
for f in [ f for f in os.listdir(".") if os.path.isdir(os.path.join(".", f))]:
    current_folder = os.path.join(".",f)
    files = os.listdir(os.path.join(".",f))
    len_list = [len(os.listdir(os.path.join(os.path.join(current_folder, fil), os.listdir(os.path.join(current_folder,fil))[0]))) for fil in files]
    if len(len_list) != 1:
        shutil.rmtree(os.path.join(os.path.join(current_folder, files[len_list.index(min(len_list))])))
