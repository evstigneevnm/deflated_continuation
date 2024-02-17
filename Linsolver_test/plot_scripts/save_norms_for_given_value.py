import sys
import numpy as np
from matplotlib import pyplot as plt
import os
plt.rcParams.update({'font.size': 14})



def extract_norms(list_of_folders, file_name, parameter):
    all_lines = []

    for folder in list_of_folders:
        fsplit = folder.split("/")
        f_number = fsplit[-1:][0]
        with open(folder + "/" + file_name) as file:
            for line in file:
                val = float(line.split(' ')[0])
                if np.abs(val - parameter)<1.0e-5:
                    all_lines.append((f_number, line.replace("\n", "")))

    return(all_lines)



def main():
    if len(sys.argv) != 3:
        print("usage: ", sys.argv[0], " parameter_value root_folder")
        return 1
    

    file_name = "debug_curve_all.dat"
    parameter = float(sys.argv[1])
    rootdir = sys.argv[2]
    list_to_check = []
    for root, subdirs, files in os.walk(rootdir):
        # print("root = ", root, "subdirs = ", subdirs, "files = ", files)
        if file_name in files:
            list_to_check.append(root)

    data_all = extract_norms(list_to_check, file_name, parameter)
    print(data_all)
    save_file_name = "saved_norms_for_" + sys.argv[1] + ".dat"
    with open(save_file_name,"w") as file:
        for data in data_all:   
            new_str = data[0] + " " + data[1] + "\n"
            file.write(new_str)


if __name__ == '__main__':
    main()