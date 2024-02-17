import sys
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})





def main():
    argc = len(sys.argv)
    if(argc != 2) and (argc != 3):
        print("usage: ", sys.argv[0], " output_file_name.txt shift(optional)")
        return 0;
    file_name = sys.argv[1]
    shift = 0
    if(argc == 3):
        shift = np.float64(sys.argv[2])

    all_exponents = []
    with open(file_name) as f:
        for line in f:
            if "obtained norms and exponents for total time" in line:
                single_5 = []
                for j in range(0,5):
                    line = f.readline()
                    string_val = line.split('\n')[0].split(' ')[-1]
                    single_5.append(np.float64(string_val) )
                all_exponents.append(single_5)
    
    if len(all_exponents) == 0:
        with open(file_name) as f:
            for line in f:
                data_l = line.split(' ')
                count = 0;
                single_5 = []
                for value in data_l:
                    if count%2 == 1:
                        # print(value)
                        single_5.append( np.float64(value) )
                    count = count + 1
                all_exponents.append(single_5)

    all_exponents_t = np.transpose(all_exponents)
    
    for data in all_exponents_t:
        plt.scatter(range(0,len(data)), data+shift)

    plt.tight_layout()
    plt.legend(["$\lambda_1$","$\lambda_2$","$\lambda_3$","$\lambda_4$","$\lambda_5$"])
    plt.grid()
    plt.xlabel("iteration")
    plt.ylabel("Lyapunov exponent")

    plt.show()


if __name__ == '__main__':
    main()