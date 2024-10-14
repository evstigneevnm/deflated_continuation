import sys
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 28})





def main():
    argc = len(sys.argv)
    if(argc != 3) and (argc != 4):
        print("usage: ", sys.argv[0], " output_file_name.txt number_of_exponents shift(optional)")
        return 0
    file_name = sys.argv[1]
    N = int(sys.argv[2])
    shift = 0
    if(argc == 4):
        shift = np.float64(sys.argv[3])

    all_exponents = []
    with open(file_name) as f:
        for line in f:
            if "obtained norms and exponents for total time" in line:
                single_5 = []
                for j in range(0,N):
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
    
    
    plt.rcParams["figure.figsize"] = (8,6)

    for data in all_exponents_t:
        plt.scatter(range(0,len(data)), data+shift)

    plt.tight_layout()
    #    plt.legend(["$\lambda_1$","$\lambda_2$","$\lambda_3$","$\lambda_4$","$\lambda_5$"])
    plt.grid()
    plt.xlabel("iteration")
    plt.ylabel("Lyapunov exponent")

    plt.show()


if __name__ == '__main__':
    main()
