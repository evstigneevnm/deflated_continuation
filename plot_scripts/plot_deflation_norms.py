import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})


def main():
    argc = len(sys.argv)
    if(argc != 2):
        print("usage: ", sys.argv[0], " deflation_file_name.csv")
        return 0;
    file_name = sys.argv[1]
    pddata = pd.read_csv(file_name, header = None)
    nvols = pddata.shape[1]
    p = 0;
    q = 0;
    t = 0;
    limit_p = 10
    limit_q = 7
    for j in range(0, nvols):
        if np.isnan((pddata[pddata.columns[j]]) ).any():
            p = p + 1
        if not np.isnan((pddata[pddata.columns[j]]) ).any():
            q = q + 1
        if p<limit_p and np.isnan((pddata[pddata.columns[j]]) ).any():
            plt.semilogy( np.multiply(pddata[pddata.columns[j]], 1.0), '-.', label = t)
            t = t + 1
        if q<limit_q and not np.isnan((pddata[pddata.columns[j]]) ).any():
            plt.semilogy( np.multiply(pddata[pddata.columns[j]], 1.0), '-.', label = t)
            t = t + 1


    # plt.legend(title="Newton attempt", ncol=2, framealpha=0.4)
    print("number of attempts = ", nvols)
    plt.xlabel("Iteration")
    plt.ylabel("$||F||_{2}$")
    plt.grid()
    plt.tight_layout()
    plt.savefig("plot_deflation_norms.pdf")
    plt.show()


if __name__ == '__main__':
    main()