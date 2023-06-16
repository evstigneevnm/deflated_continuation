import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
import sys

y0, t0 = [2, 0], 0

def vdp(t, y, arg1):
#    return [arg1*y[0]]
    return [y[1],arg1*(1-y[0]**2)*y[1]-y[0]]


def read_res_file(f_name):
    data_t = []
    data_y0 = []
    data_y1 = []
    with open(f_name) as f:
        while True:
            line = f.readline()
            if not line:
                break
            values = line.split()
            data_t.append(float(values[0]))
            data_y0.append(float(values[1]))
            data_y1.append(float(values[2]))
    return (data_t, data_y0, data_y1)




def main(mu, t1, rtol, atol, f_name, do_print):
    
    file_data = read_res_file(f_name)

    method = 'dop853'
    print("Using: ", method, " for pythod ODE integrator with rtol = ", rtol, " and atol = ", atol)
    r = ode(vdp).set_integrator(method, rtol = rtol, atol = atol) #'dop853'
    r.set_initial_value(y0, t0).set_f_params(mu)
    
    all_t = []
    all_y = []
    dt = file_data[0][0]
    cnt = 0
    while r.successful() and r.t < t1:
        r.integrate(r.t+dt)
        all_t.append(r.t)
        all_y.append(r.y)
        cnt = cnt + 1
        dt = file_data[0][cnt] - file_data[0][cnt-1]
    
    
    if(do_print):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        axes[0].plot(all_t, np.transpose(np.transpose(all_y)[1]),'.' )
        axes[0].plot(file_data[0],file_data[2],'.')
        axes[1].plot(all_t, np.transpose(np.transpose(all_y)[0]),'.' )
        axes[1].plot(file_data[0],file_data[1],'.')
        axes[0].legend(['python native','cpp'])
        axes[1].legend(['python native','cpp'])
        plt.show()
    
    y_res_0 = np.transpose(np.transpose(all_y)[0])
    y_res_1 = np.transpose(np.transpose(all_y)[1])
    error_pointwise_0 = np.array(y_res_0-np.array(file_data[1][0:len(y_res_0) ]))
    error_pointwise_1 = np.array(y_res_1-np.array(file_data[2][0:len(y_res_1) ]))
    error_L_2 =  np.sqrt(np.sum(error_pointwise_0**2 + error_pointwise_1**2))/np.sqrt(t1) 
    error_max_0 = (np.abs(error_pointwise_0)).max()
    error_max_1 = (np.abs(error_pointwise_1)).max()
    error_L_inf = max(error_max_0, error_max_1)
    print( "L_2 error = ", error_L_2)
    print( "L_inf error = ",  error_L_inf)
    if (error_L_2 > 10.0) or (error_L_inf > 10.0):
        return 2
    elif (error_L_2 > 1.0e-3) or (error_L_inf > 1.0e-3):
        return 1
    else:
        return 0


if __name__ == '__main__':
    rtol = 1.0e-10
    atol = 1.0e-13
    res = 3 #dafault fail status for incorrect options
    if (len(sys.argv) == 4) or (len(sys.argv) == 5):
        mu = float(sys.argv[1])
        t = float(sys.argv[2])
        f_name = sys.argv[3]
        do_print = False
        if(len(sys.argv) == 5):
            do_print = bool(sys.argv[4])

        res = main(mu, t, rtol, atol, f_name, do_print)
    sys.exit(res) #for ci
