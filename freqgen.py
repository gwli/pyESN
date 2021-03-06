import numpy as np
from matplotlib import pyplot as plt
from pyESN import ESN
import copy
from pso import pso
import inspect
import lorenz_ode

import os
pid = os.getpid()
#debug = raw_input("if debug please attach to PID:{}, then Press any key to debug".format(pid))
# please QT_API=pyside before running

rng = np.random.RandomState(42)
def frequency_generator(N,min_period,max_period,n_changepoints):
    """returns a random step function with N changepoints
       and a sine wave signal that changes its frequency at
       each such step, in the limits given by min_ and max_period."""
    # vector of random indices < N, padded with 0 and N at the ends:
    changepoints = np.insert(np.sort(rng.randint(0,N,n_changepoints)),[0,n_changepoints],[0,N])
    # list of interval boundaries between which the control sequence should be constant:
    const_intervals = list(zip(changepoints,np.roll(changepoints,-1)))[:-1]
    # populate a control sequence
    frequency_control = np.zeros((N,1))
    for (t0,t1) in const_intervals:
        frequency_control[t0:t1] = rng.rand()
    periods = frequency_control * (max_period - min_period) + max_period
    # run time through a sine, while changing the period length
    frequency_output = np.zeros((N,1))
    z = 0
    for i in range(N):
        z = z + 2 * np.pi / periods[i]
        frequency_output[i] = (np.sin(z) + 1)/2
    return np.hstack([np.ones((N,1)),1-frequency_control]),frequency_output


#N = 8000 # signal length
N = 10000 # signal length
min_period = 2
max_period = 10
n_changepoints = int(N/200)
#frequency_control,frequency_output = frequency_generator(N,min_period,max_period,n_changepoints)


#lorenz
t,x,y,z = lorenz_ode.lorenz_ode_compute(N)

frequency_control = np.array(zip(x,y))
frequency_output = np.array([[i] for i in z] )
##regular
frequency_control = (frequency_control- frequency_control.min())/(frequency_control.max()-frequency_control.min())
frequency_output = (frequency_output- frequency_output.min())/(frequency_output.max()-frequency_output.min())


# from lozren_data
lozren_data = np.genfromtxt('LorenzData.csv',delimiter=',')
frequency_control = lozren_data[:,0:-1]
frequency_output = lozren_data[:,-1].reshape([4000,1])
print frequency_output.shape
#frequency_output =frequency_output.reshape([1,4000]) 
#######################################################################

traintest_cutoff = int(np.ceil(0.7*N))

train_ctrl,train_output = frequency_control[:traintest_cutoff],frequency_output[:traintest_cutoff]
test_ctrl, test_output  = frequency_control[traintest_cutoff:],frequency_output[traintest_cutoff:]


n_reservoir = 800
spectral_radius = 0.9
sparsity = 0.05
noise = 0.001
#####################
esn = ESN(n_inputs = 8,
          n_outputs = 1,
          n_reservoir = n_reservoir,
          spectral_radius = spectral_radius, 
          sparsity = sparsity,
          noise = noise,
          input_shift = [0,0,0,0,0,0,0,0],
          input_scaling = [0.01,0.01,0.01,0.01,0.01,0.01,3,3],
          teacher_scaling = 1.12,
          teacher_shift = -0.7,
          out_activation = np.tanh,
          inverse_out_activation = np.arctanh,
          random_state = rng,
          silent = False)


def pso_esn_parameters_for_scad(x):
    # 0: tao, 1:c0, 2:IC_s,3:IC_e 4:IS_s,5:IS_e, ,6:teacher sacling,7:teacher shift
    # 0:IC_s,1:IC_e, 2:IS_s,3:IS_e, 4:teacher sacling,5:teacher shift,6: tao, 7:c0,
    ic_s =x[0]
    ic_e =x[1]
    is_s = x[2]
    is_e = x[3]
    teacher_scaling = x[4]
    teacher_shift = x[5]
    tao = x[6]
    c0 =  x[7]
     
    esn = ESN(n_inputs = 2,
             n_outputs = 1,
             n_reservoir = n_reservoir,
             spectral_radius = spectral_radius, 
             sparsity = sparsity,
             noise = noise,
             input_shift = [is_s,is_e],#[0,0]
             input_scaling =[ic_s,ic_e],# [0.01, 3]
             teacher_scaling = teacher_scaling,#1.12,
             teacher_shift = teacher_shift,#-0.7,
             out_activation = np.tanh,
             inverse_out_activation = np.arctanh,
             random_state = rng,
             silent = False)
    esn.penal_tao =tao
    esn.penal_c0 = c0
    internal_states,transient = esn.train_reservior(train_ctrl,train_output)
    pred_train = esn.train_readout_with_scad(internal_states,train_output,transient)
    pred_test = esn.predict(test_ctrl)
    #test_error_rate= np.sqrt(np.mean((pred_test - test_output)**2))
    test_error_rate= np.sqrt((pred_test - test_output)**2)
    #get function name as title
    title = inspect.stack()[0][3]
    print "#### {} ## train_error:{},test_error:{}".format(title,esn.train_error_rate,test_error_rate)
    return test_error_rate
    这里画一个原始测试序列的图和一个test_output的图。
def opt_pso_scad():
    lb = [0,0,0.01,3,1.12,-2,0,3.7]
    ub = [1,1,0.3, 10,  2,-0.7,1,4]
   # 这个代表什么？
    xopt1, fopt1 = pso(pso_esn_parameters_for_scad, lb, ub,debug=True)
    print('The optimum is at:')
    print('    {}'.format(xopt1))
    print('Optimal function value:')
    print('    myfunc: {}'.format(fopt1))

def pso_esn_parameters_for_l2scad(x):
    # 0: tao, 1:c0, 2:IC_s,3:IC_e 4:IS_s,5:IS_e, ,6:teacher sacling,7:teacher shift
    # 0:IC_s,1:IC_e, 2:IS_s,3:IS_e, 4:teacher sacling,5:teacher shift,6: tao, 7:c0,
    ic_s =x[0]
    ic_e =x[1]
    is_s = x[2]
    is_e = x[3]
    teacher_scaling = x[4]
    teacher_shift = x[5]
    tao = x[6]
    c0 =  x[7]
     
    esn = ESN(n_inputs = 2,
             n_outputs = 1,
             n_reservoir = n_reservoir,
             spectral_radius = spectral_radius, 
             sparsity = sparsity,
             noise = noise,
             input_shift = [is_s,is_e],#[0,0]
             input_scaling =[ic_s,ic_e],# [0.01, 3]
             teacher_scaling = teacher_scaling,#1.12,
             teacher_shift = teacher_shift,#-0.7,
             out_activation = np.tanh,
             inverse_out_activation = np.arctanh,
             random_state = rng,
             silent = False)
    esn.penal_tao =tao
    esn.penal_c0 = c0
    internal_states,transient = esn.train_reservior(train_ctrl,train_output)
    pred_train = esn.train_readout_with_l2scad(internal_states,train_output,transient)
    pred_test = esn.predict(test_ctrl)
    test_error_rate= np.sqrt(np.mean((pred_test - test_output)**2))
    #get function name as title
    title = inspect.stack()[0][3]
    print "#### {} ## train_error:{},test_error:{}".format(title,esn.train_error_rate,test_error_rate)
    return test_error_rate
 
def opt_pso_l2scad():
    lb = [0,0,0.01,3,1.12,-2,0,3.5]
    ub = [1,1,0.3, 10,  2,-0.7,0.03,3.8]
    xopt1, fopt1 = pso(pso_esn_parameters_for_l2scad, lb, ub,debug=True)

    print('The optimum is at:')
    print('    {}'.format(xopt1))
    print('Optimal function value:')
    print('    myfunc: {}'.format(fopt1))
def pso_esn_parameters_for_lasso(x):
    ic_s =x[0]
    ic_e =x[1]
    is_s = x[2]
    is_e = x[3]
    teacher_scaling = x[4]
    teacher_shift = x[5]
    alpha =x[6]

     
    esn = ESN(n_inputs = 2,
             n_outputs = 1,
             n_reservoir = n_reservoir,
             spectral_radius = spectral_radius, 
             sparsity = sparsity,
             noise = noise,
             input_shift = [is_s,is_e],#[0,0]
             input_scaling =[ic_s,ic_e],# [0.01, 3]
             teacher_scaling = teacher_scaling,#1.12,
             teacher_shift = teacher_shift,#-0.7,
             out_activation = np.tanh,
             inverse_out_activation = np.arctanh,
             random_state = rng,
             silent = False)
    esn.alpha = alpha
    internal_states,transient = esn.train_reservior(train_ctrl,train_output)
    pred_train = esn.train_readout_with_lasso(internal_states,train_output,transient)
    pred_test = esn.predict(test_ctrl)
    test_error_rate= np.sqrt(np.mean((pred_test - test_output)**2))
    #get function name as title
    title = inspect.stack()[0][3]
    print "#### {} ## train_error:{},test_error:{}".format(title,esn.train_error_rate,test_error_rate)
    return test_error_rate
def opt_pso_lasso():
    lb = [0,0,0.01,3,1.12,-2,0]
    ub = [1,1,0.3,10,2, -0.7,1]
    xopt1, fopt1 = pso(pso_esn_parameters_for_lasso, lb, ub,debug=True)

    print('The optimum is at:')
    print('    {}'.format(xopt1))
    print('Optimal function value:')
    print('    myfunc: {}'.format(fopt1))

def pso_esn_parameters_for_ridge(x):
    ic_s =x[0]
    ic_e =x[1]
    is_s = x[2]
    is_e = x[3]
    teacher_scaling = x[4]
    teacher_shift = x[5]
    alpha =x[6]

     
    esn = ESN(n_inputs = 2,
             n_outputs = 1,
             n_reservoir = n_reservoir,
             spectral_radius = spectral_radius, 
             sparsity = sparsity,
             noise = noise,
             input_shift = [is_s,is_e],#[0,0]
             input_scaling =[ic_s,ic_e],# [0.01, 3]
             teacher_scaling = teacher_scaling,#1.12,
             teacher_shift = teacher_shift,#-0.7,
             out_activation = np.tanh,
             inverse_out_activation = np.arctanh,
             random_state = rng,
             silent = False)
    esn.alpha = alpha
    internal_states,transient = esn.train_reservior(train_ctrl,train_output)
    pred_train = esn.train_readout_with_ridge(internal_states,train_output,transient)
    pred_test = esn.predict(test_ctrl)
    test_error_rate= np.sqrt(np.mean((pred_test - test_output)**2))
    #get function name as title
    title = inspect.stack()[0][3]
    print "#### {} ## train_error:{},test_error:{}".format(title,esn.train_error_rate,test_error_rate)
    return test_error_rate
def opt_pso_ridge():
    lb = [0,0,0.01,3,1.12,-2,0]
    ub = [1,1,0.3,10,2, -0.7,1]
    xopt1, fopt1 = pso(pso_esn_parameters_for_ridge, lb, ub,debug=True)

    print('The optimum is at:')
    print('    {}'.format(xopt1))
    print('Optimal function value:')
    print('    myfunc: {}'.format(fopt1))
             
def pso_esn_parameters_for_elasticnet(x):
    ic_s =x[0]
    ic_e =x[1]
    is_s = x[2]
    is_e = x[3]
    teacher_scaling = x[4]
    teacher_shift = x[5]
    alpha =x[6]
    l1_ratio=x[7]

     
    esn = ESN(n_inputs = 2,
             n_outputs = 1,
             n_reservoir = n_reservoir,
             spectral_radius = spectral_radius, 
             sparsity = sparsity,
             noise = noise,
             input_shift = [is_s,is_e],#[0,0]
             input_scaling =[ic_s,ic_e],# [0.01, 3]
             teacher_scaling = teacher_scaling,#1.12,
             teacher_shift = teacher_shift,#-0.7,
             out_activation = np.tanh,
             inverse_out_activation = np.arctanh,
             random_state = rng,
             silent = False)
    esn.alpha = alpha
    esn.l1_ratio = l1_ratio
    internal_states,transient = esn.train_reservior(train_ctrl,train_output)
    pred_train = esn.train_readout_with_elasticnet(internal_states,train_output,transient)
    pred_test = esn.predict(test_ctrl)
    test_error_rate= np.sqrt(np.mean((pred_test - test_output)**2))
    #get function name as title
    title = inspect.stack()[0][3]
    print "#### {} ## train_error:{},test_error:{}".format(title,esn.train_error_rate,test_error_rate)
    return test_error_rate

def opt_pso_elasticnet():
    lb = [0,0,0.01,3,1.12,-2,0,0]
    ub = [1,1,0.3,10,2, -0.7,1,1]
    xopt1, fopt1 = pso(pso_esn_parameters_for_elasticnet, lb, ub,debug=True)

    print('The optimum is at:')
    print('    {}'.format(xopt1))
    print('Optimal function value:')
    print('    myfunc: {}'.format(fopt1))

def plot_result(title,esn,pred_train):
    print("test error:")


def test_error(title,esn,pred_train):

    print("test error:")
    pred_test = esn.predict(test_ctrl)
    print(np.sqrt(np.mean((pred_test - test_output)**2)))
    #esn.dump_parameters()
    #return
    
    window_tr = range(int(len(train_output)/4),int(len(train_output)/4+2000))
    plt.figure(figsize=(10,1.5))
    plt.plot(train_ctrl[window_tr,1],label='control')
    plt.plot(train_output[window_tr],label='target')
    plt.plot(pred_train[window_tr],label='model')
    plt.legend(fontsize='x-small')
    plt.title('training (excerpt)')
    plt.ylim([-0.1,1.1])

    window_test = range(test_output.shape[0])
    #window_test = range(4000)
    plt.figure(figsize=(10,1.5))
    plt.plot(test_ctrl[window_test,1],label='control')
    plt.plot(test_output[window_test],label='target')
    plt.plot(pred_test[window_test],label='model')
    plt.legend(fontsize='x-small')
    plt.title('test (excerpt)')
    plt.ylim([-0.1,1.1]);

    def draw_spectogram(data):
        plt.specgram(data,Fs=4,NFFT=256,noverlap=150,cmap=plt.cm.bone,detrend=lambda x:(x-0.5))
        plt.gca().autoscale('x')
        plt.ylim([0,0.5])
        plt.ylabel("freq")
        plt.yticks([])
        plt.xlabel("time")
        plt.xticks([])

    plt.figure(figsize=(7,1.5))
    draw_spectogram(train_output.flatten())
    plt.title("training: target")
    plt.figure(figsize=(7,1.5))
    draw_spectogram(pred_train.flatten())
    plt.title("training: model")

    plt.figure(figsize=(3,1.5))
    draw_spectogram(test_output.flatten())
    plt.title("test: target")
    plt.figure(figsize=(3,1.5))
    draw_spectogram(pred_test.flatten())
    plt.title("test: model")
    plt.show()


def compair_readout():
    esn = ESN(n_inputs = 8,
             n_outputs = 1,
             n_reservoir = n_reservoir,
             spectral_radius = spectral_radius, 
             sparsity = sparsity,
             noise = noise,
             input_shift = [0.51293657,0.51293657,0.51293657,0.51293657,0.03489584,0.03489584,0.03489584,0.03489584],
             input_scaling = [0.18636639,0.18636639,0.18636639,0.18636639, 0.11791364, 0.11791364, 0.11791364, 0.11791364],
             teacher_scaling = 1.45377531,
             teacher_shift = -0.7997228,
             out_activation = np.tanh,
             inverse_out_activation = np.arctanh,
             random_state = rng,
             silent = False)
             
    esn = ESN(n_inputs = 8,
          n_outputs = 1,
          n_reservoir = n_reservoir,
          spectral_radius = spectral_radius, 
          sparsity = sparsity,
          noise = noise,
          input_shift = [0,0,0,0,0,0,0,0],
          input_scaling = [0.01,0.01,0.01,0.01,0.01,0.01,3,3],
          teacher_scaling = 1.12,
          teacher_shift = -0.7,
          out_activation = np.tanh,
          inverse_out_activation = np.arctanh,
          random_state = rng,
          silent = False)
    esn.penal_tao =0.57201544
    esn.penal_c0 = 3.76108161

    esn.penal_tao = 0.1
    esn.penal_c0 = 3.7
    #pred_train = esn.fit(train_ctrl,train_output,inspect=True)
    internal_states,transient = esn.train_reservior(train_ctrl,train_output)
#    esn_Lasso = copy.deepcopy(esn)
#    esn_Ridge = copy.deepcopy(esn)
#    esn_ElasticNet = copy.deepcopy(esn)
    esn_SCAD = copy.deepcopy(esn)
    esn.dump_parameters()
#    print "####pin"
#    pred_train = esn.train_readout_with_pin(internal_states,train_output,transient)
#    test_error("pinv",esn,pred_train)
#    exit()
#
#    print "####ridge"
#    pred_train = esn_Ridge.train_readout_with_ridge(internal_states,train_output,transient)
#    test_error("pinv",esn_Ridge,pred_train)
#
#    print "####Lasso"
#    pred_train = esn_Lasso.train_readout_with_lasso(internal_states,train_output,transient)
#    test_error("pinv",esn_Lasso,pred_train)
#
#    print "####ElasticNet"
#    pred_train = esn_ElasticNet.train_readout_with_elasticnet(internal_states,train_output,transient)
#    test_error("pinv",esn_ElasticNet,pred_train)

    print "####SCAD"
    pred_train = esn_SCAD.train_readout_with_scad(internal_states,train_output,transient)
    test_error("pinv",esn_SCAD,pred_train)

if __name__ == "__main__":
    #opt_pso_ridge()
    #opt_pso_lasso()
    #opt_pso_elasticnet()
    #opt_pso_scad()
    #opt_pso_l2scad()
    compair_readout()
