from cProfile import label
import math
from time import time
from traceback import print_tb
import uuid
import autograd.numpy as np
import scipy.optimize as spo
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits import mplot3d
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import unflatten_optimizer
from autograd.misc.optimizers import adam as adam_one_param
from autograd.builtins import SequenceBox
from autograd.misc import flatten
from memory_profiler import profile #@profile
import random
import functools
import math

####### Misc
def calltracker(func):
    @functools.wraps(func)
    def wrapper(*args):
        wrapper.has_been_called = True
        return func(*args)
    wrapper.has_been_called = False
    return wrapper

####### Data
class Sin_Reg():
    def __init__(self, amp_range = (0.8,1.2), ph_range = (math.pi*-0.2,math.pi*0.2), sd = 0.1, sample_rng = (-5,5)):
        self.amp_range = amp_range
        self.ph_range = ph_range
        self.sd = sd
        self.sample_rng = sample_rng

    def sin_fun_gen(self):
        """
        Returns sin function with added noise from zero mean gaussian dist. with given standard deviation. 
        The sin function has randomly generated parameters for amplitude
        and phase within given ranges
        """
        a_1, a_2 = self.amp_range
        p_1, p_2 = self.ph_range
        a = npr.uniform(a_1,a_2) #Samples parameter for amplitude
        b = npr.uniform(p_1,p_2) #Samples parameter for phase
        def f(x):
            return (np.sin(x+b)*a) + np.random.normal(scale=self.sd, size=len(x))
        return f, str(uuid.uuid4()), {"Amplitude":a,"Phase":b} #Returns function of sin curve with random gaussian noise     

    def get_sin_points(self, sin_fun, num_points):
        x_min , x_max = self.sample_rng
        xs = npr.uniform(x_min, x_max, num_points)
        ys = sin_fun(xs)
        xs = np.transpose(np.array([xs]))
        ys = np.transpose(np.array([ys]))
        return [xs, ys]

class Sin_Time():
    def __init__(self, amp=1.0, t_0=0, time_around_t_0=math.pi*0.1, sd = 0.1, sample_rng = (-5,5)):
        self.amp = amp
        self.t_0 = t_0
        self.time_around_t_0 = time_around_t_0
        self.sd = sd
        self.sample_rng = sample_rng

        def uni_time_dist():
            return npr.uniform(t_0 - self.time_around_t_0, t_0 + self.time_around_t_0)
        self.time_dist = uni_time_dist

        def calc_phase_eq(time):
            return time
        self.get_phase = calc_phase_eq

    def sin_fun_gen(self):
        """
        Returns sin function with added noise from zero mean gaussian dist. with given standard deviation. 
        The sin function has set phase (class attribute) and is given a random time (phase)
        """
        time = self.time_dist() #Samples parameter for phase
        phase = self.get_phase(time)
        def f(x):
            return (np.sin(x+phase)*self.amp) + np.random.normal(scale=self.sd, size=len(x))
        return f, "time:"+str(time) , {"Amplitude":self.amp,"Phase":phase} #Returns function of sin curve with random gaussian noise    

    def get_sin_points(self, sin_fun, num_points):
        x_min , x_max = self.sample_rng
        xs = npr.uniform(x_min, x_max, num_points)
        ys = sin_fun(xs)
        xs = np.transpose(np.array([xs]))
        ys = np.transpose(np.array([ys]))
        return [xs, ys] 

class Clas_Time():
    def __init__(self, t_0=0, time_around_t_0=0.5, sd = 0.5, mean_0=0, mean_1=1):
        self.t_0 = t_0
        self.time_around_t_0 = time_around_t_0
        self.sd = sd
        self.mean_0 = mean_0
        self.mean_1 = mean_1

    def clas_fun_gen(self):
        """
        Returns a tuple of functions for generating points from two  normal distributions offset from t_0 
        by a time picked uniformly from [t_0 - time_around_t_0, t_0 + time_around_t_0],
        along with a task-id and dictionary of information on the task
        """
        time = npr.uniform(self.t_0 - self.time_around_t_0, self.t_0 + self.time_around_t_0)
        mean_zero = self.mean_0 + time
        mean_one = self.mean_1 + time
        def f(num_points):
            return np.random.normal(loc=mean_zero,scale=self.sd, size=num_points)
        def g(num_points):
            return np.random.normal(loc=mean_one,scale=self.sd, size=num_points)
        return (f,g), "time:"+str(time) , {"Mean of zero":mean_zero,"Mean of one":mean_one,"Sd":self.sd}   

    def get_clas_points(self, class_funs, num_points):
        f_zero, f_one = class_funs

        xs = np.append(f_zero(math.floor(num_points/2)), f_one(math.ceil(num_points/2)))
        ys = np.append(np.zeros(math.floor(num_points/2)), np.ones(math.ceil(num_points/2)))

        perm = np.random.permutation(num_points)
        xs = xs[perm]
        ys = ys[perm]

        xs = np.transpose(np.array([xs]))
        ys = np.transpose(np.array([ys]))

        return [xs, ys] 

class Data_Generator():
    def __init__(self, task_gen, datapoint_retreiver):
        self.task_gen = task_gen # This function should generate a random task and return it along with a unique ID pertaining to the task and useful task info (could be None)
        self.dp_ret = datapoint_retreiver # This function when given a task and a number n, should get n data-points for given task
                                          # data-points returned as a list, containing a list of inputs to the network and a second list of targets for the output
    
    def k_shot(self, k):
        task, task_id, task_info = self.task_gen()
        points = self.dp_ret(task, k)
        return (task_id, [np.array([points[0]]), np.array([points[1]])] ), task, task_info

    def task_data(self, num_batches_fortask, num_points_pbatch):
        task, task_id, task_info = self.task_gen()
        batches_for_task_xs = []
        batches_for_task_ys = []
        for _ in range(num_batches_fortask):
            one_batch = self.dp_ret(task, num_points_pbatch)
            batches_for_task_xs.append(one_batch[0])
            batches_for_task_ys.append(one_batch[1])
        batches_for_task = (task_id, [np.array(batches_for_task_xs), np.array(batches_for_task_ys)])
        return batches_for_task, task, task_info

    def multiple_tasks_data(self, num_batches_oftasks, num_tasks_pbatch, num_batches_ptask, num_points_pbatch):
        batches_oftasks = []
        all_tasks = []
        all_tasks_info = []
        for _ in range(num_batches_oftasks):
            ls = [ self.task_data(num_batches_ptask, num_points_pbatch) for _ in range(num_tasks_pbatch) ] 
            data = [item[0] for item in ls]
            tasks = [item[1] for item in ls]
            tasks_info = [item[2] for item in ls]

            batches_oftasks.append(data)
            all_tasks.append(tasks)
            all_tasks_info.append(tasks_info)
        return batches_oftasks, all_tasks, all_tasks_info

    def multiple_tasks_test_data(self, num_tasks_pbatch, num_batches_ptask, num_points_pbatch):

        ls = [ self.task_data(num_batches_ptask, num_points_pbatch) for _ in range(num_tasks_pbatch) ] 
        batch_oftasks = [[item[0] for item in ls]]
        tasks = [[item[1] for item in ls]]
        tasks_info = [[item[2] for item in ls]]

        return batch_oftasks, tasks, tasks_info

    def task_test_data(self, num_points_pbatch):
        task, task_id, task_info = self.task_gen()

        batch = self.dp_ret(task, num_points_pbatch)
        xs = batch[0]
        ys = batch[1]
        batch_for_task = (task_id, [np.array(xs), np.array(ys)])

        return batch_for_task, task, task_info    

####### Activation functions
def ReLu(input, comparator=0.0):
    return np.maximum(input,comparator) #ReLu activation function

def sigmoid(input):
    return 1 / (1 + np.exp(-input)) #Sigmoid activation function

def hyperbolic(input):
    return np.tanh(input) #Hyperbolic activation function

####### Loss
def least_squares(output, d_out):
    """
    Returns least squares error of the network for given output of network and desired output
    """
    return np.sum( (d_out - output)**2 ) #Returns the Least squares error between given ys and ys_dash output from network

def neg_log_likelihood(output, d_out):
    """ 
    Computes negative log likelihood
    """
    label_probabilities = output * d_out + (1 - output) * (1 - d_out)
    return -np.sum(np.log(label_probabilities))

def forward_pass(input, w_b, act_fun):
    """
    Performs forward pass of network, given network parameters and input
    """
    output = False
    for t in w_b:
        weights, biases = t
        input = np.matmul(input,weights) + biases #Sum over the inputs multiplied by their respective weights + bias
        input = act_fun(input) #Assigns output of neurons using activation function assigned to class
        output = input
    return output

class Loss():
    def __init__(self, task, act_fun, loss_fun, num_batches_ptask, track_loss=False, test_data_gen = False):
        self.task = task #Inits the current task data
        self.act_fun = act_fun #Inits the activation function of the class
        self.num_batches = num_batches_ptask #Inits the number of batches we are using for the data
        self.loss_fun = loss_fun #Inits the loss function to be used

        self.track_loss = track_loss
        self.loss_tracker = []
        if test_data_gen != False: 
            self.test_data_gen = test_data_gen
            self.test_tracker = []
            self.test_fun = False
        else: 
            self.test_data_gen = test_data_gen

        """ We initialise some attributes to keep track of whats going on during use of this class for learning"""
        self.pass_num = "Not assigned yet"
        self.batch_num = "Not assigned yet"

    def update_atrributes(self, task):
        self.task = task #Updates the current task data

    def get_indices(self, iter):
        batch_num = iter%self.num_batches
        epoch_num = int(iter/self.num_batches)

        self.epoch_num = epoch_num
        self.batch_num = batch_num
        return epoch_num, batch_num

    @calltracker
    def objective(self, w_b, iter):
        """ 
        This is function called during learning,
        It keeps track (by updating attributes) of current data being used,
        It also more importantly returns the error for learning on the data 
        given current weights and biases and data iteration number
        """
        _, batch_num = self.get_indices(iter)
        out = self.forward_pass(self.task[1][0][batch_num], w_b)
        l = self.loss_fun(out, self.task[1][1][batch_num])
        if self.track_loss : self.loss_tracker.append(l._value)
        return l
    
    @calltracker
    def objective_kshot(self, w_b, _):
        out = self.forward_pass(self.task[0][0], w_b)
        return self.loss_fun(out, self.task[0][1])

    def print_perf(self, *args, **kargs):
        """
        Prints information to keep us in the know about the learning process
        """
        if self.test_data_gen != False:
            if self.test_fun == False:
                if self.objective.has_been_called and (not self.objective_kshot.has_been_called):
                    l = Loss(self.test_data_gen(), self.act_fun, self.loss_fun, self.num_batches)
                    def test(args):
                        l.task = self.test_data_gen()
                        w_b, iter = args[0:2]
                        return l.objective(w_b, 0)
                    self.test_fun = test
                    self.test_tracker.append(self.test_fun(args))
                elif self.objective_kshot.has_been_called:
                    l = Loss(self.test_data_gen(), self.act_fun, self.loss_fun, self.num_batches)
                    def test(args):
                        l.task = self.test_data_gen()
                        w_b = args[0]
                        return l.objective_kshot(w_b)
                    self.test_fun = test
                    self.test_tracker.append(self.test_fun(args))
            else:
                self.test_tracker.append(self.test_fun(args))

        print("Epoch: ",self.epoch_num,", Batch: ",self.batch_num) #,", Point: ",self.point_num)

    def forward_pass(self, input, w_b):
        """
        Performs forward pass of network, given network parameters and input
        """
        output = False
        for t in w_b:
            weights, biases = t
            input = np.matmul(input,weights) + biases #Sum over the inputs multiplied by their respective weights + bias
            input = self.act_fun(input) #Assigns output of neurons using activation function assigned to class
            output = input
        return output

class Meta_Loss():
    def __init__(self, batched_tasks_data, act_fun, opt_fun, loss_fun, num_batches_ptask, num_of_tasks_pbatch, num_batches_oftasks,
                inner_param, track_loss=False, test_data_gen = False, inner_callback=None):
        
        self.tau_tracker = [] #TODO: remove

        self.batched_tasks_data = batched_tasks_data
        self.loss_fun = loss_fun
        self.act_fun = act_fun
        self.opt_fun = opt_fun
        self.inner_param = inner_param

        self.num_batches_oftasks = num_batches_oftasks
        self.num_of_tasks_pbatch = num_of_tasks_pbatch
        self.num_batches_ptask = num_batches_ptask

        self.losses, self.grad_losses = self.init_losses(act_fun, loss_fun, num_batches_ptask)

        self.track_loss = track_loss
        self.loss_tracker = []
        if test_data_gen != False: 
            self.test_data_gen = test_data_gen
            self.test_tracker = []
            self.test_fun = False
        else: 
            self.test_data_gen = test_data_gen
        self.inner_callback = inner_callback

    def init_losses(self, act_fun, loss_fun, num_batches_ptask):
        losses = []
        grad_losses = []
        for i in range(self.num_of_tasks_pbatch):
            loss = Loss(None, act_fun, loss_fun, num_batches_ptask)
            losses.append(loss)
            grad_losses.append(grad(loss.objective))
        return losses, grad_losses

    def update_losses(self, batch_num):
        ids = []
        for i in range(self.num_of_tasks_pbatch):
            id, _ = self.batched_tasks_data[batch_num][i]
            self.losses[i].update_atrributes(self.batched_tasks_data[batch_num][i]) 
            ids.append(id)
        return ids

    def get_indices(self, iter):
        batch_num = iter%self.num_batches_oftasks
        epoch_num = int(iter/self.num_batches_oftasks)

        self.epoch_num = epoch_num
        self.batch_num = batch_num
        return epoch_num, batch_num

    @calltracker
    def objective_std(self, w_b, iter):
        _, batch_num = self.get_indices(iter)
        self.update_losses(batch_num)
        sum = 0
        for index in range(len(self.losses)):
            theta_i = self.opt_fun(self.grad_losses[index], w_b, step_size=self.inner_param, num_iters=self.num_batches_ptask, callback=self.inner_callback)
            out = self.losses[index].forward_pass(self.losses[index].task[1][0], theta_i)
            calc_losses = self.loss_fun(out, self.losses[index].task[1][1])
            sum = sum + calc_losses
        if self.track_loss : self.loss_tracker.append(sum._value)
        return sum    

    @calltracker
    def objective_with_alpha(self, w_b, alphas, iter): 
        _, batch_num = self.get_indices(iter)
        alpha_ids = self.update_losses(batch_num)
        sum = 0
        for index in range(len(self.losses)):
            theta_i = self.opt_fun(self.grad_losses[index], w_b, step_size=alphas.get(alpha_ids[index]), num_iters=self.num_batches_ptask, callback=self.inner_callback)
            out = self.losses[index].forward_pass(self.losses[index].task[1][0], theta_i)
            calc_losses = self.loss_fun(out, self.losses[index].task[1][1])
            sum = sum + calc_losses
        if self.track_loss : self.loss_tracker.append(sum._value)
        return sum  

    @calltracker
    def objective_with_time(self, w_b, tau, iter):
        if isinstance(tau, float):
            self.tau_tracker.append(tau) #TODO: remove
        elif isinstance(tau, np.ndarray):
            self.tau_tracker.append(tau) #TODO: remove
        # else:
        #     self.tau_tracker.append(tau._value) #TODO: remove
        
        _, batch_num = self.get_indices(iter)
        times = self.update_losses(batch_num)
        sum = 0
        for index in range(len(self.losses)):
            alpha = tau *abs( self.t_0-float(times[index][5:]) )
            theta_i = self.opt_fun(self.grad_losses[index], w_b, step_size=alpha, num_iters=self.num_batches_ptask, callback=self.inner_callback)
            out = self.losses[index].forward_pass(self.losses[index].task[1][0], theta_i)
            calc_losses = self.loss_fun(out, self.losses[index].task[1][1])
            sum = sum + calc_losses
        if self.track_loss and isinstance(w_b, SequenceBox) : self.loss_tracker.append(sum._value)
        return sum  

    def print_perf(self, *args, **kargs):
        """
        Prints information to keep us in the know about the learning process
        """
        if self.test_data_gen != False:
            if self.test_fun == False:
                if self.objective_std.has_been_called:
                    ml = Meta_Loss(self.test_data_gen(), self.act_fun, self.opt_fun, self.loss_fun, self.num_batches_ptask, 
                                    self.num_of_tasks_pbatch, self.num_batches_oftasks, self.inner_param)
                    def test(args):
                        ml.batched_tasks_data = self.test_data_gen()
                        w_b, iter = args[0:2]
                        return ml.objective_std(w_b, 0)
                    self.test_fun = test
                    self.test_tracker.append(self.test_fun(args))
                elif self.objective_with_alpha.has_been_called:
                    ml = Meta_Loss(self.test_data_gen(), self.act_fun, self.opt_fun, self.loss_fun, self.num_batches_ptask, 
                                    self. num_of_tasks_pbatch, self.num_batches_oftasks, self.inner_param)
                    def test(args):
                        ml.batched_tasks_data = self.test_data_gen()
                        w_b, alphas, iter = args[0:3]
                        return ml.objective_with_alpha(w_b, alphas, 0)
                    self.test_fun = test
                    self.test_tracker.append(self.test_fun(args))
                elif self.objective_with_time.has_been_called:
                    ml = Meta_Loss(self.test_data_gen(), self.act_fun, self.opt_fun, self.loss_fun, self.num_batches_ptask, 
                                    self. num_of_tasks_pbatch, self.num_batches_oftasks, self.inner_param)
                    ml.t_0 = self.t_0
                    def test(args):
                        ml.batched_tasks_data = self.test_data_gen()
                        w_b, tau, iter = args[0:3]
                        return ml.objective_with_time(w_b, tau, 0)
                    self.test_fun = test
                    self.test_tracker.append(self.test_fun(args))
            else:
                self.test_tracker.append(self.test_fun(args))
        
        print("Meta Epoch: ",self.epoch_num,", Meta Batch: ",self.batch_num)

####### Optimizers
def adam(grad_x0, x0, step_size=0.001, num_iters=100, callback=None, **kwargs):
    """
    Performs optimization using adam
    """
    def two_param(grad_x0, x0, grad_x1, x1, callback=None, step_size=0.001, num_iters=100, b1=0.9, b2=0.999, eps=10**-8):
        len0, _ = flatten(x0)
        m0 = np.zeros(len(len0))
        v0 = np.zeros(len(len0))

        len1, _ = flatten(x1)
        m1 = np.zeros(len(len1))
        v1 = np.zeros(len(len1))
        for i in range(num_iters):
            g_x0 = grad_x0(x0, x1, i)
            x0, unflat_x0 = flatten(x0)
            g_x0, _ = flatten(g_x0)
            m0 = (1 - b1) * g_x0      + b1 * m0  # First  moment estimate.
            v0 = (1 - b2) * (g_x0**2) + b2 * v0  # Second moment estimate.
            mhat0 = m0 / (1 - b1**(i + 1))    # Bias correction.
            vhat0 = v0 / (1 - b2**(i + 1))
            x0 = x0 - step_size*mhat0/(np.sqrt(vhat0) + eps)
            x0 = unflat_x0(x0)

            g_x1 = grad_x1(x0, x1, i)
            x1, unflat_x1 = flatten(x1)
            g_x1, _ = flatten(g_x1)
            m1 = (1 - b1) * g_x1      + b1 * m1  # First  moment estimate.
            v1 = (1 - b2) * (g_x1**2) + b2 * v1  # Second moment estimate.
            mhat1 = m1 / (1 - b1**(i + 1))    # Bias correction.
            vhat1 = v1 / (1 - b2**(i + 1))
            x1 = x1 - step_size*mhat1/(np.sqrt(vhat1) + eps)
            x1 = unflat_x1(x1)

            if callback: callback(x0, x1, i)
        return x0, x1

    if len(kwargs) == 0:
        return adam_one_param(grad_x0, x0, callback=callback, step_size=step_size, num_iters=num_iters)
    elif len(kwargs) == 2:
        grad_x1 = kwargs.get("grad_x1")
        x1 = kwargs.get("x1")
        return two_param(grad_x0, x0, grad_x1, x1, callback=callback, step_size=step_size, num_iters=num_iters)
    else:
        raise Exception("Only support optimizing over one or two parameters")

def basic(grad_x0, x0, step_size=0.001, num_iters=100, callback=None, **kwargs):
    """
    Performs basic optimization
    """
    @unflatten_optimizer
    def one_param(grad, params, callback=None, step_size=0.001, num_iters=100):
        """
        Performs basic optimization
        """
        for i in range(num_iters):
            g = grad(params, i)
            if callback: callback(params, i, g)
            params = params - (step_size*g)
        return params

    def two_param(grad_x0, x0, grad_x1, x1, callback=None, step_size=0.001, num_iters=100):
        for i in range(num_iters):
            g_x0 = grad_x0(x0, x1, i)
            x0, unflat_x0 = flatten(x0)
            g_x0, _ = flatten(g_x0)
            x0 -= g_x0 * step_size
            x0 = unflat_x0(x0)

            g_x1 = grad_x1(x0, x1, i)
            x1, unflat_x1 = flatten(x1)
            g_x1, _ = flatten(g_x1)
            x1 -= g_x1 * step_size
            x1 = unflat_x1(x1)

            if callback: callback(x0, x1, i)
        return x0, x1

    if len(kwargs) == 0:
        return one_param(grad_x0, x0, callback=callback, step_size=step_size, num_iters=num_iters)
    elif len(kwargs) == 2:
        grad_x1 = kwargs.get("grad_x1")
        x1 = kwargs.get("x1")
        return two_param(grad_x0, x0, grad_x1, x1, callback=callback, step_size=step_size, num_iters=num_iters)
    else:
        raise Exception("Only support optimizing over one or two parameters")
    
####### Neural Network
def init_alphas(alpha, data):
    ids = {task[0] for batch_of_tasks in data for task in batch_of_tasks}
    d = {key:alpha for key in ids}
    d["init_param_for_alpha"] = alpha
    return d
    
def init_weights_biases(scale, num_neurons):
    """
    Return a list of tuples containing the matrices for the weights and biases of each layer 
    (matrices of the appropriate size for given layer sizes in the neural network)
    """
    weights = [] #List where we will append our weight matrices for each layer
    for i in range(len(num_neurons)-1):
        weights.append( (npr.random((num_neurons[i],num_neurons[i+1])) * scale , npr.random(num_neurons[i+1])) ) #Appends randomised scaled matrix of weights and bias vector (of correct sixe) 
    return weights

class NeuralNetwork():
    def __init__(self, inner_param, beta,  activation_function, loss_function, scale=0.1, num_neurons = [1,128,64,1], optimizer="basic"):
        self.w_b = init_weights_biases(scale, num_neurons) #Initialises weights for the network
        self.num_neurons = num_neurons #Initialises class attribute containing number of neurons on each layer
        self.inner_param = inner_param #Initialise alpha hyper-parameter for network
        self.beta = beta #Initialise beta hyper-parameter for network
        self.act_fun = activation_function
        self.loss_fun = loss_function
        self.assign_opt(optimizer)

    def assign_opt(self, opt_string):
        if opt_string == "adam":
            self.opt = adam
        elif opt_string == "basic":
            self.opt  = basic
        else:
            raise ValueError("Not given valid optimizer string")

    def train(self, task_data, num_iters, num_batches, task_id=False, set_params=True, track_loss=False, test_data_gen=False, time=False):
        """
        Trains the network on given data for a task using given activation function and loss function
        on num_iters iterations, and retuns loss_function object with final parameters assigned to its attribute theta_i
        and trains according to name of optimization function given
        """
        loss = Loss(task_data, self.act_fun, self.loss_fun, num_batches, track_loss=track_loss, test_data_gen=test_data_gen) 
        grad_loss = grad( loss.objective )

        if task_id == False:
            if isinstance(self.inner_param,dict):
                alpha = self.inner_param.get("init_param_for_alpha")
            elif task_data[0][0:5] == "time:" and isinstance(self.inner_param, tuple):
                alpha = self.inner_param[0] * abs(self.inner_param[1] - float(task_data[0][5:]))
            else:
                alpha = self.inner_param
        else:
            alpha = self.inner_param.get(task_id)
        theta_i = self.opt(grad_loss, self.w_b, step_size=alpha, num_iters=num_iters*num_batches, callback=loss.print_perf) 
        if set_params : self.w_b = theta_i
        return loss, theta_i
    
    def meta_train(self, tasks_data, num_iters, num_batches_ptask, num_of_tasks_pbatch, num_batches_oftasks, 
                    gd_wrt_alpha = False, set_params = True, track_loss=False, time=False, test_data_gen=False):
        """
        Performs meta-learning, given data on tasks, along with various knowledge on the structure of that data
        """
        meta_loss = Meta_Loss(tasks_data, self.act_fun, self.opt, self.loss_fun, num_batches_ptask, num_of_tasks_pbatch, num_batches_oftasks, 
                                self.inner_param, track_loss=track_loss, test_data_gen=test_data_gen)
     
        if gd_wrt_alpha:
            if time!=False:
                raise Exception("Cant have time and gd_wrt_alpha arguments")
            if tasks_data[0][0][0][0:5] == "time":
                raise Exception("Data has wrong id format, is using time format")
            grad_obj_w_b = grad(meta_loss.objective_with_alpha, argnum=0)
            grad_obj_alpha = grad(meta_loss.objective_with_alpha, argnum=1)
            alphas = init_alphas(self.inner_param, tasks_data)
            theta, alphas = self.opt( grad_obj_w_b, self.w_b, step_size=self.beta, num_iters=num_iters, callback=meta_loss.print_perf, grad_x1=grad_obj_alpha, x1=alphas) 
            if set_params : 
                self.w_b = theta
                self.inner_param = alphas
            return meta_loss, theta
        elif time != False:
            if tasks_data[0][0][0][0:5] != "time:":
                raise Exception("Data has wrong id format, is not using time format")
            grad_obj_w_b = grad(meta_loss.objective_with_time, argnum=0)
            grad_obj_tau = grad(meta_loss.objective_with_time, argnum=1)
            meta_loss.t_0 = time[1] 
            theta, tau = self.opt( grad_obj_w_b, self.w_b, step_size=self.beta, num_iters=num_iters, callback=meta_loss.print_perf, grad_x1=grad_obj_tau, x1=time[0]) 
            if set_params : 
                self.w_b = theta
                self.inner_param = (tau, time[1])
            return meta_loss, theta
        else:
            grad_obj_w_b = grad(meta_loss.objective_std, argnum=0)
            theta = self.opt( grad_obj_w_b, self.w_b, step_size=self.beta, num_iters=num_iters, callback=meta_loss.print_perf)
            if set_params : self.w_b = theta
            return meta_loss, theta

####### K-shot
def k_shot(k, data_gen, neural_net):
    k_data, task, task_info = data_gen.k_shot(k)
    _, updated_theta = neural_net.train(k_data, 1, 1, set_params=False)
    return updated_theta, task, task_info, k_data

def n_k_shot(n, k, data_gen, neural_net):
    ls_updated_theta = []
    ls_task = []
    ls_task_info = []
    ls_k_data = []
    for i in range(n):
        updated_theta, task, task_info, k_data = k_shot(k, data_gen, neural_net)
        ls_updated_theta.append(updated_theta)
        ls_task.append(task)
        ls_task_info.append(task_info)
        ls_k_data.append(k_data)
    return   ls_updated_theta, ls_task, ls_task_info, ls_k_data

####### Plotting - Sin Regression
class sin():
    def __init__(self):
        pass
    @staticmethod
    def get_outputs(inputs, act_fun, w_b):
        loss = Loss(None, act_fun, None, None)
        results = []
        xs_t = np.transpose([inputs])
        for x in xs_t:
            results.append( loss.forward_pass(x, w_b)[0] )
        results = np.array(results)
        return results
    @staticmethod
    def flat_data(data_to_flat):
        """
        Give data in meta structure, will flatten such that all x_values pertaining to a certain
        task appear in the same list, does the same for y-values (also strips out the ids for the tasks)
        """
        batches_of_tasks = [batches_of_task for meta_batch in data_to_flat for batches_of_task in meta_batch] #List containing list of batches for each task
        batches_of_tasks = [batches_of_task[1] for batches_of_task in batches_of_tasks] #We get rid of the ID's
        xs = [np.concatenate([np.transpose(batch)[0] for batch in task[0]]) for task in batches_of_tasks] #We flatten the together the batches of each task to get list of x-values for each task
        ys = [np.concatenate([np.transpose(batch)[0] for batch in task[1]]) for task in batches_of_tasks] #We do the same for the y-values (Note that for both we also transpose the batches)
        data = [(xs[index],ys[index]) for index in range(len(xs))] #We create a list of tuples, by putting the x-values and y-values belonging to the same task into a tuple
        return data
    @staticmethod
    def flat_funs_or_params(thing_to_flat):
        """
        Flattens meta structured (only to task depth) data such that it is just a list with each element corresponding to whats at the task depth
        """
        batches_of_tasks = [batches_of_task for meta_batch in thing_to_flat for batches_of_task in meta_batch] #List containing list of batches for each task
        return batches_of_tasks
    @staticmethod
    def plot_fun(file_stem, xs, training_data, results, fun_parameters, loss_function, show=False):
        """
        Debugging plot
        """
        if len(xs)!=len(results):
            raise Exception("Number of x-values does not match number of y-values (results)")
        
        amp , phase = fun_parameters[0].values()
        avg_loss = loss_function(results[0], np.sin(xs+phase)*amp )/len(xs)

        fig = plt.figure()

        plt.plot(xs, np.sin(xs+phase)*amp, color='blue') #Scatter plot of true Sin curve
        # train_xs = [item for batch in training_data[index] for item in batch[0]]
        # train_ys = [item for batch in training_data[index] for item in batch[1]]
        train_xs = training_data[0].flatten()
        train_ys = training_data[1].flatten()
        plt.scatter(train_xs,train_ys, color="lightblue") #Scatter plot of training points used 
        plt.scatter(xs, results, color='red') #Scatter plot of Test points

        plt.title("Simple sin regression")
        plt.xlabel("x\n\n"+"mean "+loss_function.__name__+" loss = "+str(avg_loss))
        plt.ylabel("y")

        # loss_function.__name__+" loss = "+str(avg_loss)
        plt.savefig(file_stem+"_function_"+"_graph", bbox_inches="tight")

        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def plot_test_data_vs_output(training_data, xs, results, filename="test_data_vs_metaout", show=False):
        """
        Plots the data (data given in meta structure) used to train the neural network, each task with a different color,
        then plots on the same graph the results of the network (given) corresponding to the x-values in xs (given)
        """
        colors = {}
        for i in range(len(training_data)):
            for j in range(len(training_data[0])):
                train_xs = training_data[i][j][1][0].flatten()
                train_ys = training_data[i][j][1][1].flatten()

                if colors.get(training_data[i][j][0]) == None :
                    r = npr.random()
                    g = npr.random()
                    b = npr.random()
                    color = (r, g, b, 0.4)
                    colors[training_data[i][j][0]] = color
                else:
                    color = colors.get(training_data[i][j][0])
                plt.scatter(train_xs,train_ys, color=color)

        plt.scatter(xs, results, color='red')

        plt.title("Simple sin regression")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.savefig(filename, bbox_inches="tight")

        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def write_info(*args, filename="sim_info"):
        """
        Creates a file and then writes to the file, on each line printing each key and value pair, 
        it does this for each dictionary given as an argument
        """
        f = open(filename, "w")
        for arg in args:
            for key in arg.keys():
                f.write( str(key) + " : " + str(arg.get(key)) + "\n")
        f.close()
    @staticmethod
    def vis_sin_data(params, data_points, filename="visualisation_of_sin_data", info=True, show=False):
        """
        Plots data points against their underlying sin curves,
        data_points is a list of tuples containing a list of x and a list of y coordinates for a single task (cant have ids)
        (can use flat_data to obtain this kind of structure from meta structured data)
        """
        if len(params) != len(data_points):
            raise Exception("Dont have data points corresponding to each function")
        if (not isinstance(params,list)) and (not isinstance(data_points,list)): 
            # We format arguments as lists if only single function and tuple of data given
            params = [params]
            data_points = [data_points]
        colors = [(random.random(),random.random(),random.random()) for i in range(len(params))]
        min = data_points[0][0][0]
        max = data_points[0][0][0]
        for index in range(len(data_points)):
            if data_points[index][0].min() <= min:
                min = data_points[index][0].min()
            if data_points[index][0].max() >= max:
                max = data_points[index][0].max()
            plt.scatter(data_points[index][0], data_points[index][1], color=colors[index], label="Sampled Data Points From Sin Curve "+str(index)) #Scatter plot of sampled data points
        for index in range(len(params)):
            xs = np.linspace(min,max)
            amp = params[index].get("Amplitude")
            ph = params[index].get("Phase")
            ys = np.sin(xs+ph)*amp
            plt.plot(xs, ys, color=colors[index], label="Sin Curve "+str(index)) #Scatter plot of true Sin curve

        if info == "not_time" :
            for i in range(len(params)):
                if str(params[i]["Phase"])[0] == '-':
                    plt.figtext(0.95, 0.42-(i*0.08), "Sin curve "+str(i)+" : amplitude="+str(params[i]["Amplitude"])[0:5]+", phase="+str(params[i]["Phase"])[0:6])
                else:
                    plt.figtext(0.95, 0.42-(i*0.08), "Sin curve "+str(i)+" : amplitude="+str(params[i]["Amplitude"])[0:5]+", phase="+str(params[i]["Phase"])[0:5])

        elif info == "time" :
            for i in range(len(params)):
                if str(params[i]["Phase"])[0] == '-':
                    plt.figtext(0.98, 0.42-(i*0.08), "Sin curve "+str(i)+" : time="+str(params[i]["Phase"])[0:6])
                else:
                    plt.figtext(0.98, 0.42-(i*0.08), "Sin curve "+str(i)+" : time="+str(params[i]["Phase"])[0:5])

        plt.title("Underlying Sin Curve and Sampled Points")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc=(1.1,0.5))
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod       
    def plot_k_shot(k_shot_data, w_b, act_fun, amp=1.0 , rng=(-5,5), time=False , filename="k_shot", show=False):
        """
        Given list of k - (x, y) coords, with no ids, plots them against the output of the network using parameters of the network after k-shot learning (w_b)
        """
        plt.scatter(k_shot_data[0], k_shot_data[1],label=str(len(k_shot_data[0]))+" Training Data Points")
        ins = np.linspace(rng[0],rng[1])
        outs = get_outputs(ins, act_fun, w_b)
        plt.plot(ins, outs, label="Learned Network Output for Specific Task")
        plt.ylim(-1.0*amp -0.5 ,1.0*amp +0.5)
        if time != False :
            if time[5] == '-':
                plt.figtext(1, 0.42, "Time of task: "+time[5:12])
            else:
                plt.figtext(1, 0.42, "Time of task: "+time[5:11])

        plt.title("K - Sample Points and Output of K-shot Learning")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc=(1.1,0.5))
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def plot_n_k_shot(n_k_shot_data, w_bs, act_fun, amps=False ,rng=(-5,5), times=False, filename="k_shot", show=False):
        """
        Given a list of k-shot data, where each element in structure requested by plot_k_shot, and list of w_b after the k-shot learning
        """
        if amps == False: amps = [amps for i in range(len(n_k_shot_data))]
        if times == False: times = [times for i in range(len(n_k_shot_data))]

        for index in range(len(n_k_shot_data)):
            plot_k_shot(n_k_shot_data[index], w_bs[index], act_fun, amp=amps[index] ,rng=rng, time=times[index], filename=filename+"_"+str(index), show=show)
    @staticmethod
    def plot_k_shot_with_meta(k_shot_data, k_w_b, meta_w_b, act_fun, max_amp=1.0 , rng=(-5,5), time=False , filename="k_shot", show=False):
        """
        Given list of k - (x, y) coords, with no ids, plots them against the output of the network using parameters of the network after k-shot learning (k_w_b),
        and the output of the network immediately after meta-learning (using params meta_w_b)
        """ 
        xs = np.transpose(k_shot_data[0][0])[0]
        ys = np.transpose(k_shot_data[1][0])[0]

        plt.scatter(xs, ys, label=str(len(xs))+" Training Data Points")
        ins = np.linspace(rng[0],rng[1])
        k_outs = sin.get_outputs(ins, act_fun, k_w_b)
        meta_outs = sin.get_outputs(ins, act_fun, meta_w_b)
        plt.plot(ins, k_outs, label="Learned Network Output for Specific Task")
        plt.plot(ins, meta_outs, label="Learned Network Output After Meta-Learning")
        plt.ylim(-1.0*max_amp *1.2 ,1.0*max_amp *1.2)
        if time != False :
            if time[5] == '-':
                plt.figtext(1, 0.42, "Time of task: "+time[5:12])
            else:
                plt.figtext(1, 0.42, "Time of task: "+time[5:11])

        plt.title("K - Sample Points and Output of K-shot Learning")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc=(1.1,0.5))
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def plot_n_k_shot_with_meta(n_k_shot_data, k_w_bs, meta_w_b, act_fun, max_amp=1.0 ,rng=(-5,5), times=False, filename="k_shot", show=False):
        """
        Given list of k-shot data, parameters after k-shot learning plots (where each element in list uses data structure required for plot_k_shot_with_meta),
        plots the k-shot data against the output of the network after k-shot learning and the output of the network immediately after meta-learning (using params meta_w_b)
        """
        if times == False: times = [times for i in range(len(n_k_shot_data))]

        for index in range(len(n_k_shot_data)):
            sin.plot_k_shot_with_meta(n_k_shot_data[index], k_w_bs[index], meta_w_b, act_fun, max_amp=max_amp ,rng=rng, time=times[index], filename=filename+"_"+str(index), show=show)
    @staticmethod
    def metaparam_vs_sin(w_b, act_fun, rng=(-5,5), filename="network_out_vs_sin", show=False):
        """
        Plots output of neural network given network parameters and activation function used for network against a sin curve
        """
        ins = np.linspace(rng[0],rng[1])
        outs = sin.get_outputs(ins, act_fun, w_b)
        plt.plot(ins, outs, label="Network Output")
        plt.plot(ins, np.sin(ins), label="Sin curve")

        plt.title("Output of Meta-learning Trained Network and Generic Sin Curve")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc=(1.1,0.5))
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def training_loss(loss_list, filename="training_loss", show=False, loss_function="Least Squares"):
        """
        Given a list of integer valued training loss values (ordered chronologically for each epoch), it plots them against the epoch it corresponds to
        """
        plt.plot([i for i in range(len(loss_list))], loss_list)

        plt.title("Loss Over Training")
        plt.xlabel("Epoch")
        plt.ylabel(loss_function+" Loss")
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def test_loss(loss_list, filename="test_loss", show=False, loss_function="Least Squares"):
        """
        Given a list of integer valued test loss values (ordered chronologically for each epoch), it plots them against the epoch it corresponds to
        """
        plt.plot([i for i in range(len(loss_list))], loss_list)

        plt.title("Loss on Test Data Over Training")
        plt.xlabel("Epoch")
        plt.ylabel(loss_function+" Test Loss")
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def train_and_test_loss(test_loss_list, train_loss_list, filename="test_and_train_loss", show=False, loss_function="Least Squares"):
        """
        Given a list of integer valued test and training loss values (ordered chronologically for each epoch), it plots them against the epoch it corresponds to
        """
        plt.plot([i for i in range(len(test_loss_list))], test_loss_list, label="Test Loss")
        plt.plot([i for i in range(len(train_loss_list))], train_loss_list, label="Train Loss")

        plt.title("Loss on Test and Train Data Over Training")
        plt.xlabel("Epoch")
        plt.ylabel(loss_function+" Loss")
        plt.legend()
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def sin_points_vs_expected_sin(amp_range, phase_range,num_curves=200,num_points_per_curve=10, rng=(-5,5), time=False):
        x_min, x_max = rng
        a_min, a_max = amp_range
        p_min, p_max = phase_range
        for i in range(num_curves):
            xs = npr.uniform(x_min, x_max, num_points_per_curve)
            a_i = npr.uniform(a_min, a_max)
            p_i = npr.uniform(p_min, p_max)
            ys = np.sin(xs+p_i) * a_i
            r = npr.random()
            g = npr.random()
            b = npr.random()
            color = (r, g, b, 0.4)
            plt.scatter(xs,ys, color=color)
        xs = np.linspace(x_min, x_max)
        plt.plot(xs, np.sin(xs+((p_min+p_max)/2))*((a_min+a_max)/2),color="black")
        plt.xlabel("x")
        plt.ylabel("y")
        if time == False : plt.title("Points from sin curves in task distribution and \"expected\" sin curve")
        else : plt.title("Points from sin curves in task distribution and sin curve at $\mathregular{t_0}$")
        plt.savefig("sin_points_vs_expected_sin")
        plt.close()
    @staticmethod
    def dl_plot_test_data_vs_output(training_data, xs, results, filename="test_data_vs_out", show=False):    
        """
        Plots the data used to train the neural network,
        then plots on the same graph the results of the network (given) corresponding to the x-values in xs (given)
        """
        r = npr.random()
        g = npr.random()
        b = npr.random()
        color = (r, g, b, 0.4)
        for i in range(len(training_data[1][0])):
            train_xs = training_data[1][0][i].flatten()
            train_ys = training_data[1][1][i].flatten()
    
            plt.scatter(train_xs,train_ys, color=color)

        plt.plot(xs, results, color='red')

        plt.title("Simple sin regression")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.savefig(filename, bbox_inches="tight")

        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def dl_vis_sin_data(params, data_points, filename="visualisation_of_sin_data", info=True, show=False, rng=(-5,5)):
        """
        Plots data points against their underlying sin curves,
        data_points is a list of tuples containing a list of x and a list of y coordinates for a single task (cant have ids)
        (can use flat_data to obtain this kind of structure from meta structured data)
        """
        if len(params) != len(data_points):
            raise Exception("Dont have data points corresponding to each function")

        colors = (random.random(),random.random(),random.random())
        # min = data_points[1][0][0][0]
        # max = data_points[1][0][0][0]
        for index in range(len(data_points[1][0])):
            # if data_points[1][0][index].min() <= min:
            #     min = data_points[1][0][index].min()
            # if data_points[1][0][index].max() >= max:
            #     max = data_points[1][0][index].max()
            if index==0:  plt.scatter(data_points[1][0][index], data_points[1][1][index], color=colors, label="Sampled Data Points From Sin Curve") #Scatter plot of sampled data points
            else: plt.scatter(data_points[1][0][index], data_points[1][1][index], color=colors) #Scatter plot of sampled data points

        min_, max_ = rng
        xs = np.linspace(min_,max_)
        amp = params.get("Amplitude")
        ph = params.get("Phase")
        ys = np.sin(xs+ph)*amp
        colors = (random.random(),random.random(),random.random())
        plt.plot(xs, ys, color=colors, label="Sin Curve") #Scatter plot of true Sin curve

        if info != False :
            if str(params["Phase"])[0] == '-':
                plt.figtext(0.95, 0.42, "Sin curve with: amplitude="+str(params["Amplitude"])[0:5]+", phase="+str(params["Phase"])[0:6])
            else:
                plt.figtext(0.95, 0.42, "Sin curve with: amplitude="+str(params["Amplitude"])[0:5]+", phase="+str(params["Phase"])[0:5])

        plt.title("Underlying Sin Curve and Sampled Points")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc=(1.1,0.5))
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def dl_metaparam_vs_sin(w_b, act_fun, sin_curve_params, rng=(-5,5), filename="network_out_vs_sin", show=False):
        """
        Plots output of neural network given network parameters and activation function used for network against a sin curve
        """
        ins = np.linspace(rng[0],rng[1])
        outs = sin.get_outputs(ins, act_fun, w_b)
        plt.plot(ins, outs, label="Network Output")
        plt.plot(ins, np.sin(ins+sin_curve_params["Phase"])*sin_curve_params["Amplitude"], label="Sin curve")

        plt.title("Output of Deep Learning Trained Network and Sin Curve it is trying to learn")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc=(1.1,0.5))
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def sin_time_vis(phase_range, num_curves=20, rng=(-5,5), filename="sin_time_vis", show=False):
        fig = plt.figure()
        ax = plt.axes(projection='3d')  
        x_min, x_max = rng
        p_min, p_max = phase_range
        xs = np.linspace(x_min, x_max)
        for i in range(num_curves):
            p_i = npr.uniform(p_min, p_max)
            ys = np.sin(xs + p_i)
            ax.plot3D(xs, np.full(len(xs),float(p_i)), ys, color=cmx.bwr(p_i+p_max))

        plt.title("Randomly sampled tasks from task distribution")
        ax.set_xlabel("x")
        ax.set_ylabel("time")
        ax.set_zlabel("y")
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def skew_norm(loc = -0.314, scale = 0.4, skewness = 3, numValues = 10000, in_range = False, filename="skewed_norm", show=False):
        random = stats.skewnorm.rvs(a = skewness,loc=loc, scale=scale, size=numValues)
        _, bins, _ = plt.hist(random,30,density=True, color = 'red', alpha=0.4, label="Histogram")
        y = stats.skewnorm.pdf(bins,skewness,loc=loc,scale=scale)
        if in_range != False:
            b, t = in_range
            b_prob = stats.skewnorm.cdf(b,skewness,loc=loc,scale=scale)
            t_prob = 1- stats.skewnorm.cdf(t,skewness,loc=loc,scale=scale)
            plt.figtext(0.98, 0.42, "P(time<"+str(b)[0:7]+")="+str(b_prob)[0:6])
            plt.figtext(0.98, 0.42-0.08, "P(time>"+str(t)[0:6]+")="+str(t_prob)[0:6])
        plt.plot(bins, y, '--', label="Fitted distribution")
        plt.title("Density histogram of skewed normal random variable")
        plt.ylabel("Probability density")
        plt.xlabel("time")
        plt.legend(loc=(1.1,0.5))
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def varying_rate_of_change(fun, time_range, filename="varying_rate_of_change", show=False):
        min_, max_ = time_range
        xs = np.linspace(min_, max_)
        plt.plot(xs, fun(xs), label="New relationship of phase with time")
        plt.plot(xs, xs, '--', label="Previous constant relationship of phase with time")

        plt.title("Phase as function of time")
        plt.ylabel("phase")
        plt.xlabel("time")
        plt.legend(loc=(1.1,0.5))
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def quantitative(data_gen, param_dict, neural_net, test_num=100,filename="quantitative"):
        losses = []
        for i in range(test_num):
            updated_theta, task, task_info, k_data = k_shot(param_dict["num_points"], data_gen, neural_net)

            batch_for_test = data_gen.dp_ret(task, param_dict["num_points"])
       
            xs = np.transpose(batch_for_test[0])[0]
            ys = np.transpose(batch_for_test[1])[0]
            outs = sin.get_outputs(xs, param_dict["activation_function"], updated_theta)
            loss = param_dict["loss_function"](outs,ys)
            losses.append(loss)
        f = open(filename, "w")
        f.write( "Avg least squares loss:" + str(sum(losses)/test_num) +"\n")
        f.write( "Num"+ str(param_dict["num_points"]) +"-shot:" + str(test_num) +"\n")
        lower = np.percentile(losses, 2.5)
        f.write( "Lower 95% CI:"+ str(lower) +"\n")
        upper = np.percentile(losses, 97.5)
        f.write( "Upper 95% CI:"+ str(upper) +"\n")
        f.close()

####### Plotting - Classification
class clas():
    def __init__(self):
        pass
    @staticmethod
    def get_outputs(inputs, act_fun, w_b):
        loss = Loss(None, act_fun, None, None)
        results = []
        xs_t = np.transpose([inputs])
        for x in xs_t:
            results.append( loss.forward_pass(x, w_b)[0] )
        results = np.array(results)
        return results
    @staticmethod
    def flat_data(data_to_flat):
        """
        Give data in meta structure, will flatten such that all x_values pertaining to a certain
        task appear in the same list, does the same for y-values (also strips out the ids for the tasks)
        """
        batches_of_tasks = [batches_of_task for meta_batch in data_to_flat for batches_of_task in meta_batch] #List containing list of batches for each task
        batches_of_tasks = [batches_of_task[1] for batches_of_task in batches_of_tasks] #We get rid of the ID's
        xs = [np.concatenate([np.transpose(batch)[0] for batch in task[0]]) for task in batches_of_tasks] #We flatten the together the batches of each task to get list of x-values for each task
        ys = [np.concatenate([np.transpose(batch)[0] for batch in task[1]]) for task in batches_of_tasks] #We do the same for the y-values (Note that for both we also transpose the batches)
        data = [(xs[index],ys[index]) for index in range(len(xs))] #We create a list of tuples, by putting the x-values and y-values belonging to the same task into a tuple
        return data
    @staticmethod
    def flat_funs_or_params(thing_to_flat):
        """
        Flattens meta structured (only to task depth) data such that it is just a list with each element corresponding to whats at the task depth
        """
        batches_of_tasks = [batches_of_task for meta_batch in thing_to_flat for batches_of_task in meta_batch] #List containing list of batches for each task
        return batches_of_tasks
    @staticmethod
    def flat_times(data_to_flat):
        """
        Flattens meta structured (only to task depth) data such that it is just a list with each element corresponding to whats at the task depth
        """
        batches_of_tasks = [batches_of_task[0] for meta_batch in data_to_flat for batches_of_task in meta_batch] #List containing list of batches for each task
        return batches_of_tasks
    @staticmethod
    def plot_test_data_with_time_vs_output(training_data, xs, results, filename="test_data_vs_metaout", show=False):
        """
        Plots the data (data given in meta structure) used to train the neural network, each task at y-value according to time,
        colors on the colder side of the spectrum corresponding to 0 label and warmer for 1 label,
        then plots on the same graph the results of the network (closer to 0 more blue, closer to 1 more red) corresponding to the x-values in xs
        """
        colors = {}
        for i in range(len(training_data)):
            for j in range(len(training_data[0])):
                flat_xs = training_data[i][j][1][0].flatten()
                flat_ys = training_data[i][j][1][1].flatten()
                time = float(training_data[i][j][0][5:])
                idxs = [idx for idx in range(len(flat_ys)) if flat_ys[idx]==0]

                train_xs_0 = flat_xs[idxs]

                mask = np.ones_like(flat_xs, bool)
                mask[idxs]=False
                train_xs_1 = flat_xs[mask]

                if colors.get(training_data[i][j][0]) == None :
                    color_0 = cmx.winter(random.random())
                    color_0 = (color_0[0], color_0[1], color_0[2], 0.4)
                    color_1 = cmx.spring(random.random())
                    color_1 = (color_1[0], color_1[1], color_1[2], 0.4)     
                    color = (color_0, color_1)
                    colors[training_data[i][j][0]] = color
                else:
                    color = colors.get(training_data[i][j][0])
                plt.scatter(train_xs_0,np.full(len(train_xs_0),time), color=color_0)
                plt.scatter(train_xs_1,np.full(len(train_xs_0),time), color=color_1)

        plt.scatter(xs, np.full(len(xs),-1), color=cmx.bwr(results))

        plt.title("Gaussian Classification")
        plt.xlabel("x")
        plt.ylabel("time")

        plt.savefig(filename, bbox_inches="tight")

        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def vis_clas_data(params, data_points, times, filename="visualisation_of_clas_data", show=False):
        """
        Plots data points against their underlying distributions,
        data_points is a list of tuples containing a list of x and a list of y coordinates for a single task (cant have ids)
        (can use flat_data to obtain this kind of structure from meta structured data)
        """
        if len(params) != len(data_points):
            raise Exception("Dont have data points corresponding to each function")
        if (not isinstance(params,list)) and (not isinstance(data_points,list)): 
            # We format arguments as lists if only single function and tuple of data given
            params = [params]
            data_points = [data_points]

        fig = plt.figure()
        ax = plt.axes(projection='3d')      

        colors = [(random.random(),random.random(),random.random()) for i in range(len(params))]
        min = data_points[0][0][0]
        max = data_points[0][0][0]
        for index in range(len(data_points)):
            if data_points[index][0].min() <= min:
                min = data_points[index][0].min()
            if data_points[index][0].max() >= max:
                max = data_points[index][0].max()
            
            ax.scatter3D(data_points[index][0], np.full(len(data_points[index][0]),float(times[index][5:])), np.zeros(len(data_points[index][0])), color=cmx.bwr(data_points[index][1])) #Scatter plot of sampled data points
        for index in range(len(params)):
            pdf_0 = stats.norm(loc = params[index].get("Mean of zero"), scale = params[index].get("Sd"))
            pdf_1 = stats.norm(loc = params[index].get("Mean of one"), scale = params[index].get("Sd"))
            xs = np.linspace(min,max)
            ys_0 = pdf_0.pdf(xs)
            ys_1 = pdf_1.pdf(xs)
            ax.plot3D(xs, np.full(len(xs),float(times[index][5:])), ys_0, color="blue")
            ax.plot3D(xs, np.full(len(xs),float(times[index][5:])), ys_1, color="red")

        plt.title("Pdf's of Normal distributions with sampled points")
        ax.set_xlabel("x")
        ax.set_ylabel("time")
        ax.set_zlabel("probability density of x given time")
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def metaparam_vs_clas(w_b, act_fun, rng=(-3,3), means=(0,1), sd=0.25, filename="network_out_vs_sin", show=False):
        """
        Plots output of neural network given network parameters and activation function used for network against underlying 
        classification distributions 
        """
        ins = np.linspace(rng[0],rng[1])
        outs = clas.get_outputs(ins, act_fun, w_b)
        plt.plot(ins, outs, label="Network Output")

        mean_0, mean_1 = means
        pdf_0 = stats.norm(loc = mean_0, scale = sd)
        pdf_1 = stats.norm(loc = mean_1, scale = sd)
        ys_0 = pdf_0.pdf(ins)
        ys_1 = pdf_1.pdf(ins)
        plt.plot(ins, ys_0, label="N("+str(mean_0)+","+str(sd)+")")
        plt.plot(ins, ys_1, label="N("+str(mean_1)+","+str(sd)+")")

        plt.title("Output of Meta-learning Trained Network and Gaussian Distributions for time=0")
        plt.xlabel("x")
        plt.ylabel("probability density")
        plt.legend(loc=(1.1,0.5))
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def plot_k_shot_with_meta(k_shot_data, k_w_b, meta_w_b, act_fun , rng=(-5,5), time=False , filename="k_shot", show=False):
        """
        Given list of k - (x, y) coords, with no ids, plots them against the output of the network using parameters of the network after k-shot learning (k_w_b),
        and the output of the network immediately after meta-learning (using params meta_w_b)
        """
        xs = np.transpose(k_shot_data[0][0])[0]
        ys = np.transpose(k_shot_data[1][0])[0]
        idxs = [idx for idx in range(len(ys)) if ys[idx]==0]
        xs_0 = xs[idxs]
        mask = np.ones_like(xs, bool)
        mask[idxs]=False
        xs_1 = xs[mask]
        plt.scatter(xs_0, np.zeros(len(xs_0)),label=str(len(xs_0))+" Training Data Points, label = 0", color="blue")
        plt.scatter(xs_1, np.zeros(len(xs_1)),label=str(len(xs_1))+" Training Data Points, label = 1", color="red")
        ins = np.linspace(rng[0],rng[1])
        k_outs = clas.get_outputs(ins, act_fun, k_w_b)
        meta_outs = clas.get_outputs(ins, act_fun, meta_w_b)
        plt.plot(ins, k_outs, label="Learned Network Output for Specific Task")
        plt.plot(ins, meta_outs, label="Learned Network Output After Meta-Learning")
        if time != False :
            if time[5] == '-':
                plt.figtext(1, 0.42, "Time of task: "+time[5:12])
            else:
                plt.figtext(1, 0.42, "Time of task: "+time[5:11])

        plt.title("K - Sample Points and Output of K-shot Learning")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc=(1.1,0.5))
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def plot_n_k_shot_with_meta(n_k_shot_data, k_w_bs, meta_w_b, act_fun,rng=(-5,5), times=False, filename="k_shot", show=False):
        """
        Given list of k-shot data, parameters after k-shot learning plots (where each element in list uses data structure required for plot_k_shot_with_meta),
        plots the k-shot data against the output of the network after k-shot learning and the output of the network immediately after meta-learning (using params meta_w_b)
        """
        if times == False: times = [times for i in range(len(n_k_shot_data))]

        for index in range(len(n_k_shot_data)):
            clas.plot_k_shot_with_meta(n_k_shot_data[index], k_w_bs[index], meta_w_b, act_fun,rng=rng, time=times[index], filename=filename+"_"+str(index), show=show)
    @staticmethod
    def train_and_test_loss(test_loss_list, train_loss_list, filename="test_and_train_loss", show=False, loss_function="Negative Log Likelihood"):
        """
        Given a list of integer valued test and training loss values (ordered chronologically for each epoch), it plots them against the epoch it corresponds to
        """
        plt.plot([i for i in range(len(test_loss_list))], test_loss_list, label="Test Loss")
        plt.plot([i for i in range(len(train_loss_list))], train_loss_list, label="Train Loss")

        plt.title("Loss on Test and Train Data Over Training")
        plt.xlabel("Epoch")
        plt.ylabel(loss_function+" Loss")
        plt.legend()
        plt.savefig(filename, bbox_inches="tight")
        if show == True:
            plt.show()
        else:
            plt.close()
    @staticmethod
    def write_info(*args, filename="sim_info"):
        """
        Creates a file and then writes to the file, on each line printing each key and value pair, 
        it does this for each dictionary given as an argument
        """
        f = open(filename, "w")
        for arg in args:
            for key in arg.keys():
                f.write( str(key) + " : " + str(arg.get(key)) + "\n")
        f.close()
    @staticmethod
    def accuracy(data_gen, param_dict, neural_net, test_num=100 ,filename="accuracy"):
        accs = []
        for i in range(test_num):
            updated_theta, task, task_info, k_data = k_shot(param_dict["num_points"], data_gen, neural_net)

            batch_for_test = data_gen.dp_ret(task, param_dict["num_points"])
       
            xs = np.transpose(batch_for_test[0])[0]
            ys = np.transpose(batch_for_test[1])[0]
            outs = clas.get_outputs(xs, param_dict["activation_function"], updated_theta)
            outs = [round(out)  for out in outs]
            correct = [1 for i in range(len(outs)) if outs[i]==ys[i]]
            accs.append( sum(correct) /param_dict["num_points"])
        accuracy = (sum(accs)/test_num)
        f = open(filename, "w")
        f.write( "Accuracy:" + str(accuracy) +"\n")
        interval = 1.96 * np.sqrt( (accuracy * (1 - accuracy)) / test_num)
        f.write( "Interval:" + str(interval) +"\n")
        f.close()


####### Functions for running regular learning
def run_reg(data_gen, param_dict, seed=27092021, test=False, track_loss=False, nn =False):
    if isinstance(seed,int):
        npr.seed(seed)
    if test:
        if nn==False: nn = NeuralNetwork(param_dict["alpha"], param_dict["beta"], param_dict["activation_function"], param_dict["loss_function"], optimizer=param_dict["optimizer"])
        batches_for_task, task, task_info = data_gen.task_data(param_dict["num_batches_ptask"],param_dict["num_points"])
        def test_data_gen():
            # batch_for_test, test_task, test_task_info = data_gen.task_data(param_dict["num_batches_ptask"],param_dict["num_points"])
            batch_for_test = data_gen.dp_ret(task, param_dict["num_points"])
            xs = batch_for_test[0]
            ys = batch_for_test[1]
            batch_for_test = [[xs],[ys]]
            return (batches_for_task[0],batch_for_test)
        loss, theta = nn.train(batches_for_task, param_dict["non_meta_num_iters"], param_dict["num_batches_ptask"], test_data_gen=test_data_gen, track_loss=track_loss)
    else:
        nn = NeuralNetwork(param_dict["alpha"], param_dict["beta"], param_dict["activation_function"], param_dict["loss_function"], optimizer=param_dict["optimizer"])
        batches_for_task, task, task_info = data_gen.task_data(param_dict["num_batches_ptask"],param_dict["num_points"])
        loss, theta = nn.train(batches_for_task, param_dict["non_meta_num_iters"], param_dict["num_batches_ptask"],track_loss=track_loss)
    return nn, loss, theta, task, task_info, batches_for_task

####### Functions for running meta learning

def run_meta(data_gen, param_dict, seed = 27092021, test=False, gd_wrt_alpha = False, time=False, track_loss=False):
    if seed != False:
        npr.seed(seed)
    nn = NeuralNetwork(param_dict["alpha"], param_dict["beta"], param_dict["activation_function"], param_dict["loss_function"], optimizer=param_dict["optimizer"])
    meta_train_data, funs, params = data_gen.multiple_tasks_data(param_dict["num_batches_oftasks"], param_dict["num_of_tasks_pbatch"], 
                                    param_dict["num_batches_ptask"], param_dict["num_points"])
    if test: 
        def test_data_gen():
            meta_test_data, test_funs, test_params = data_gen.multiple_tasks_test_data(param_dict["num_of_tasks_pbatch"], param_dict["num_batches_ptask"], 
                                                                                        param_dict["num_points"])
            return meta_test_data
        meta_loss, theta = nn.meta_train(meta_train_data, param_dict["num_iters"], param_dict["num_batches_ptask"], param_dict["num_of_tasks_pbatch"], 
                                        param_dict["num_batches_oftasks"], gd_wrt_alpha=gd_wrt_alpha, track_loss=track_loss, time=time, test_data_gen= test_data_gen)                      
    else:
        meta_loss, theta = nn.meta_train(meta_train_data, param_dict["num_iters"], param_dict["num_batches_ptask"], param_dict["num_of_tasks_pbatch"], 
                                        param_dict["num_batches_oftasks"], gd_wrt_alpha=gd_wrt_alpha, track_loss=track_loss, time = time)
    return nn, meta_loss, theta, funs, params, meta_train_data

def full_run(Parameter_dict, to_plot, run_type, test, track_loss, num_k_shots, override={}): 
    if "Reg_Regr_DL" == run_type:
        dparams = Parameter_dict["Sin_reg"]
        dtype = Sin_Reg(amp_range = dparams["amp_range"], ph_range = dparams["ph_range"], sd = dparams["sd"], sample_rng = dparams["sample_rng"])
        dgen = Data_Generator(dtype.sin_fun_gen, dtype.get_sin_points)

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_regr"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_regr"]
        min_, max_ = dparams["sample_rng"]

        neural_net, loss, _, _, params, train_data = run_reg(dgen, Parameter_dict, track_loss=track_loss, test=test, seed=override.get("seed",Parameter_dict["seed"]), nn=override.get("neural_net",False))
        results = sin.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            sin.dl_plot_test_data_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            sin.dl_vis_sin_data(params, train_data, filename=filename, rng=(min_,max_),info="not_time")
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            print(params)
            sin.dl_metaparam_vs_sin(neural_net.w_b, Parameter_dict["activation_function_regr"], params, filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            max_amp = max([i['Amplitude'] for i in task_info])
            sin.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function_regr"], max_amp=max_amp, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function"] == least_squares :
                sin.train_and_test_loss(loss.test_tracker, loss.loss_tracker,filename=filename, loss_function="Least Squares")
            else:
                raise Exception("Loss function is not least squares, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            sin.write_info(Parameter_dict, {"k-shot info":task_info},filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            print("Not plotted tau as regular MAML doesn't have or learn tau")
        if to_plot["quant"]['plot']==True:
            ovr = to_plot["quant"].get("f_name",False)
            if ovr == False:
                filename = "accuracy"+to_plot["quant"].get("f_extension","")
            else:
                filename = ovr
            sin.quantitative(dgen, Parameter_dict, neural_net, filename=filename)   
        return {
            "DL" : {"neural_net":neural_net, "loss":loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }

    elif "Reg_Regr_DL_twice" == run_type:
        sin.sin_points_vs_expected_sin(Parameter_dict["Sin_reg"]["amp_range"], Parameter_dict["Sin_reg"]["ph_range"],num_curves=100,num_points_per_curve=10, rng=(-5,5))
        run_type = "Reg_Regr_DL"
        test = True
        track_loss = True
        num_k_shots = 20
        info = full_run(Parameter_dict, to_plot, run_type, test, track_loss, num_k_shots)
        print(info["DL"]["neural_net"])
        for plot in to_plot:
            if isinstance(to_plot[plot],dict):
                to_plot[plot]["f_extension"] = "_2ndrun"
        Parameter_dict["num_batches_ptask"] = 10
        Parameter_dict["num_points"] = 10
        iterations = 20 
        Parameter_dict["non_meta_num_iters"] = iterations # * Parameter_dict["num_batches_ptask"] 
        full_run(Parameter_dict, to_plot, run_type, test, track_loss, num_k_shots, override={"neural_net":info["DL"]["neural_net"], "seed":1899334})

    elif "Reg_Regr_MAML" == run_type:
        dparams = Parameter_dict["Sin_reg"]
        dtype = Sin_Reg(amp_range = dparams["amp_range"], ph_range = dparams["ph_range"], sd = dparams["sd"], sample_rng = dparams["sample_rng"])
        dgen = Data_Generator(dtype.sin_fun_gen, dtype.get_sin_points)

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_regr"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_regr"]
        min_, max_ = dparams["sample_rng"]

        neural_net, meta_loss, _, _, params, train_data = run_meta(dgen, Parameter_dict, track_loss=track_loss, test=test, time=False, seed=Parameter_dict["seed"])
        results = sin.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            sin.plot_test_data_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            sin.vis_sin_data(sin.flat_funs_or_params(params)[0:num_data], sin.flat_data(train_data)[0:num_data], filename=filename,info="not_time")
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            sin.metaparam_vs_sin(neural_net.w_b, Parameter_dict["activation_function_regr"], filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            max_amp = max([i['Amplitude'] for i in task_info])
            sin.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function_regr"], max_amp=max_amp, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function_regr"] == least_squares :
                sin.train_and_test_loss(meta_loss.test_tracker, meta_loss.loss_tracker,filename=filename, loss_function="Least Squares")
            else:
                raise Exception("Loss function is not least squares, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            sin.write_info(Parameter_dict, {"k-shot info":task_info},filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            print("Not plotted tau as regular MAML doesn't have or learn tau")
        if to_plot["quant"]['plot']==True:
            ovr = to_plot["quant"].get("f_name",False)
            if ovr == False:
                filename = "accuracy"+to_plot["quant"].get("f_extension","")
            else:
                filename = ovr
            sin.quantitative(dgen, Parameter_dict, neural_net, filename=filename)   
        return {
            "Meta" : {"neural_net":neural_net, "meta_loss":meta_loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }

    elif "Time_Regr_MAML" == run_type:
        dparams = Parameter_dict["Sin_time"]
        dtype = Sin_Time(amp = dparams["amp"], t_0=dparams["t_0"], time_around_t_0=dparams["time_around_t_0"], sd = dparams["sd"], sample_rng=dparams["sample_rng"])
        dgen = Data_Generator(dtype.sin_fun_gen, dtype.get_sin_points)

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_regr"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_regr"]
        min_, max_ = dparams["sample_rng"]

        neural_net, meta_loss, _, _, params, train_data = run_meta(dgen, Parameter_dict, track_loss=track_loss, test=test, time=False, seed=Parameter_dict["seed"])
        results = sin.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            sin.plot_test_data_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            sin.vis_sin_data(sin.flat_funs_or_params(params)[0:num_data], sin.flat_data(train_data)[0:num_data], filename=filename,info="time")
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            sin.metaparam_vs_sin(neural_net.w_b, Parameter_dict["activation_function_regr"], filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            max_amp = max([i['Amplitude'] for i in task_info])
            times= [k[0] for k in k_data]
            sin.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function"], times=times, max_amp=max_amp, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function_regr"] == least_squares :
                sin.train_and_test_loss(meta_loss.test_tracker, meta_loss.loss_tracker,filename=filename, loss_function="Least Squares")
            else:
                raise Exception("Loss function is not least squares, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            sin.write_info(Parameter_dict, {"k-shot info":task_info},filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            print("Not plotted tau as regular MAML doesn't have or learn tau")
        if to_plot["quant"]['plot']==True:
            ovr = to_plot["quant"].get("f_name",False)
            if ovr == False:
                filename = "accuracy"+to_plot["quant"].get("f_extension","")
            else:
                filename = ovr
            sin.quantitative(dgen, Parameter_dict, neural_net, filename=filename)   
        return {
            "Meta" : {"neural_net":neural_net, "meta_loss":meta_loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }

    elif "Time_Regr_MATML" == run_type:
        dparams = Parameter_dict["Sin_time"]
        dtype = Sin_Time(amp = dparams["amp"], t_0=dparams["t_0"], time_around_t_0=dparams["time_around_t_0"], sd = dparams["sd"], sample_rng=dparams["sample_rng"])
        dgen = Data_Generator(dtype.sin_fun_gen, dtype.get_sin_points)

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_regr"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_regr"]
        min_, max_ = dparams["sample_rng"]

        neural_net, meta_loss, _, _, params, train_data = run_meta(dgen, Parameter_dict, track_loss=track_loss, test=test, time=(Parameter_dict["alpha"] , dparams["t_0"]), seed=Parameter_dict["seed"])
        results = sin.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            sin.plot_test_data_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            sin.vis_sin_data(sin.flat_funs_or_params(params)[0:num_data], sin.flat_data(train_data)[0:num_data], filename=filename,info="time")
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            sin.metaparam_vs_sin(neural_net.w_b, Parameter_dict["activation_function_regr"], filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            max_amp = max([i['Amplitude'] for i in task_info])
            times= [k[0] for k in k_data]
            sin.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function"], times=times, max_amp=max_amp, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function_regr"] == least_squares :
                sin.train_and_test_loss(meta_loss.test_tracker, meta_loss.loss_tracker,filename=filename, loss_function="Least Squares")
            else:
                raise Exception("Loss function is not least squares, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            sin.write_info(Parameter_dict, {"k-shot info":task_info}, filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            ovr = to_plot["plot_tau"].get("f_name",False)
            if ovr == False:
                filename = "tau"+to_plot["plot_tau"].get("f_extension","")
            else:
                filename = ovr
            plt.plot([i for i in range(len(meta_loss.tau_tracker))], meta_loss.tau_tracker)
            plt.title("Tau Over Training")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        if to_plot["quant"]['plot']==True:
            ovr = to_plot["quant"].get("f_name",False)
            if ovr == False:
                filename = "accuracy"+to_plot["quant"].get("f_extension","")
            else:
                filename = ovr
            sin.quantitative(dgen, Parameter_dict, neural_net, filename=filename)   
        return {
            "Meta" : {"neural_net":neural_net, "meta_loss":meta_loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }
    
    elif "Time_Clas_MAML" == run_type:
        dparams = Parameter_dict["Clas_time"]
        dtype = Clas_Time(t_0 = dparams["t_0"], time_around_t_0=dparams["time_around_t_0"], sd = dparams["sd"], mean_0 = dparams["mean_0"], mean_1 = dparams["mean_1"])
        dgen = Data_Generator(dtype.clas_fun_gen, dtype.get_clas_points)

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_clas"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_clas"]
        min_, max_ = to_plot.get("clas_plot_rng",(-3,4))

        neural_net, meta_loss, _, _, params, train_data = run_meta(dgen, Parameter_dict, track_loss=track_loss, test=test, time=False, seed=Parameter_dict["seed"])
        results = clas.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            clas.plot_test_data_with_time_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            clas.vis_clas_data(clas.flat_funs_or_params(params)[0:num_data], clas.flat_data(train_data)[0:num_data], clas.flat_times(train_data)[0:5], filename=filename)
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            clas.metaparam_vs_clas(neural_net.w_b, Parameter_dict["activation_function"], rng=(min_,max_), means=(dparams["mean_0"],dparams["mean_1"]), sd=dparams["sd"] ,filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            times= [k[0] for k in k_data]
            clas.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function"], times=times, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function"] == neg_log_likelihood :
                clas.train_and_test_loss(meta_loss.test_tracker, meta_loss.loss_tracker,filename=filename, loss_function="Negative Log Likelihood")
            else:
                raise Exception("Loss function is not neg_log_likelihood, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            clas.write_info(Parameter_dict, {"k-shot info":task_info}, filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            print("Not plotted tau as regular MAML doesn't have or learn tau")
        if to_plot["quant"]['plot']==True:
            ovr = to_plot["quant"].get("f_name",False)
            if ovr == False:
                filename = "accuracy"+to_plot["quant"].get("f_extension","")
            else:
                filename = ovr
            clas.accuracy(dgen, Parameter_dict, neural_net, filename=filename)   
        return {
            "Meta" : {"neural_net":neural_net, "meta_loss":meta_loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }

    elif "Time_Clas_MATML" == run_type:
        dparams = Parameter_dict["Clas_time"]
        dtype = Clas_Time(t_0 = dparams["t_0"], time_around_t_0=dparams["time_around_t_0"], sd = dparams["sd"], mean_0 = dparams["mean_0"], mean_1 = dparams["mean_1"])
        dgen = Data_Generator(dtype.clas_fun_gen, dtype.get_clas_points)

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_clas"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_clas"]
        min_, max_ = to_plot.get("clas_plot_rng",(-3,4))

        neural_net, meta_loss, _, _, params, train_data = run_meta(dgen, Parameter_dict, track_loss=track_loss, test=test, time=(Parameter_dict["alpha"] , dparams["t_0"]), seed=Parameter_dict["seed"])
        results = clas.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            clas.plot_test_data_with_time_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            clas.vis_clas_data(clas.flat_funs_or_params(params)[0:num_data], clas.flat_data(train_data)[0:num_data], clas.flat_times(train_data)[0:5], filename=filename)
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            clas.metaparam_vs_clas(neural_net.w_b, Parameter_dict["activation_function"], rng=(min_,max_), means=(dparams["mean_0"],dparams["mean_1"]), sd=dparams["sd"] ,filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            times= [k[0] for k in k_data]
            clas.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function"], times=times, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function"] == neg_log_likelihood :
                clas.train_and_test_loss(meta_loss.test_tracker, meta_loss.loss_tracker,filename=filename, loss_function="Negative Log Likelihood")
            else:
                raise Exception("Loss function is not neg_log_likelihood, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            clas.write_info(Parameter_dict, {"k-shot info":task_info}, filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            ovr = to_plot["plot_tau"].get("f_name",False)
            if ovr == False:
                filename = "tau"+to_plot["plot_tau"].get("f_extension","")
            else:
                filename = ovr
            plt.plot([i for i in range(len(meta_loss.tau_tracker))], meta_loss.tau_tracker)
            plt.title("Tau Over Training")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        if to_plot["quant"]['plot']==True:
            ovr = to_plot["quant"].get("f_name",False)
            if ovr == False:
                filename = "accuracy"+to_plot["quant"].get("f_extension","")
            else:
                filename = ovr
            clas.accuracy(dgen, Parameter_dict, neural_net, filename=filename)        
        return {
            "Meta" : {"neural_net":neural_net, "meta_loss":meta_loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }

    elif "Time_Regr_MATML_skew_t0" == run_type:
        dparams = Parameter_dict["Sin_time"]
        dtype = Sin_Time(amp = dparams["amp"], t_0=dparams["t_0"], time_around_t_0=dparams["time_around_t_0"], sd = dparams["sd"], sample_rng=dparams["sample_rng"])
        Parameter_dict["Sin_time"]["t_0"] = Parameter_dict["Sin_time"]["t_0"] - Parameter_dict["Sin_time"]["time_around_t_0"]*0.5
        dparams["t_0"] =  Parameter_dict["Sin_time"]["t_0"]
        dgen = Data_Generator(dtype.sin_fun_gen, dtype.get_sin_points)

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_regr"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_regr"]
        min_, max_ = dparams["sample_rng"]

        neural_net, meta_loss, _, _, params, train_data = run_meta(dgen, Parameter_dict, track_loss=track_loss, test=test, time=(Parameter_dict["alpha"] , dparams["t_0"]), seed=Parameter_dict["seed"])
        results = sin.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            sin.plot_test_data_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            sin.vis_sin_data(sin.flat_funs_or_params(params)[0:num_data], sin.flat_data(train_data)[0:num_data], filename=filename,info="time")
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            sin.metaparam_vs_sin(neural_net.w_b, Parameter_dict["activation_function_regr"], filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            max_amp = max([i['Amplitude'] for i in task_info])
            times= [k[0] for k in k_data]
            sin.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function"], times=times, max_amp=max_amp, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function_regr"] == least_squares :
                sin.train_and_test_loss(meta_loss.test_tracker, meta_loss.loss_tracker,filename=filename, loss_function="Least Squares")
            else:
                raise Exception("Loss function is not least squares, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            sin.write_info(Parameter_dict, {"k-shot info":task_info}, filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            ovr = to_plot["plot_tau"].get("f_name",False)
            if ovr == False:
                filename = "tau"+to_plot["plot_tau"].get("f_extension","")
            else:
                filename = ovr
            plt.plot([i for i in range(len(meta_loss.tau_tracker))], meta_loss.tau_tracker)
            plt.title("Tau Over Training")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        return {
            "Meta" : {"neural_net":neural_net, "meta_loss":meta_loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }

    elif "Time_Regr_MATML_skewed_norm_td" == run_type:
        loc = -0.45
        scale = 0.4
        skewness = 3 
        def skew_norm_time_dist():
            return stats.skewnorm.rvs(a = skewness,loc=loc, scale=scale, size=1)[0]
        Parameter_dict["Sin_time"]["t_0"] = stats.skewnorm.mean(skewness,loc=loc,scale=scale)
        dparams = Parameter_dict["Sin_time"]
        dtype = Sin_Time(amp = dparams["amp"], t_0=dparams["t_0"], time_around_t_0=dparams["time_around_t_0"], sd = dparams["sd"], sample_rng=dparams["sample_rng"])
        dtype.time_dist = skew_norm_time_dist
        dgen = Data_Generator(dtype.sin_fun_gen, dtype.get_sin_points)
        
        sin.skew_norm(loc=loc, scale=scale, skewness=skewness, in_range=( - Parameter_dict["Sin_time"]["time_around_t_0"], Parameter_dict["Sin_time"]["time_around_t_0"]))

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_regr"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_regr"]
        min_, max_ = dparams["sample_rng"]

        neural_net, meta_loss, _, _, params, train_data = run_meta(dgen, Parameter_dict, track_loss=track_loss, test=test, time=(Parameter_dict["alpha"] , dparams["t_0"]), seed=Parameter_dict["seed"])
        results = sin.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            sin.plot_test_data_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            sin.vis_sin_data(sin.flat_funs_or_params(params)[0:num_data], sin.flat_data(train_data)[0:num_data], filename=filename,info="time")
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            sin.metaparam_vs_sin(neural_net.w_b, Parameter_dict["activation_function_regr"], filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            max_amp = max([i['Amplitude'] for i in task_info])
            times= [k[0] for k in k_data]
            sin.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function"], times=times, max_amp=max_amp, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function_regr"] == least_squares :
                sin.train_and_test_loss(meta_loss.test_tracker, meta_loss.loss_tracker,filename=filename, loss_function="Least Squares")
            else:
                raise Exception("Loss function is not least squares, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            sin.write_info(Parameter_dict, {"k-shot info":task_info}, filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            ovr = to_plot["plot_tau"].get("f_name",False)
            if ovr == False:
                filename = "tau"+to_plot["plot_tau"].get("f_extension","")
            else:
                filename = ovr
            plt.plot([i for i in range(len(meta_loss.tau_tracker))], meta_loss.tau_tracker)
            plt.title("Tau Over Training")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        return {
            "Meta" : {"neural_net":neural_net, "meta_loss":meta_loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }

    elif "Time_Regr_MATML_rapid_change" == run_type:
        dparams = Parameter_dict["Sin_time"]
        dtype = Sin_Time(amp = dparams["amp"], t_0=dparams["t_0"], time_around_t_0=dparams["time_around_t_0"], sd = dparams["sd"], sample_rng=dparams["sample_rng"])
        def calc_phase_mul(time):
            return time * 2
        dtype.get_phase = calc_phase_mul
        dgen = Data_Generator(dtype.sin_fun_gen, dtype.get_sin_points)

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_regr"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_regr"]
        min_, max_ = dparams["sample_rng"]

        neural_net, meta_loss, _, _, params, train_data = run_meta(dgen, Parameter_dict, track_loss=track_loss, test=test, time=(Parameter_dict["alpha"] , dparams["t_0"]), seed=Parameter_dict["seed"])
        results = sin.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            sin.plot_test_data_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            sin.vis_sin_data(sin.flat_funs_or_params(params)[0:num_data], sin.flat_data(train_data)[0:num_data], filename=filename,info="time")
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            sin.metaparam_vs_sin(neural_net.w_b, Parameter_dict["activation_function_regr"], filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            max_amp = max([i['Amplitude'] for i in task_info])
            times= [k[0] for k in k_data]
            sin.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function"], times=times, max_amp=max_amp, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function_regr"] == least_squares :
                sin.train_and_test_loss(meta_loss.test_tracker, meta_loss.loss_tracker,filename=filename, loss_function="Least Squares")
            else:
                raise Exception("Loss function is not least squares, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            sin.write_info(Parameter_dict, {"k-shot info":task_info}, filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            ovr = to_plot["plot_tau"].get("f_name",False)
            if ovr == False:
                filename = "tau"+to_plot["plot_tau"].get("f_extension","")
            else:
                filename = ovr
            plt.plot([i for i in range(len(meta_loss.tau_tracker))], meta_loss.tau_tracker)
            plt.title("Tau Over Training")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        return {
            "Meta" : {"neural_net":neural_net, "meta_loss":meta_loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }

    elif "Time_Regr_MATML_min_change_t0" == run_type:
        dparams = Parameter_dict["Sin_time"]
        dtype = Sin_Time(amp = dparams["amp"], t_0=dparams["t_0"], time_around_t_0=dparams["time_around_t_0"], sd = dparams["sd"], sample_rng=dparams["sample_rng"])
        def calc_phase_min_change_t0(time):
            return np.square(time) *2 *np.sign(time)
        dtype.get_phase = calc_phase_min_change_t0
        sin.varying_rate_of_change(calc_phase_min_change_t0, (-dparams["time_around_t_0"],dparams["time_around_t_0"]))
        dgen = Data_Generator(dtype.sin_fun_gen, dtype.get_sin_points)

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_regr"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_regr"]
        min_, max_ = dparams["sample_rng"]

        neural_net, meta_loss, _, _, params, train_data = run_meta(dgen, Parameter_dict, track_loss=track_loss, test=test, time=(Parameter_dict["alpha"] , dparams["t_0"]), seed=Parameter_dict["seed"])
        results = sin.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            sin.plot_test_data_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            sin.vis_sin_data(sin.flat_funs_or_params(params)[0:num_data], sin.flat_data(train_data)[0:num_data], filename=filename,info="time")
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            sin.metaparam_vs_sin(neural_net.w_b, Parameter_dict["activation_function_regr"], filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            max_amp = max([i['Amplitude'] for i in task_info])
            times= [k[0] for k in k_data]
            sin.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function"], times=times, max_amp=max_amp, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function_regr"] == least_squares :
                sin.train_and_test_loss(meta_loss.test_tracker, meta_loss.loss_tracker,filename=filename, loss_function="Least Squares")
            else:
                raise Exception("Loss function is not least squares, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            sin.write_info(Parameter_dict, {"k-shot info":task_info}, filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            ovr = to_plot["plot_tau"].get("f_name",False)
            if ovr == False:
                filename = "tau"+to_plot["plot_tau"].get("f_extension","")
            else:
                filename = ovr
            plt.plot([i for i in range(len(meta_loss.tau_tracker))], meta_loss.tau_tracker)
            plt.title("Tau Over Training")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        return {
            "Meta" : {"neural_net":neural_net, "meta_loss":meta_loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }

    elif "Time_Regr_MATML_max_change_t0" == run_type:
        dparams = Parameter_dict["Sin_time"]
        dtype = Sin_Time(amp = dparams["amp"], t_0=dparams["t_0"], time_around_t_0=dparams["time_around_t_0"], sd = dparams["sd"], sample_rng=dparams["sample_rng"])
        def calc_phase_min_change_t0(time):
            if time >= 0:
                return (-np.square(0.75*time-0.5) +0.25)*4
            else:
                return (np.square(0.75*time+0.5) -0.25)*4
        calc_phase_min_change_t0 = np.vectorize(calc_phase_min_change_t0)
        dtype.get_phase = calc_phase_min_change_t0
        sin.varying_rate_of_change(calc_phase_min_change_t0, (-dparams["time_around_t_0"],dparams["time_around_t_0"]))
        dgen = Data_Generator(dtype.sin_fun_gen, dtype.get_sin_points)

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_regr"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_regr"]
        min_, max_ = dparams["sample_rng"]

        neural_net, meta_loss, _, _, params, train_data = run_meta(dgen, Parameter_dict, track_loss=track_loss, test=test, time=(Parameter_dict["alpha"] , dparams["t_0"]), seed=Parameter_dict["seed"])
        results = sin.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            sin.plot_test_data_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            sin.vis_sin_data(sin.flat_funs_or_params(params)[0:num_data], sin.flat_data(train_data)[0:num_data], filename=filename,info="time")
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            sin.metaparam_vs_sin(neural_net.w_b, Parameter_dict["activation_function_regr"], filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            max_amp = max([i['Amplitude'] for i in task_info])
            times= [k[0] for k in k_data]
            sin.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function"], times=times, max_amp=max_amp, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function_regr"] == least_squares :
                sin.train_and_test_loss(meta_loss.test_tracker, meta_loss.loss_tracker,filename=filename, loss_function="Least Squares")
            else:
                raise Exception("Loss function is not least squares, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            sin.write_info(Parameter_dict, {"k-shot info":task_info}, filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            ovr = to_plot["plot_tau"].get("f_name",False)
            if ovr == False:
                filename = "tau"+to_plot["plot_tau"].get("f_extension","")
            else:
                filename = ovr
            plt.plot([i for i in range(len(meta_loss.tau_tracker))], meta_loss.tau_tracker)
            plt.title("Tau Over Training")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        return {
            "Meta" : {"neural_net":neural_net, "meta_loss":meta_loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }

    elif "Time_Regr_MATML_ran" == run_type:
        dparams = Parameter_dict["Sin_time"]
        dtype = Sin_Time(amp = dparams["amp"], t_0=dparams["t_0"], time_around_t_0=dparams["time_around_t_0"], sd = dparams["sd"], sample_rng=dparams["sample_rng"])
        def calc_phase_ran(time):
            return npr.uniform(low=dparams["t_0"]-dparams["time_around_t_0"], high=dparams["t_0"]+dparams["time_around_t_0"])
        dtype.get_phase = calc_phase_ran
        # sin.varying_rate_of_change(calc_phase_ran, (-dparams["time_around_t_0"],dparams["time_around_t_0"]))
        dgen = Data_Generator(dtype.sin_fun_gen, dtype.get_sin_points)

        Parameter_dict["activation_function"] = Parameter_dict["activation_function_regr"]
        Parameter_dict["loss_function"] = Parameter_dict["loss_function_regr"]
        min_, max_ = dparams["sample_rng"]

        neural_net, meta_loss, _, _, params, train_data = run_meta(dgen, Parameter_dict, track_loss=track_loss, test=test, time=(Parameter_dict["alpha"] , dparams["t_0"]), seed=Parameter_dict["seed"])
        results = sin.get_outputs(np.linspace(min_, max_), Parameter_dict["activation_function"], neural_net.w_b) #We get the outputs of the network for range of x-values

        updated_theta, task, task_info, k_data = n_k_shot(num_k_shots, 10, dgen, neural_net)

        if to_plot["plot_test_data_vs_output"]["plot"] ==True:
            ovr = to_plot["plot_test_data_vs_output"].get("f_name",False)
            if ovr == False:
                filename = "test_data_vs_metaout"+to_plot["plot_test_data_vs_output"].get("f_extension","")
            else:
                filename = ovr
            sin.plot_test_data_vs_output(train_data, np.linspace(min_, max_), results, filename=filename)
        if to_plot["vis_data"]["plot"] == True:
            ovr = to_plot["vis_data"].get("f_name",False)
            if ovr == False:
                filename = "visualisation_of_sin_data"+to_plot["vis_data"].get("f_extension","")
            else:
                filename = ovr
            num_data = to_plot["vis_data"].get("num_data",3)
            sin.vis_sin_data(sin.flat_funs_or_params(params)[0:num_data], sin.flat_data(train_data)[0:num_data], filename=filename,info="time")
        if to_plot["metaparam_vs_underlying"]["plot"]==True:
            ovr = to_plot["metaparam_vs_underlying"].get("f_name",False)
            if ovr == False:
                filename = "network_out_vs_sin"+to_plot["metaparam_vs_underlying"].get("f_extension","")
            else:
                filename = ovr
            sin.metaparam_vs_sin(neural_net.w_b, Parameter_dict["activation_function_regr"], filename=filename)
        if to_plot["plot_n_k_shot_with_meta"]["plot"]==True:
            ovr = to_plot["plot_n_k_shot_with_meta"].get("f_name",False)
            if ovr == False:
                filename = "k_shot"+to_plot["plot_n_k_shot_with_meta"].get("f_extension","")
            else:
                filename = ovr
            max_amp = max([i['Amplitude'] for i in task_info])
            times= [k[0] for k in k_data]
            sin.plot_n_k_shot_with_meta([k[1] for k in k_data], updated_theta, neural_net.w_b, Parameter_dict["activation_function"], times=times, max_amp=max_amp, filename=filename, rng=(min_,max_))
        if to_plot["train_and_test_loss"]["plot"]==True:
            ovr = to_plot["train_and_test_loss"].get("f_name",False)
            if ovr == False:
                filename = "train_and_test_loss"+to_plot["train_and_test_loss"].get("f_extension","")
            else:
                filename = ovr
            if Parameter_dict["loss_function_regr"] == least_squares :
                sin.train_and_test_loss(meta_loss.test_tracker, meta_loss.loss_tracker,filename=filename, loss_function="Least Squares")
            else:
                raise Exception("Loss function is not least squares, not given known loss function for this task")
        if to_plot["write_info"]["plot"]==True:
            ovr = to_plot["write_info"].get("f_name",False)
            if ovr == False:
                filename = "sim_info"+to_plot["write_info"].get("f_extension","")
            else:
                filename = ovr
            sin.write_info(Parameter_dict, {"k-shot info":task_info}, filename=filename)
        if to_plot["plot_tau"]["plot"]==True:
            ovr = to_plot["plot_tau"].get("f_name",False)
            if ovr == False:
                filename = "tau"+to_plot["plot_tau"].get("f_extension","")
            else:
                filename = ovr
            plt.plot([i for i in range(len(meta_loss.tau_tracker))], meta_loss.tau_tracker)
            plt.title("Tau Over Training")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        return {
            "Meta" : {"neural_net":neural_net, "meta_loss":meta_loss, "params":params, "train_data":train_data}, 
            "k-shot" : {"updated_theta":updated_theta, "task":task, "task_info":task_info, "k_data":k_data}
            }
    else:
        raise Exception("Not given valid run type")

####### Main
if __name__ == '__main__':
    Parameter_dict = { 
        "seed" : 27092021 , # 27092021

        "alpha" : 0.0025 , #was 0.0025
        "beta" :  0.001 , # change to 0.001

        "activation_function_regr" : hyperbolic ,
        "loss_function_regr" : least_squares ,
        "activation_function_clas" : sigmoid ,
        "loss_function_clas" : neg_log_likelihood ,
        "optimizer" : 'adam' ,
        
        "num_batches_oftasks" : 10 , #Change to 10
        "num_of_tasks_pbatch" : 50 , #Change to 50
        "num_batches_ptask" : 20 , #Change to 1
        "num_points" : 10 , #Change to 10

        "Sin_time" : {"sd":0.1, "time_around_t_0":math.pi*0.2, "amp":1.0, "t_0":0, "sample_rng" : (-5,5)},
        "Clas_time" : {"t_0":0, "time_around_t_0":0.5, "sd":0.5,"mean_0":0, "mean_1":1},
        "Sin_reg" : {"amp_range":(0.9,1.1), "ph_range":(math.pi*-0.1,math.pi*0.1), "sd":0.1, "sample_rng" : (-5,5)}
    }
    iterations = 20 #Change to 50
    Parameter_dict["num_iters"] = Parameter_dict["num_batches_oftasks"] *iterations 
    Parameter_dict["non_meta_num_iters"] = iterations # * Parameter_dict["num_batches_ptask"] 
    
    to_plot = {
        "plot_test_data_vs_output": {"plot":True} ,
        "vis_data": {"plot":True, "num_data":3},
        "metaparam_vs_underlying": {"plot":True},
        "plot_n_k_shot_with_meta": {"plot":True},
        "train_and_test_loss": {"plot":True},
        "write_info": {"plot":True},
        "plot_tau": {"plot":True},

        "clas_plot_rng" : (-3,4), 
        "quant" : {"plot" : True}
    }

    # sin.sin_time_vis( (-Parameter_dict["Sin_time"]["time_around_t_0"], Parameter_dict["Sin_time"]["time_around_t_0"]) )
    # sin.sin_points_vs_expected_sin((1,1), (-Parameter_dict["Sin_time"]["time_around_t_0"], Parameter_dict["Sin_time"]["time_around_t_0"]),num_curves=100,num_points_per_curve=10, rng=(-5,5), time=True)

    test = True
    track_loss = True
    num_k_shots = 1

    run_type =  "Reg_Regr_DL"
    info = full_run(Parameter_dict, to_plot, run_type, test, track_loss, num_k_shots)

    # run_type =  "Time_Regr_MATML_min_change_t0"
    # for key in to_plot.keys():
    #     if isinstance(to_plot[key],dict):
    #         to_plot[key]["f_extension"] = "slow_t0"
    # info = full_run(Parameter_dict, to_plot, run_type, test, track_loss, num_k_shots)

    # run_type =  "Time_Regr_MATML_max_change_t0"
    # for key in to_plot.keys():
    #     if isinstance(to_plot[key],dict):
    #         to_plot[key]["f_extension"] = "rapid_t0"
    # info = full_run(Parameter_dict, to_plot, run_type, test, track_loss, num_k_shots)
