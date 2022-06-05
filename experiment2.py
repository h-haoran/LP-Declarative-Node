import cvxpy as cvx
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import seaborn as sn



def LPSolver(A, B, C):
    x = cvx.Variable(shape=(2,1))
    constraints = [cvx.matmul(A, x) <= B]
    objective = cvx.Minimize(cvx.matmul(C,x))
    problem = cvx.Problem(objective, constraints)
    problem.solve()
    return x.value

A = np.array([[-1,0],[0,-1],[1,1]]) 
B = np.array([[0],[0],[1]])
# C_true = np.array([3,1])
C_true = np.array([-1,-1])
C = np.array([1,2])
x_star = LPSolver(A,B,C_true)

t = 0.5
def loss_log_barrier(t, A, B, C, x_star):
    
    x = cvx.Variable(shape=(2,1))
    # constraints = [cvx.matmul(A, x) <= B]
    log_barrier = cvx.sum(cvx.log(B-cvx.matmul(A,x))) * t
    # cvx.sum((B-cvx.matmul(A,x))**2)
    objective_log_barrier = cvx.Minimize(cvx.matmul(C,x) - log_barrier)
    problem_log_barrier = cvx.Problem(objective_log_barrier)
    solution_log_barrier = problem_log_barrier.solve()
    # x1 = x.value[0][0]
    # x2 = x.value[1][0]
    # dxx = -t *np.array([[1/x1,0],[0,1/x2]]) 
    # dxc = np.eye(2)
    # grad_barrier  = np.linalg.inv(dxx) @ dxc
    # print(solution_barrier)
    # print(x.value)
    # print(grad_barrier)
    grad =  gradient_log_barrier(t,A,B,C,x.value,x_star)
    return np.linalg.norm(x.value-x_star)/2, grad

def gradient_log_barrier(t, A, B, C, x, x_star):
    dxx = np.zeros((2,2))
    for i in range(3):
        a = A[i,:].reshape((2,1))
        dxx = dxx + a @ a.T
    dxx = dxx/ ((B[0] - A[0,:]@x)**2 + (B[1] - A[1,:]@x)**2 +(B[2] - A[2,:]@x)**2)
    dxc = t * np.eye(2)
    grad_barrier  = -np.linalg.inv(dxx) @ dxc
    grad_barrier = (x - x_star).reshape((1,2)) @ grad_barrier
    return grad_barrier

# loss_log_barrier(t,A,B,C, [[0],[0]])

############
def loss_neg_entrophy(eta,A,B,C,x_star):
    x = cvx.Variable(shape=(2,1))
    constraints = [cvx.matmul(A, x) <= B]
    barrier = -cvx.sum(cvx.entr(x))
    objective_barrier = cvx.Minimize(eta*cvx.matmul(C,x) + barrier)
    problem_barrier = cvx.Problem(objective_barrier)
    solution_barrier = problem_barrier.solve()
    grad =  grad_neg_entrophy(eta, A, B, C, x, x_star)
    return np.linalg.norm(x.value-x_star)/2,grad

def grad_neg_entrophy(eta,A,B,C,x,x_star):
    x1 = x.value[0][0]
    x2 = x.value[1][0]
    dxx = -np.array([[1/x1,0],[0,1/x2]]) 
    dxc = eta*np.eye(2)
    grad_barrier  = np.linalg.inv(dxx) @ dxc
    grad_barrier =  (x.value - x_star).reshape((1,2)) @ grad_barrier
    return grad_barrier

##############
temp = 1
sample_size = 440

def loss_grad_stochastic(temp, A, B, C, sample_size, x_star):
    x = cvx.Variable(shape=(2,1))
    constraints = [cvx.matmul(A, x) <= B]
    x_perturbed = []
    grad_perturbed = []
    for i in range(sample_size):
        noise_gumbel = temp * rnd.gumbel(size=2)
        C_pertubed = -C + noise_gumbel
        # print(C_pertubed, C)
        objective_perturbed = cvx.Maximize(cvx.matmul(C_pertubed,x))
        problem_perturbed = cvx.Problem(objective_perturbed, constraints)
        solution_perturbed = problem_perturbed.solve()
        v = temp*noise_gumbel + np.exp(-temp*noise_gumbel)
        dv = [1,1]-np.exp(-noise_gumbel)
        # print(dv)
        grad_perturbed.append ((x.value.reshape(2,1) @ dv.reshape(1,2)) / temp)
        if x.value is None:
            print(A,B,C)
        x_perturbed.append(x.value)
    grad = np.mean(grad_perturbed, axis=0)
    x_sol = np.mean(x_perturbed, axis=0)
    grad = (x_sol - x_star).reshape((1,2)) @ grad
    # print(grad)
    # print(x_sol)
    return np.linalg.norm(x_sol-x_star)/2,-grad


# loss_neg_entrophy(t,A,B,C,x_star)
# loss_log_barrier(t,A,B,C,x_star)
# print(loss_grad_stochastic(temp,A,B,C,sample_size,x_star))

# c     -> x
# f     -> L(log/entrophy/barrier)
# df(x) -> g(x)
# dx    -> -1*g(x)
def obj_log(c):
    return loss_log_barrier(t,A,B,c,x_star)[0]
def obj_negent(c):
    return loss_neg_entrophy(t,A,B,c,x_star)[0]
def obj_stochastic(c):
    return loss_grad_stochastic(temp,A,B,c,sample_size,x_star)[0]
def grad_log(c):
    return loss_log_barrier(t,A,B,c,x_star)[1]
def grad_negent(c):
    return loss_neg_entrophy(t,A,B,c,x_star)[1]
def grad_stochastic(c):
    return loss_grad_stochastic(temp,A,B,c,sample_size,x_star)[1]

# print(obj_log(C))
# print(obj_negent(C))
# print(obj_stochastic(C))
# print(grad_log(C))
# print(grad_negent(C))
# print(grad_stochastic(C))


def gradient_descent_no_linesearch(x, f, g, eps=1.0e-3, max_iters=500, t=1.0e-1, gamma=0.98):
    """
    Implements gradient descent with fixed step size to minimize function f.

    :param x: Starting point in domain of f.
    :param f: The function to be optimized. Returns scalar.
    :param g: The gradient function. Returns vector in R^n.
    :param eps: Tolerance for stopping.
    :param max_iters: Maximum number of iterations for stopping.
    :param t: Initial step size parameter.
    :param gamma: Step size decay schedule (set to 1.0 for fixed step size).
    :return: Optimization path (i.e., array of x's). The last point is the optimal point.
    """

    path = [x.copy()]

    for iter in range(max_iters):
        # Compute gradient
        dx = -1.0 * g(x) / np.linalg.norm(g(x))
        # Stopping criterion
        print("...iter {}, f(x) = {}".format(iter, f(x)))
        if np.linalg.norm(f(x)) <= eps:
            break

        # Update
        x_new = x + t * dx
        # print(x_new)
        if f(x_new) != float('inf') and np.linalg.norm(x_new) >= 1e-3:
            x = x_new
            path.append(x.copy())
            
        # step size decay
        t *= gamma 
    return path


log_path = gradient_descent_no_linesearch(C.reshape((1,2)), obj_log, grad_log)
negent_path = gradient_descent_no_linesearch(C.reshape((1,2)), obj_negent, grad_negent)
stochastic_path = gradient_descent_no_linesearch(C.reshape((1,2)), obj_stochastic, grad_stochastic)


plt.figure()
plt.semilogy(range(len(log_path)), [obj_log(c) for c in log_path], lw=2)
plt.semilogy(range(len(negent_path)), [obj_negent(c)  for c in negent_path], lw=2)
plt.semilogy(range(len(stochastic_path)), [obj_stochastic(c) for c in stochastic_path], lw=2)
plt.title("Learning curve of different methods")
plt.xlabel("$iterations$"); plt.ylabel(r"$\frac{1}{2}\|x - x^\star\|$")
plt.legend(["logarithm barrier", "negative entropy", "stochastic"])
plt.show()





