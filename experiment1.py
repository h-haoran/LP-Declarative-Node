import cvxpy as cvx
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import seaborn as sn

# A = np.array([[1,1],[-1,0],[0,-1]])
# B = np.array([[2],[0],[0]])
# C = np.array([3,1])

#entrophy barrier function
def neg_entrophy_method(eta,A,B,C):
    x = cvx.Variable(shape=(2,1))
    barrier = -cvx.sum(cvx.entr(x))
    objective_barrier = cvx.Minimize(eta*cvx.matmul(C,x) + barrier)
    problem_barrier = cvx.Problem(objective_barrier)
    solution_barrier = problem_barrier.solve()
    return x.value


def LPSolver(A, B, C):
    x = cvx.Variable(shape=(2,1))
    constraints = [cvx.matmul(A, x) <= B]
    objective = cvx.Maximize(cvx.matmul(C,x))
    problem = cvx.Problem(objective, constraints)
    problem.solve()
    return x.value

# log barrier function
t = 0.1
def log_barrier_method(t, A, B, C):
    x = cvx.Variable(shape=(2,1))
    log_barrier = cvx.sum(cvx.log(B-cvx.matmul(A,x)))
    cvx.sum((B-cvx.matmul(A,x))**2)
    objective_log_barrier = cvx.Minimize(t * cvx.matmul(C,x) - log_barrier)
    problem_log_barrier = cvx.Problem(objective_log_barrier)
    solution_log_barrier = problem_log_barrier.solve()
    return x.value

# stocahstic approximation

def stochastic_method(t, A, B, C, sample_size):
    x = cvx.Variable(shape=(2,1))
    constraints = [cvx.matmul(A, x) <= B]
    x_perturbed = []
    for i in range(sample_size):
        noise_gumbel = t * rnd.gumbel(size=2)
        C_pertubed = C + noise_gumbel
        objective_perturbed = cvx.Minimize(cvx.matmul(C_pertubed,x))
        problem_perturbed = cvx.Problem(objective_perturbed, constraints)
        solution_perturbed = problem_perturbed.solve()
        if x.value is None:
            print(A,B,C)
        x_perturbed.append(x.value)
    return np.mean(x_perturbed, axis=0)

# Randomize LP problems
LP_problems = []


# i_loop = 0
# while i_loop < 6:
#     a = 20 * rnd.random_sample((3,2)) -10
#     b = 20 * rnd.random_sample((3,1)) -10
#     c = 20 * rnd.random_sample((1,2)) -10
#     if LPSolver(a,b,c) is not None and LPSolver(a,b,-c) is not None:
#         # print(LPSolver(a,b,c))
#         LP_problems.append((a,b,c))
#         i_loop = i_loop + 1

# Simplex
A = np.array([[-1,0],[0,-1],[1,1]]) 
B = np.array([[0],[0],[1]])
C = np.array([3,1])
LP_problems.append((A,B,C))


# Calculate Euclidean distance between two approximation        
distance = np.zeros((10,10))
X_barrier = []
X_perturbed = []
for i in range(10):
    t = 0.1 * (i+1)
    x_perturbed = []
    x_barrier = []
    for A,B,C in LP_problems:
        x_perturbed.append(stochastic_method(1/t,A,B,C, sample_size=500).reshape(2))
        x_barrier.append(neg_entrophy_method(t,A,B,C).reshape(2))
        # x_barrier.append(log_barrier_method(t,A,B,C).reshape(2))
    X_perturbed.append(np.mean(x_perturbed, axis=0))
    X_barrier.append(np.mean(x_barrier, axis=0))


for i in range(10):
    for j in range(10):
        distance[i][j] = np.linalg.norm(X_barrier[i] - X_perturbed[j])

x_axis_labels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # labels for x-axis
# y_axis_labels = [11,22,33,44,55,66,77,88,99,101,111,121] # labels for y-axis
hm = sn.heatmap(data = distance,cmap="mako",xticklabels = x_axis_labels, yticklabels = x_axis_labels)
hm.invert_yaxis()
hm.set_xlabel(r"$stochastic temperature parameter \frac{1}{\epsilon}$", fontsize=10)
hm.set_ylabel('barrier parameter t', fontsize=10)
plt.title("Euclidean distance between optimal and solutions of stochastic and barrier method")
plt.show()
