import numpy as np
import cvxpy as cp


class GameAgent(object):
    def __init__(self):
        pass

    def solve(self, R):
        n = len(R)
        A = np.array(R).T
        A *= -1
        Y = [[1, 1, 1], [-1, -1, -1]]
        A = np.vstack((A, Y))

        utility = [1 for i in range(n)] + [0 for i in range(n - 1)]

        A = np.insert(A, 0, utility, axis=1)
        b = np.array([0, 0, 0, 1, -1])
        c = np.array([-1, 0, 0, 0])

        # Define and solve the CVXPY problem.
        x = cp.Variable(n + 1, pos=True)
        prob = cp.Problem(cp.Minimize(c.T@x), [A@x <= b, x >= 0])
        prob.solve()
        # print(x.value[1:])
        return x.value[1:]

