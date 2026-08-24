import numpy as np
import matplotlib.pyplot as plt
import argparse
import networkx as nx
import pickle


parser = argparse.ArgumentParser('Generate stochastic graph diffusion data')

parser.add_argument('--graph', type=str, default='ER')
parser.add_argument('--num-nodes', type=int, default=50)
parser.add_argument('--p', type=float, default=0.1)
parser.add_argument('--k', type=int, default=2)

parser.add_argument('--exp_num', type=int, default=1)
parser.add_argument('--tr_num', type=int, default=100)
parser.add_argument('--va_num', type=int, default=30)
parser.add_argument('--te_num', type=int, default=30)

parser.add_argument('--steps', type=int, default=200)
parser.add_argument('--step_size', type=float, default=0.1)

parser.add_argument('--beta', type=float, default=1.0,
                    help='Graph interaction strength')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='Mean-reversion strength; alpha > 0 gives stationarity')
parser.add_argument('--sigma', type=float, default=0.2,
                    help='Noise strength')
parser.add_argument('--burn_in', type=int, default=200,
                    help='Number of initial steps discarded')

args = parser.parse_args()


# dX = -(beta L X + alpha X) dt + sigma dW
def simulation(t):
    dt = args.step_size

    # Initial state Uniform
    #state = np.random.random(n)
    # Initial state: N(0, 1)
    state = np.random.randn(n)

    # Burn-in so saved data starts close to stationarity
    for _ in range(args.burn_in):
        drift = -beta * L @ state - alpha * state
        noise = sigma * np.sqrt(dt) * np.random.randn(n)
        state = state + drift * dt + noise

    x = np.zeros((len(t), n))
    x[0] = state

    for i in range(len(t) - 1):
        drift = -beta * L @ x[i] - alpha * x[i]
        noise = sigma * np.sqrt(dt) * np.random.randn(n)

        x[i + 1] = x[i] + drift * dt + noise

    return np.expand_dims(x, axis=-1)


def plot_trajectory(x, t):
    for nid in range(n):
        plt.plot(t, x[:, nid, 0])

    plt.xlabel('time')
    plt.ylabel('x(t)')
    plt.show()


if __name__ == '__main__':

    assert args.graph in {'ER', 'NWS', 'BA'}, 'Unknown Graph Type'
    assert args.alpha > 0, 'alpha must be positive for stationarity'

    for exp_id in range(args.exp_num):
        n = args.num_nodes
        p = args.p
        k = args.k

        beta = args.beta
        alpha = args.alpha
        sigma = args.sigma

        np.random.seed(exp_id)

        if args.graph == 'ER':
            G = nx.erdos_renyi_graph(n, p, seed=exp_id)
        elif args.graph == 'NWS':
            G = nx.newman_watts_strogatz_graph(n, k, p, seed=exp_id)
        else:
            G = nx.barabasi_albert_graph(n, k, seed=exp_id)

        A = nx.to_numpy_array(G)

        degree = A.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-6))

        # Normalized graph Laplacian
        L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

        # Sufficient Euler stability check since lambda_max(L) <= 2
        if args.step_size * (alpha + 2 * beta) >= 2:
            raise ValueError(
                'step_size is too large for stable Euler simulation'
            )

        t = np.arange(args.steps) * args.step_size

        x_tr = np.zeros((args.tr_num, args.steps, n, 1))
        for i in range(args.tr_num):
            print(f'Simulating train trajectory: {i+1}/{args.tr_num}')
            x_tr[i] = simulation(t)

        x_va = np.zeros((args.va_num, args.steps, n, 1))
        for i in range(args.va_num):
            print(f'Simulating validation trajectory: {i+1}/{args.va_num}')
            x_va[i] = simulation(t)

        x_te = np.zeros((args.te_num, args.steps, n, 1))
        for i in range(args.te_num):
            print(f'Simulating test trajectory: {i+1}/{args.te_num}')
            x_te[i] = simulation(t)

        result = [
            x_tr.astype(np.float32),
            x_va.astype(np.float32),
            x_te.astype(np.float32),
            A.astype(np.float32)
        ]

        data_path = f'SDE{args.sigma}_{args.burn_in}_DF_{args.graph}{n}_exp{exp_id}.pickle'

        with open(data_path, 'wb') as f:
            pickle.dump(result, f)
