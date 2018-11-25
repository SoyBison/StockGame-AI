import random as r
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing as mp

Model = tf.keras.models.Model
Dense = tf.keras.layers.Dense
Conv2D = tf.keras.layers.Conv2D
Input = tf.keras.layers.Input
concatenate = tf.keras.layers.concatenate

naive_q = np.array([[[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]],
                    [[1., 0.],
                     [0., 1.]]])


class StocksGame:
    """
    This class represents the environment which takes a boolean choice in the input and then returns a reward. Then
    the output system will employ some sort of learning, which is ingrained in some other file. On reset, the nature
    will choose three states, x0, x1, news, and fact. The player that I had in mind when writing this knows the first
    three, but not the fact state.  x0 will be zero. this can be changed but it won't really make that much of a
    difference, but the ability to change it is supported. x1 represents the purchase price, and is decided by a markov
    process, which since x0 is constant, means it chooses x1 on a standard normal gaussian curve. Overall, not very
    interesting. fact is a boolean, and is chosen at random. news, is based on fact, and returns fact 75 percent of the
    time, and returns not fact the other 25 percent of the time. Then, x2 is chosen on a normal curve with the mean
    being x1 10 if fact and x1 - 10 if not fact. The variance is the natural logarithm of the absolute value of x, plus
    1. The reward then, is x2-x1.

    :method
    step
    reset
    """

    def __init__(self):
        self.x0 = 0
        self.x1 = r.gauss(0, 1)
        self.fact = r.randrange(100) < 50
        self.news = r.choices([self.fact, not self.fact], weights=[3 / 4, 1 / 4])[0]
        self.x2 = None
        self.reward = 0
        self.done = False
        self.state = [self.x1, self.news, self.done, self.reward]

    def reset(self):
        self.x0 = 0
        self.x1 = r.gauss(0, 2)
        self.fact = r.randrange(100) < 50
        self.news = r.choices([self.fact, not self.fact], weights=[3 / 4, 1 / 4])[0]
        self.x2 = None
        self.reward = 0
        self.done = False
        self.state = [self.x1, self.news, self.done, self.reward]
        return self.state

    def step(self, action):
        if not self.done:
            self.x2 = r.gauss(self.x1 + 5 - 10 * (not self.fact), math.log(abs(self.x1)) + 2)
            if action:
                self.reward = self.x2 - self.x1
                self.done = True
                self.state = [self.x2, self.news, self.done, self.reward]
                return self.state
            if not action:
                self.done = True
                self.reward = 0
                self.state = [self.x2, self.news, self.done, self.reward]
                return self.state
        if self.done:
            print("Game is over, environment must be reset.")


def eps_q_learning(env, episodes=500, eps=.5, lr=.8, y=.95, decay_factor=.999):
    q_table = np.zeros((26, 2, 2))
    bins = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    for i in range(episodes):
        x1, news, done = env.reset()[0:3]
        x1 = np.digitize(x1, bins)
        eps *= decay_factor
        if np.random.random() < eps or np.sum(q_table[x1, int(news), :]) == 0:
            a = np.random.randint(0, 2)
        else:
            a = np.argmax(q_table[x1, int(news), :])
        _, news, done, reward = env.step(a)
        q_table[x1, int(news), a] += reward + lr * (y * np.max(q_table[x1, int(news), a]) - q_table[x1, int(news), a])
    return q_table


def test_q_table(array: np.ndarray, env: StocksGame):
    if array.shape != (26, 2, 2):
        raise TypeError("Wrong sized table supplied.")
    tot_reward = 0
    for _ in range(100):
        s, n, _, _ = env.reset()
        bins = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        s = np.digitize(s, bins)
        a = np.argmax(array[s, int(n), :])
        _, _, _, reward = env.step(a)
        tot_reward += reward
    return tot_reward


def comprehensive_test(array: np.ndarray, env: StocksGame):
    if array.shape != (26, 2, 2):
        raise TypeError("Wrong sized table supplied.")
    a = []
    for _ in range(100):
        a.append(test_q_table(array, env))
    return np.mean(a)


def keras_q_learning(env: StocksGame, episodes=500, eps=.5, y=.95, decay_factor=.999, plot=True):
    # build model:
    game_state = Input(shape=(28,))

    layer1 = Dense(64, activation='softsign')(game_state)
    layer2 = Dense(2, activation='softsign')(layer1)
    model = Model(game_state, outputs=layer2)
    model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'])
    bins = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    q_vector = np.zeros((26, 2, 2))
    r_avg_list = []
    for i in range(episodes):
        x1, n, _, _ = env.reset()
        x1 = np.digitize(x1, bins)
        x1_mat = np.identity(26)[x1]
        n = int(n)
        n_mat = np.identity(2)[n]
        eps *= decay_factor
        state = np.array([np.concatenate((x1_mat, n_mat))])  # this seems idiosyncratic, I realize, but keras has some
        #  problems with constructing multi-input models.
        if np.random.random() < eps:
            a = np.random.randint(0, 2)
        else:
            a = np.argmax(model.predict(state))
        _, _, done, reward = env.step(a)
        prediction = model.predict(state)
        target = reward + y * np.max(prediction)
        q_vector[x1, n, a] = target
        model.fit(state, q_vector[x1, n, :].reshape(-1, 2), epochs=1, verbose=0)
        r_avg_list.append(reward)
    if plot:
        plt.plot(r_avg_list)
        plt.ylabel('Average Profit per game')
        plt.xlabel('Number of games')
        plt.show()
    return q_vector


def construct_super_q(env, generations=100):
    qs = []
    scores = []
    for i in range(generations):
        q = eps_q_learning(env)
        qs.append(q)
        scores.append(comprehensive_test(q, env))
        print(i / generations * 100, " percent complete")
    print(max(scores))

    out = np.average(qs, weights=scores, axis=0)
    return out


def construct_super_keras_q(env, generations=100, episodes=500, eps=.6, y=.95, decay_factor=.999):
    qs = []
    scores = []
    for i in range(generations):
        q = keras_q_learning(env, episodes=episodes, eps=eps, y=y, decay_factor=decay_factor, plot=False)
        qs.append(q)
        scores.append(comprehensive_test(q, env))
        print(i / generations * 100, " percent complete")
    print(max(scores))
    return sum(qs) / len(qs)


def method_tester(episodes=1000, eps=.6, y=.95, decay_factor=.999):
    env = StocksGame()
    q = keras_q_learning(env, episodes=episodes, eps=eps, y=y, decay_factor=decay_factor, plot=False)
    b = eps_q_learning(env, episodes=episodes, eps=eps, y=y, decay_factor=decay_factor)
    b = comprehensive_test(b, env)
    q = comprehensive_test(q, env)
    print("Traditional Q learning yields ", b)
    print("Neural Network Based Q learning yields ", q)
    return [q, b]


def long_test(rounds=100, episodes=500, eps=.6, y=.95, decay_factor=.999):
    qnn = []
    qtrad = []
    for i in range(rounds):
        x = method_tester(episodes=episodes, eps=eps, y=y, decay_factor=decay_factor)
        qnn.append(x[0])
        qtrad.append(x[1])
    qnn = np.mean(qnn)
    qtrad = np.mean(qtrad)
    print("Average effectiveness of Neural Network based Q learning: %f" % qnn)
    print("Average effectiveness of traditional Q learning: %f " % qtrad)


if __name__ == '__main__':
    game = StocksGame()
    empty = [game] * 100
    p = mp.Pool()
    tables = p.map(eps_q_learning, empty)

    tables = sum(tables) / len(tables)
    print(comprehensive_test(tables, game))
