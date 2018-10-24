import numpy as np

class Player:
    def __init__(self, symbol):
        self.symbol = symbol
        self.state_history = []
        self.all_states = []
        self.enumerate_states(9, 9, '', self.all_states)
        self.init_values = np.random.uniform(1e-6, 1e-5, size=(3**9,3,3))
        self.value_function = dict(zip(self.all_states, self.init_values))
        self.alpha = 0.001

    def update_state_history(self, state):
        self.state_history.append(state)

    def update(self, env):

        reward = np.zeros(shape=(3,3))

        if env.check_winner(self.symbol):
            indices = env.get_symbol_indices(self.symbol)
            reward[indices] = 1

        Vprime = reward
        for state in reversed(self.state_history):
            enum_state = self.enumerate_state(state)
            V = self.value_function[enum_state]

            V = V + self.alpha*(Vprime - V)

            self.value_function[enum_state] = V
            Vprime = V


    def take_action(self, env, prob, debug=False):
        state = self.enumerate_state(env.board)
        values = self.value_function[state]
        allowed_moves = env.get_allowed_moves()
        max_value = (values[allowed_moves]).max()

        if debug:
            print(values)

        if np.random.uniform() > prob:
            index = tuple(np.where(values == max_value))
        else:
            ln = len(allowed_moves[0])
            i = np.random.randint(0, ln)
            index = (allowed_moves[0][i],allowed_moves[1][i])

        index = tuple(np.where(values == max_value))
        # print(str(index) + ' ' + self.symbol)
        env.make_move(self.symbol, index)
        self.update_state_history(env.board)

    def enumerate_state(self, state):

        state_enum = ''

        for ch in (state.flatten().astype(str)):
            state_enum +=ch

        return state_enum

    def enumerate_states(self, n_states, state_len, state, results):

        if n_states > 0:

            for sym in ('x', 'o', '_'):
                self.enumerate_states(n_states - 1, state_len, state + sym, results)

        if len(state) == state_len:
            results.append(state)

#player = Player('x')