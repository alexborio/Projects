import numpy as np
from copy import deepcopy


class Player:
    def __init__(self, symbol):
        self.symbol = symbol
        self.state_history = []
        self.all_states = []
        self.enumerate_states(9, 9, '', self.all_states)
        self.init_values = np.random.uniform(0.4, 0.5, size=(3**9,))
        self.value_function = dict(zip(self.all_states, self.init_values))
        self.alpha = 0.5

    def update_state_history(self, state):
        self.state_history.append(state)

    def update(self, env):

        reward = 0

        if env.check_winner(self.symbol):
            reward = 1

        Vprime = reward
        for state in reversed(self.state_history):
            enum_state = self.enumerate_state(state)
            V = self.value_function[enum_state]
            V = V + self.alpha*(Vprime - V)
            self.value_function[enum_state] = V
            Vprime = V

    def take_action(self, env, prob, debug=False):

        allowed_moves = env.get_allowed_moves()
        ln = len(allowed_moves[0])

        values = []

        if debug:
            print(values)

        if np.random.uniform() > prob:
            for i in range(ln):
                index = (allowed_moves[0][i], allowed_moves[1][i])
                board = deepcopy(env.board)
                board[index] = self.symbol
                state = self.enumerate_state(board)
                value = self.value_function[state]
                values.append(value)

            max_index = np.argmax(values)
            index = (allowed_moves[0][max_index], allowed_moves[1][max_index])

        else:
            i = np.random.randint(ln)
            index = (allowed_moves[0][i], allowed_moves[1][i])

        # index = tuple(np.where(values == max_value))
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