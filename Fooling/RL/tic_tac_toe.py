import numpy as np

class Player:
    def __init__(self, symbol):
        self.symbol = symbol
        self.state_history = []
        self.value_function = {}
        self.results = ['']*3**9

    def update_state_history(self, state):
        self.state_history.append(tuple(state))

    def update(self, env):

        terminal_state = env.check_terminal_state(self.symbol)

        if terminal_state == 0:
            self.value_function.append(0.5)
        elif terminal_state == 1:
            self.value_function.append(1)
        elif terminal_state == -1:
            self.value_function.append(0)

    def take_action(self, env):
        pass

    def enumerate_states(self, n_states, result, results):

        if n_states > 0:

            for val in ('x', 'o', '_'):
                result = [r + val for r in result]
                self.enumerate_states(n_states - 1, result, results)

        results.append(result)
        result = ['']

        return results




player = Player('x')
results = player.enumerate_states(9, ['','',''], [])

print(len(results[0]))




