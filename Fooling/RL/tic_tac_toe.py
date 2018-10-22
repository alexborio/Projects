class Player:
    def __init__(self, symbol):
        self.symbol = symbol
        self.state_history = []
        self.value_function = []

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

        




