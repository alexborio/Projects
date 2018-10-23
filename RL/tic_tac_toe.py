import game
import player

px = player.Player('x')
po = player.Player('o')

game_ends = False

for i in range(20000):
    env = game.Game(3)
    px.state_history = []
    po.state_history = []
    while not game_ends:

        px.take_action(env, 0.1)
        if env.check_winner(px.symbol):
            game_ends = True
            break

        env.draw_board()

        po.take_action(env, 0.1)
        if env.check_winner(po.symbol):
            game_ends = True
            break

        env.draw_board()

    env.draw_board()

    if env.check_winner(px.symbol):
        print("px won!")

    if env.check_winner(po.symbol):
        print("po won!")

    game_ends = False

    px.update(env)
    po.update(env)
    env = None
    px_symbol = px.symbol
    po_symbol = po.symbol

    px.symbol = po_symbol
    po.symbol = px_symbol

for i in range(2):
    env = game.Game(3)
    px.state_history = []
    px.symbol= 'x'
    while not game_ends:

        px.take_action(env, 0.0)
        if env.check_winner(px.symbol):
            game_ends = True
            break

        env.draw_board()

        num1, num2 = map(int, input().split())

        env.make_move('o', (num1, num2))

        if env.check_winner('o'):
            game_ends = True
            break

        env.draw_board()

    env.draw_board()