def enumerate_states(n_states, state_len, state, results):

    if n_states > 0:

        for sym in ('x', 'o', '_'):
            enumerate_states(n_states - 1, state_len, state + sym, results)

    if len(state) == state_len:
        results.append(state)


state_len = 9
results = []
enumerate_states(state_len, state_len, '', results)
print('done')

