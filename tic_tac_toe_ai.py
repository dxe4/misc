'''
Calculate the scores for tic tac toe moves
needs few improvements
'''
from copy import deepcopy
from collections import defaultdict


INFINITY = float('inf')
possible_players = ['X', 'O']
player_swap = {'X': 'O', 'O': 'X'}


scores = defaultdict(int)


class Game(object):

    def __init__(self, board=None, current_player='X'):
        if board is None:
            board = [[None for j in range(0, 3)] for i in range(0, 3)]

        self.board = board

        self.current_player = current_player
        self._swap = {'X': 'O', 'O': 'X'}

    def swap_player(self):
        self.current_player = player_swap[self.current_player]
        return self.current_player


def possible(board):
    result = []
    for row_idx in range(0, len(board)):
        for col_idx in range(0, len(board[row_idx])):
            if board[row_idx][col_idx] is None:
                result.append((row_idx, col_idx))
    else:
        return result


def non_vertical_rows(board):
    vector_positions = [
        [(0, 0), (1, 1), (2, 2)],
        [(2, 0), (1, 1), (0, 2)],
        #
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
    ]
    result = []
    for pos in vector_positions:
        row = []
        for i in pos:
            y, x = i
            row.append(board[x][y])
        else:
            result.append(row)
    else:
        return result


def clean_item(i):
    if i is None:
        return '-'
    else:
        return i


def get_score(game, depth, maximizing):
    vertical = game.board
    other = non_vertical_rows(game.board)

    if maximizing:
        adjust = 1
    else:
        adjust = -1

    for row in vertical + other:
        row_str = ''.join((clean_item(i) for i in row))
        if game.current_player * 3 in row_str:
            return 1 * adjust
    else:
        return 0


def game_over(game, depth, maximizing):
    score = get_score(game, depth, maximizing)
    return score in (1, -1)


def make_move(game, move):
    board = deepcopy(game.board)
    new_game = Game(board=board, current_player=game.current_player)
    new_game.board[move[0]][move[1]] = new_game.current_player
    new_game.swap_player()

    return new_game


def get_scores(game, depth, maximizing, root_pos):
    moves = possible(game.board)

    if game_over(game, depth, maximizing) or depth == 8 or not moves:
        score = get_score(game, depth, maximizing)
        if score != 0:
            scores[root_pos] += score

        return score

    for move in moves:
        if depth == 0:
            root_pos = move

        new_game = make_move(game, move)
        score = get_scores(new_game, depth + 1, not maximizing, root_pos)

        if score != 0 and depth != 0:
            scores[root_pos] += score
    else:
        return score


if __name__ == '__main__':
    game = Game()
    res = get_scores(game, 0, True, [])
    scores = dict(scores)
    score_sum = sum((abs(i) for i in scores.values()))

    result = []
    for point, score in scores.items():
        result.append([point, float(score) / float(score_sum)])

    result = list(sorted(result, key=lambda x: x[1], reverse=True))
    print(result)
