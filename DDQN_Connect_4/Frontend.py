import pygame
import sys
from Agent import Agent
from Game import Game
import Utils

WIDTH, HEIGHT = 350, 300
LINE_WIDTH = 5
BOARD_ROWS = 6
BOARD_COLS = 7
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 5

BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
P2_COLOR = (255, 255, 0)
P1_COLOR = (255, 0, 0)


def init_screen():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4")
    screen.fill(BG_COLOR)
    draw_lines(screen)
    return screen


def draw_lines(screen):
    # Horizontal lines
    for r in range(1, BOARD_ROWS):
        pygame.draw.line(
            screen,
            LINE_COLOR,
            (0, r * SQUARE_SIZE),
            (WIDTH, r * SQUARE_SIZE),
            LINE_WIDTH
        )

    # Vertical lines
    for c in range(1, BOARD_COLS):
        pygame.draw.line(
            screen,
            LINE_COLOR,
            (c * SQUARE_SIZE, 0),
            (c * SQUARE_SIZE, HEIGHT),
            LINE_WIDTH
        )


def draw_figures(board, screen):
    board = board[::-1] # flip
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 1:
                pygame.draw.circle(
                    screen,
                    P1_COLOR,
                    (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                     row * SQUARE_SIZE + SQUARE_SIZE // 2),
                    CIRCLE_RADIUS,
                    CIRCLE_WIDTH
                )
            elif board[row][col] == -1:
                pygame.draw.circle(
                    screen,
                    P2_COLOR,
                    (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                     row * SQUARE_SIZE + SQUARE_SIZE // 2),
                    CIRCLE_RADIUS,
                    CIRCLE_WIDTH
                )


def check_win(board, screen, player):
    board = board[::-1] # flip
    # Horizontal
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS - 4 + 1):
            if all(board[row][col + i] == player for i in range(4)):
                draw_horizontal_win(screen, row, col, player)
                return True
    # Vertical
    for col in range(BOARD_COLS):
        for row in range(BOARD_ROWS - 4 + 1):
            if all(board[row + i][col] == player for i in range(4)):
                draw_vertical_win(screen, row, col, player)
                return True
    # Descending diagonal (\)
    for row in range(BOARD_ROWS - 4 + 1):
        for col in range(BOARD_COLS - 4 + 1):
            if all(board[row + i][col + i] == player for i in range(4)):
                draw_desc_diagonal(screen, row, col, player)
                return True
    # Ascending diagonal (/)
    for row in range(4 - 1, BOARD_ROWS):
        for col in range(BOARD_COLS - 4 + 1):
            if all(board[row - i][col + i] == player for i in range(4)):
                draw_asc_diagonal(screen, row, col, player)
                return True
    return False



def draw_vertical_win(screen, row, col, player):
    color = P2_COLOR if player == -1 else P1_COLOR

    x = col * SQUARE_SIZE + SQUARE_SIZE // 2
    y_start = row * SQUARE_SIZE + 15
    y_end = (row + 4) * SQUARE_SIZE - 15

    pygame.draw.line(screen, color, (x, y_start), (x, y_end), LINE_WIDTH)


def draw_horizontal_win(screen, row, col, player):
    color = P2_COLOR if player == -1 else P1_COLOR

    y = row * SQUARE_SIZE + SQUARE_SIZE // 2
    x_start = col * SQUARE_SIZE + 15
    x_end = (col + 4) * SQUARE_SIZE - 15

    pygame.draw.line(screen, color, (x_start, y), (x_end, y), LINE_WIDTH)


def draw_desc_diagonal(screen, row, col, player):
    color = P2_COLOR if player == -1 else P1_COLOR

    x_start = col * SQUARE_SIZE + 15
    y_start = row * SQUARE_SIZE + 15
    x_end = (col + 4) * SQUARE_SIZE - 15
    y_end = (row + 4) * SQUARE_SIZE - 15

    pygame.draw.line(screen, color, (x_start, y_start), (x_end, y_end), LINE_WIDTH)


def draw_asc_diagonal(screen, row, col, player):
    color = P2_COLOR if player == -1 else P1_COLOR

    x_start = col * SQUARE_SIZE + 15
    y_start = (row + 1) * SQUARE_SIZE - 15
    x_end = (col + 4) * SQUARE_SIZE - 15
    y_end = (row - 4 + 1) * SQUARE_SIZE + 15

    pygame.draw.line(screen, color, (x_start, y_start), (x_end, y_end), LINE_WIDTH)



def restart(screen):
    screen.fill(BG_COLOR)
    draw_lines(screen)


def pva(agent):
    pygame.init()
    screen = init_screen()
    game = Game()
    game_over = False

    while True:
        if 0 not in game.board:
            game_over = True
        if agent.player == game.player and not game_over:
            # import cProfile

            # pr = cProfile.Profile()
            # pr.enable()
            move = agent.make_move(game)
            # pr.disable()
            # pr.print_stats(sort="cumulative")
            # sys.exit()
            if check_win(game.board, screen, -game.player):
                game_over = True
            draw_figures(game.board, screen)
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                    x, y = event.pos
                    move = x // SQUARE_SIZE
                    legal_moves = Utils.get_legal_moves(game.board)
                    
                    if move in legal_moves:
                        game.make_move(move)
                        if check_win(game.board, screen, -game.player):
                            game_over = True
                        draw_figures(game.board, screen)

                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    restart(screen)
                    game = Game()
                    game_over = False

        pygame.display.update()


if __name__ == "__main__":
    pva(agent=Agent(1))