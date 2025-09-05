import random
import streamlit as st


def play_snake(game_state, game_board_size):
    snake = game_state["snake"]
    food = game_state["food"]
    direction = game_state["direction"]

    head_x, head_y = snake[0]
    new_head_x = head_x + direction[0]
    new_head_y = head_y + direction[1]

    if (new_head_x, new_head_y) == food:
        snake.insert(0, (new_head_x, new_head_y))
        game_state["score"] += 1
        game_state["food"] = generate_food(snake, game_board_size)
    elif (
        0 <= new_head_x < game_board_size
        and 0 <= new_head_y < game_board_size
        and (new_head_x, new_head_y) not in snake
    ):
        snake.insert(0, (new_head_x, new_head_y))
        snake.pop()
    else:
        game_state["game_over"] = True
        return

    game_state["snake"] = snake
    draw_board(snake, food, game_board_size)

    keys = st.experimental_get_query_params()
    if "up" in keys and direction != (0, 1):
        game_state["direction"] = (0, -1)
    elif "down" in keys and direction != (0, -1):
        game_state["direction"] = (0, 1)
    elif "left" in keys and direction != (1, 0):
        game_state["direction"] = (-1, 0)
    elif "right" in keys and direction != (-1, 0):
        game_state["direction"] = (1, 0)


def generate_food(snake, game_board_size):
    while True:
        food = (
            random.randint(0, game_board_size - 1),
            random.randint(0, game_board_size - 1),
        )
        if food not in snake:
            return food


def draw_board(snake, food, game_board_size):
    board = [["." for _ in range(game_board_size)] for _ in range(game_board_size)]
    for x, y in snake:
        board[y][x] = "#"
    board[food[1]][food[0]] = "*"
    st.text("\n".join(["".join(row) for row in board]))


def reset_game(game_state, game_board_size):
    game_state["game_over"] = False
    game_state["snake"] = [(game_board_size // 2, game_board_size // 2)]
    game_state["food"] = generate_food(game_state["snake"], game_board_size)
    game_state["direction"] = (1, 0)
    game_state["score"] = 0
    game_state["game_board_size"] = game_board_size
