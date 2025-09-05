import streamlit as st
from utils.helpers import play_snake, reset_game

st.title("Snake Game")

game_state = st.session_state

if "game_over" not in game_state:
    game_state["game_over"] = False

if "snake" not in game_state:
    game_state["snake"] = [(0, 0)]

if "food" not in game_state:
    game_state["food"] = (0, 0)

if "direction" not in game_state:
    game_state["direction"] = (1, 0)

if "score" not in game_state:
    game_state["score"] = 0

if "game_board_size" not in game_state:
    game_state["game_board_size"] = 20

if not game_state["game_over"]:
    game_board_size = game_state["game_board_size"]
    col1, col2 = st.columns([1, 2])
    with col1:
        game_board_size = st.number_input(
            "Board Size:",
            min_value=10,
            max_value=50,
            value=game_state["game_board_size"],
            step=1,
        )
        if game_board_size != game_state["game_board_size"]:
            reset_game(game_state, game_board_size)

    with col2:
        st.write(f"Score: {game_state['score']}")
        play_snake(game_state, game_board_size)

    if game_state["game_over"]:
        st.write("Game Over!")
        if st.button("Play Again"):
            reset_game(game_state, game_board_size)
