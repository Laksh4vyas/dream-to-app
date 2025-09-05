import streamlit as st
from utils.helpers import play_round

st.title("SmashKart Guessing Game")

if "game_state" not in st.session_state:
    st.session_state.game_state = {
        "round": 1,
        "score": 0,
        "max_rounds": 5,
        "secret_number": 0,
    }

if st.session_state.game_state["round"] <= st.session_state.game_state["max_rounds"]:
    st.write(
        f"Round {st.session_state.game_state['round']}/{st.session_state.game_state['max_rounds']}"
    )
    if st.session_state.game_state["round"] == 1:
        st.session_state.game_state["secret_number"] = st.secrets["secret_number"]

    guess = st.number_input(
        "Guess a number between 1 and 10:", min_value=1, max_value=10, step=1
    )
    if st.button("Submit Guess"):
        result = play_round(guess, st.session_state.game_state["secret_number"])
        st.session_state.game_state["score"] += result["points"]
        st.session_state.game_state["round"] += 1
        st.write(result["message"])
        if result["game_over"]:
            st.write(
                f"Game Over! Your final score is: {st.session_state.game_state['score']}"
            )

else:
    st.write("Game Over! Your final score is: ", st.session_state.game_state["score"])
    if st.button("Play Again"):
        st.session_state.game_state = {
            "round": 1,
            "score": 0,
            "max_rounds": 5,
            "secret_number": 0,
        }
