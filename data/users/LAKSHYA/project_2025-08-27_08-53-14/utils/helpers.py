import random


def play_round(guess, secret_number):
    if guess == secret_number:
        return {
            "message": "Correct! You earned 10 points!",
            "points": 10,
            "game_over": False,
        }
    elif guess < secret_number:
        return {"message": "Too low!", "points": 0, "game_over": False}
    else:
        return {"message": "Too high!", "points": 0, "game_over": False}
