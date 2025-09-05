import json
import os


def load_todos(username):
    filepath = f"data/{username}_todos.json"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_todos(username, todos):
    os.makedirs("data", exist_ok=True)
    filepath = f"data/{username}_todos.json"
    with open(filepath, "w") as f:
        json.dump(todos, f, indent=4)


def login_user(username, password):
    # Replace this with your actual authentication logic
    # For this example, we'll just allow any username and password
    return True


def logout_user():
    # Clear session state
    if "user" in st.session_state:
        del st.session_state.user
