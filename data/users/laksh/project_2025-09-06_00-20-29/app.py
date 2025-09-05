import streamlit as st
import json
from utils.helpers import load_todos, save_todos, login_user, logout_user

st.title("To-Do List App")

# Load user data (replace with actual authentication mechanism in a real app)
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):  # Placeholder for authentication
            st.session_state.user = username
            st.success(f"Welcome, {username}!")
            st.experimental_rerun()
else:
    todos = load_todos(st.session_state.user)

    new_todo = st.text_input("Add a new to-do:")
    if st.button("Add To-Do"):
        if new_todo:
            todos.append(
                {"task": new_todo, "completed": False, "notes": "", "reminder": ""}
            )
            save_todos(st.session_state.user, todos)
            st.success("To-do added!")
            st.experimental_rerun()

    st.subheader("Your To-Do List")
    for i, todo in enumerate(todos):
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        with col1:
            todo["completed"] = st.checkbox("Completed", value=todo["completed"])
        with col2:
            todo["task"] = st.text_input("", value=todo["task"], key=f"task_{i}")
        with col3:
            todo["notes"] = st.text_area(
                "Notes", value=todo["notes"], key=f"notes_{i}", height=30
            )
        with col4:
            todo["reminder"] = st.date_input(
                "Reminder", value=todo.get("reminder"), key=f"reminder_{i}"
            )

    if st.button("Save Changes"):
        save_todos(st.session_state.user, todos)
        st.success("Changes saved!")

    if st.button("Logout"):
        logout_user()
        st.success("Logged out!")
        st.experimental_rerun()
