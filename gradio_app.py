import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000"  # FastAPI server URL

# Track user session
def login_user(username, password, session):
    payload = {"username": username, "password": password}
    response = requests.post(f"{API_URL}/login", json=payload)

    if response.status_code == 200:
        session["logged_in"] = True
        session["username"] = username
        return f"Welcome, {username}!", True  # Login successful
    else:
        return f"Error: {response.json().get('detail', 'Unknown error')}", False  # Login failed

def register_user(username, password, confirm_password):
    payload = {"username": username, "password": password, "confirm_password": confirm_password}
    response = requests.post(f"{API_URL}/register", json=payload)

    if response.status_code == 200:
        return response.json()["message"]
    else:
        return f"Error: {response.json().get('detail', 'Unknown error')}"

# Chat functionality (Only accessible if logged in)
def submit_message(user_message, chat_history, session):
    if not session.get("logged_in"):
        return "Error: You must be logged in to chat.", chat_history, chat_history

    payload = {"user_message": user_message}
    response = requests.post(f"{API_URL}/chat", json=payload)

    if response.status_code == 200:
        assistant_reply = response.json()["assistant_message"]
    else:
        assistant_reply = "Error: Could not connect to backend."

    chat_history.append((user_message, assistant_reply))
    return "", chat_history, chat_history

# Logout function
def logout_user(session):
    session["logged_in"] = False
    return "You have been logged out.", False  # Update login state

# Main Gradio UI
def main():
    with gr.Blocks() as demo:
        session_state = gr.State({"logged_in": False, "username": ""})

        gr.Markdown("# Chat with GPT-4o-mini via FastAPI Backend")

        with gr.Tabs():
            # Login Tab
            with gr.Tab("Login"):
                gr.Markdown("## Login")
                login_username_input = gr.Textbox(label="Username")
                login_password_input = gr.Textbox(label="Password", type="password")
                login_button = gr.Button("Login")
                login_output = gr.Textbox(label="Status", interactive=False)
                login_status = gr.State(False)

                login_button.click(
                    fn=login_user,
                    inputs=[login_username_input, login_password_input, session_state],
                    outputs=[login_output, login_status]
                )

            # Register Tab
            with gr.Tab("Register"):
                gr.Markdown("## Register")
                username_input = gr.Textbox(label="Username")
                password_input = gr.Textbox(label="Password", type="password")
                confirm_password_input = gr.Textbox(label="Confirm Password", type="password")
                register_button = gr.Button("Register")
                register_output = gr.Textbox(label="Status", interactive=False)

                register_button.click(
                    fn=register_user,
                    inputs=[username_input, password_input, confirm_password_input],
                    outputs=register_output
                )

            # Chat Tab (Visible Only If Logged In)
            with gr.Tab("Chat") as chat_tab:
                with gr.Row(visible=False) as chat_ui:  # Initially hidden
                    chatbot = gr.Chatbot()
                    state = gr.State([])
                    user_input = gr.Textbox(label="Your message")
                    logout_button = gr.Button("Logout")

                    user_input.submit(
                        fn=submit_message,
                        inputs=[user_input, state, session_state],
                        outputs=[user_input, chatbot, state]
                    )

                    logout_button.click(
                        fn=logout_user,
                        inputs=[session_state],
                        outputs=[login_output, login_status]
                    )

            # Dynamic visibility control based on login state
            def toggle_chat_visibility(login_status):
                return gr.update(visible=login_status)

            login_status.change(
                fn=toggle_chat_visibility,
                inputs=[login_status],
                outputs=[chat_ui]
            )

    demo.launch(share=True, server_port=7860)

if __name__ == "__main__":
    main()
