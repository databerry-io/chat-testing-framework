import chat
import streamlit as st
from streamlit_chat import message

#Creating the chatbot interface
st.title("Mobius Labs")
st.subheader("Chat Testing Framework")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# Define a function to clear the input text
def clear_input_text():
    global input_text
    input_text = ""

# We will get the user's input by calling the get_text function
def get_text():
    global input_text
    input_text = st.text_input("Ask your Question", key="input", on_change=clear_input_text)
    return input_text

def main():
    user_input = get_text()

    if user_input:
        answer, sources = chat.answer(user_input)

        answer += "\n\n---------------Sources---------------\n\n"
        output = answer + "\n".join(f"Source {i}:\n{source}" for i, source in enumerate(sources))
        # store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# Run the app
if __name__ == "__main__":
    main()

