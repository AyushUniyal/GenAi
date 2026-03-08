import streamlit as st
from langchain_core.messages import HumanMessage
import uuid
from ChatBot_sqlite import retrieve_chat


@st.cache_resource
def load_workflow():
    from ChatBot_sqlite import workflow
    return workflow
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_thread"]:
        st.session_state["chat_thread"].append(thread_id)

def laod_converation(thread_id):
    return workflow.get_state(config={"configurable" : {"thread_id": thread_id}}).values["message"]
    
workflow = load_workflow()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = retrieve_chat()

add_thread(st.session_state["thread_id"])

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()


st.sidebar.header("My conversations")
for thread_id in st.session_state["chat_thread"]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        message = laod_converation(thread_id)
        temp_msg = []
        for msg in message:
            if isinstance(msg,HumanMessage):
                role = "user"
            else: 
                role = "AI"
            
            temp_msg.append({"role" : role, "message" : msg.content})
        st.session_state["message_history"] = temp_msg

        






# def stream_data(message):
#         for word in AI_message.split(" "):
#             yield word + " "
#             time.sleep(0.10)



config = {"configurable" : {"thread_id": st.session_state["thread_id"]}}



for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["message"])

userIput = st.chat_input("Type Here...")

if userIput:
    st.session_state["message_history"].append({"role" : "user", "message" : userIput})
    with st.chat_message("user"):
        st.write(userIput)

    # response = workflow.invoke({"message":[HumanMessage(content=userIput)]}, config=config)
    # AI_message = response["message"][-1].content
    # st.session_state["message_history"].append({"role" : "AI", "message" : AI_message})
    with st.chat_message("AI"):
        AI_message = st.write_stream(
             message_chunk.content for message_chunk, metadata in workflow.stream(
                {"message":[HumanMessage(content=userIput)]}, 
                config=config,
                stream_mode="messages"
            )
        )

    st.session_state["message_history"].append({"role" : "AI", "message" : AI_message})


            
# _LOREM_IPSUM =""" 
# Lorem ipsum dolor sit amet, **consectetur adipiscing** elit, sed do eiusmod tempor
# incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
# nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
#  """


# def stream_data():
#     for word in _LOREM_IPSUM.split(" "):
#         yield word + " "
#         time.sleep(0.10)

# if st.button("Stream data"):
#     st.write_stream(stream_data)