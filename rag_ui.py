import streamlit as st
from rag_query import RAGSystem  # Your multimodal RAG backend

# Cache the heavy RAG system so it loads only once per session
@st.cache_resource
def load_rag_system():
    return RAGSystem()

def main():
    st.set_page_config(page_title="Multimodal RAG Chatbot", layout="wide")
    st.title("Multimodal RAG Chatbot")
    
    # Initialize chat history and rerun trigger in session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize the RAG model (cached)
    rag = load_rag_system()
    
    # Container for chat messages - will be updated dynamically
    chat_container = st.container()
    
    # Chat input - placing it before message display ensures it stays at bottom
    user_input = st.chat_input("Type your message here:")
    
    # Display all messages in the container
    with chat_container:
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["user"])
            with st.chat_message("assistant"):
                st.write(chat["assistant"])
                if chat.get("images"):
                    st.markdown("**Retrieved Images:**")
                    for idx, img in enumerate(chat["images"]):
                        st.image(img, caption=f"Image {idx+1}")
                if chat.get("citations"):
                    st.markdown("**Citations:**")
                    st.write(chat["citations"])
    
    # Handle new user input
    if user_input:
        # First show the user message
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
            
            # Show a placeholder for the assistant's response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    # Prepare chat history
                    history_tuples = [(msg["user"], msg["assistant"]) for msg in st.session_state.chat_history]
                    
                    # Generate response
                    answer_text, images, citations = rag.generate_answer(
                        query=user_input,
                        chat_history=history_tuples
                    )
                    
                    # Display the response components
                    st.write(answer_text)
                    if images:
                        st.markdown("**Retrieved Images:**")
                        for idx, img in enumerate(images):
                            st.image(img, caption=f"Image {idx+1}")
                    if citations:
                        st.markdown("**Citations:**")
                        st.write(citations)
        
        # Update session state after displaying
        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": answer_text,
            "images": images,
            "citations": citations
        })

if __name__ == "__main__":
    main()