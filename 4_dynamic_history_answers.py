from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Connect to your document database
persistent_directory = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Set up AI model
model = ChatOpenAI(model="gpt-4o")

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    # print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents.
    - Answer naturally and directly.
    - Avoid robotic or conditional wording.
    - Do not expose reasoning or calculations to the user.
    - Do not phrase answers as "if X, then Y" when you can directly determine the result.
    - Prefer concise, human-friendly responses.
    - Remove redundant information."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="""You are a retrieval-augmented question answering assistant.
Your task is to answer the user's question using ONLY the information provided in the retrieved context.
RULES:
1. Use only information explicitly present in the provided context.
2. Do not invent, assume, infer, or hallucinate any information.
3. Do not add unnecessary disclaimers, apologies, explanations, or meta-commentary.
4. Never say phrases such as:
   - "Unfortunately, there is no additional information..."
   - "I don't have enough information..."
   - "The provided documents do not mention..."
   - "Based on the available information..."
   - "I cannot provide..."
   unless the user explicitly asks whether the information exists.
5. If the context contains relevant information, answer directly using that information.
6. If only partial information is available, answer with the information that is available. Do not explain what is missing.
7. If the context contains no information relevant to the question, respond only with:
   "I couldn't find that information in the provided documents."
8. Keep the answer concise and focused on the user's question.
9. Do not mention the retrieval process, documents, context, embeddings, vector database, or RAG system unless explicitly asked.
10. Do not create additional facts to make the answer appear complete.

IMPORTANT:
The retrieved context is the source of truth. Never use your general knowledge to fill gaps.
"""),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer

# Simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()