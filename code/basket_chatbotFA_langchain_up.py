
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text

# Essential langchain imports only
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter



# Set up the Azure Chat OpenAI model
os.environ["AZURE_OPENAI_API_KEY"] = "#####"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://####"
OPENAI_API_KEY = "###"
OPENAI_DEPLOYMENT_NAME = "gpt4"
MODEL_NAME = "gpt-4"
OPENAI_API_VERSION = "2024-03-01-preview"


'''Feeds data into a vector-based database Chroma using document embeddings generated with Azure OpenAI.

loader = TextLoader(r"C:\backupcgi\tr_sum_with_insights.csv", encoding = 'UTF-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
    documents=chunks, 
    embedding=AzureOpenAIEmbeddings(deployment='textembedding3large',
    model='text-embedding-3-large', 
    azure_endpoint=OPENAI_API_BASE,
    openai_api_key=OPENAI_API_KEY,
    openai_api_type="azure"),
    persist_directory= "../data/chroma_db")
'''

# PostgreSQL connection setup
conn_str = 'postgresql://postgres:####'
engine = create_engine(conn_str)
Session = sessionmaker(bind=engine)
session = Session()

unique_items = pd.read_sql_query("SELECT * FROM top_itemsets",con=engine)

# Initialize the FastAPI app
app = FastAPI()

# Pydantic model for the chatbot request
class ChatRequest(BaseModel):
    question: str
    
    
# Function to query the basket_rule table in PostgreSQL
def query_basket_rule(product_name: str):
    query = text("""
        SELECT itemsets, support
        FROM basket_rule
        WHERE itemsets::text LIKE :product_name
        ORDER BY support DESC
        LIMIT 3;
    """)
    result = session.execute(query, {"product_name": f"%{product_name}%"}).fetchall()
    return result    

# Chatbot initialization code
@app.on_event("startup")
async def startup_event():
    global qa, session

    # Chatbot agent initialization
    chat_model = AzureChatOpenAI(
        openai_api_version="2024-03-01-preview",
        azure_deployment="gpt4",
        temperature=0.4
    )

    emb_model = AzureOpenAIEmbeddings(
        deployment='textembedding3large',
        model='text-embedding-3-large',
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint="https://###",
        openai_api_type="azure"
    )

    def get_retriever():
        loaded_vectordb = Chroma(persist_directory="../data/chroma_db",
                                 embedding_function=emb_model)
        retriever = loaded_vectordb.as_retriever()
        return retriever

    chat_retriever = get_retriever()
    chat_memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
        return_messages=True
    )

    system_template = """
    You are a virtual assistant for a retail company.
    Briefly Answer questions about products, including their details, price, and usage.
    Use the answers from the retrieved document first.
    If you cannot find the answer from the context, just say you don't know nicely.
    ---------------
    {context}
    """
    human_template = """Previous conversation: {chat_history}
    New human question: {question}
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        chain_type='stuff',
        retriever=chat_retriever,
        memory=chat_memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        question = chat_request.question
        res = await qa.acall(question)
        answer = res["answer"]
        
        product_names = extract_product_names(answer)  # `product_names` is expected to be a list
        
        if product_names:
            recommendations_set = set()  # Use a set to avoid duplicates
            for product in product_names:
                basket_results = query_basket_rule(product)
                for row in basket_results:
                    # Convert frozenset representation to a list of items
                    itemsets = row[0].replace("frozenset({", "").replace("})", "").replace("'", "").split(", ")
                    if product in itemsets:
                        itemsets.remove(product)  # Remove the product mentioned by the user
                    recommendations_set.update(itemsets)  # Add items to the set, avoiding duplicates
            
            # Convert the set back to a list if needed
            recommendations = list(recommendations_set)
            
            if recommendations:
                answer += f"\n\n********************************\n\nBased on customer preferences, you might also be interested in: {', '.join(recommendations).lower()}."

                        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_product_names(text: str):
    """
    This function returns a sub-list of items from unique_items['item'] where:
    - Each item is contained as a sub-string within `text`
    OR
    - Each item contains a sub-string of `text`
    """
    prod_lst = list(unique_items['item'])
    
    # Split the text into words to check if any word in the text contains the item
    words_in_text = text.split()

    # Filter items where the item is in the text or any word in the text contains the item
    filtered_prod_lst = [item for item in prod_lst if item in text or any(word in item for word in words_in_text)]
    
    return filtered_prod_lst



@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>AI Chatbot</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #ffe4e1; /* Shallow pink background */
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            #chat-container {
                width: 600px; /* Increase the width of the dialog box */
                background-color: #ffffff;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                border-radius: 10px;
                padding: 20px;
            }
            h1 {
                text-align: center;
                color: #333333;
            }
            #chat-history {
                max-height: 300px;
                overflow-y: auto;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 10px;
            }
            .message {
                margin-bottom: 10px;
            }
            .message.user {
                text-align: right;
                color: #007bff;
            }
            .message.bot {
                text-align: left;
                color: #333333;
            }
            #chat-form {
                display: flex;
                justify-content: space-between;
            }
            #question {
                width: 75%;
                padding: 10px;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                outline: none;
            }
            #question:focus {
                border-color: #007bff;
            }
            button {
                width: 20%;
                padding: 10px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <h1>AI Chatbot</h1>
            <div id="chat-history"></div>
            <form id="chat-form">
                <input type="text" id="question" placeholder="Type your question here...">
                <button type="submit">Send</button>
            </form>
        </div>
        <script>
            document.getElementById('chat-form').onsubmit = async function(e) {
                e.preventDefault();
                const question = document.getElementById('question').value;
                document.getElementById('question').value = '';
                
                // Add user message to chat history
                const userMessage = document.createElement('div');
                userMessage.className = 'message user';
                userMessage.textContent = question;
                document.getElementById('chat-history').appendChild(userMessage);
                
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await response.json();
                
                // Add bot response to chat history
                const botMessage = document.createElement('div');
                botMessage.className = 'message bot';
                botMessage.textContent = data.answer;
                document.getElementById('chat-history').appendChild(botMessage);

                // Scroll chat history to the bottom
                document.getElementById('chat-history').scrollTop = document.getElementById('chat-history').scrollHeight;
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
