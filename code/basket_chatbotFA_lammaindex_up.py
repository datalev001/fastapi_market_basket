# Cleaned up code with unnecessary imports removed

import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings

# Set up the Azure OpenAI model
aoai_api_key = "####"
aoai_endpoint = "https://####"
aoai_api_version = "2024-03-01-preview"

# PostgreSQL connection setup
conn_str = 'postgresql://postgres:####'
engine = create_engine(conn_str)
Session = sessionmaker(bind=engine)
session = Session()
unique_items = pd.read_sql_query("SELECT * FROM top_itemsets", con=engine)

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

# Initialize Azure OpenAI and embedding models
llm = AzureOpenAI(engine="gpt4", model="gpt-4",
    api_key=aoai_api_key,
    azure_endpoint=aoai_endpoint,
    api_version=aoai_api_version)

embed_model = AzureOpenAIEmbedding(
    model='text-embedding-3-large',
    deployment_name='textembedding3large',
    api_key=aoai_api_key,
    azure_endpoint=aoai_endpoint,
    api_version=aoai_api_version)

Settings.llm = llm
Settings.embed_model = embed_model

# Load documents and set up vector store index
documents = SimpleDirectoryReader("../data", encoding='UTF-8').load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)
query_engine = index.as_query_engine()

# Chatbot initialization code
@app.on_event("startup")
async def startup_event():
    global query_engine
    query_engine = index.as_query_engine()

# POST request to handle chatbot interactions
@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    try:
        question = chat_request.question
        print('question:', question)
        
        # Query the vector store using the provided question
        resp = query_engine.query(question)
        answer = str(resp.response)
        print('answer:', answer)
        
        product_names = extract_product_names(answer)
        
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

# Function to extract product names from chatbot answer
def extract_product_names(text: str):
    prod_lst = list(unique_items['item'])
    
    # Split the text into words and check for matches with product names
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
                background-color: #e0f7fa; /* Shallow blue background */
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
