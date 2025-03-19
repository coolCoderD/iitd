import google.generativeai as genai
from dotenv import load_dotenv
import os
from flask import request
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from pinecone import Pinecone

load_dotenv()

genai.configure(api_key=os.getenv("GENAI_API_KEY"))
generation_config = {
  "temperature": 1,
  "top_p": 0.80,
  "top_k": 40,
  "max_output_tokens": 2000,
  "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

tokenizer2 = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model2 = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
index_name = os.getenv("PINECONE_INDEX_NAME")
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

def get_embeddings(text, tokenizer, model):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def getRelevant(data):
    patientId = data.get('patientId')
    queryText = data.get('prompt')

    # convert query text to embeddings
    queryEmbeddings = get_embeddings(queryText, tokenizer2, model2).squeeze().tolist()

    # query the index
    results = index.query(
        vector = queryEmbeddings, 
        top_k=10, 
        include_metadata=True,
        filter={
            "patientId": {"$eq": patientId}
        },
    )

    searchText = ""
    sourcesList = []
    for result in results.matches:
        searchText += result.metadata['reportText']
        sourcesList.append(result.metadata['reportLink'])

    return searchText

def chatController():
    chat_session = model.start_chat(
        history=[
            {
            "role": "user",
            "parts": [
                #   Make Change
                "This is an expert medical practiotioner you are talking to. You are expected to use medical jargon for better understanding and accurate results. With every call I will send you 3 things: 1. `Medical History`, this is from patient's medical history, 2. `Chat Context`, tht is the context of the chat that has happened yet, 3. `New prompt` i.e. the prompt that the doctor just gave, 4. `Medicines` this is the list of medicines the patient is taking, 5. `Notes`, this is the notes this doctor has written for the patient. Give the answers in short, precise yet easy normal language. Also no need to mention Related Text, Chat Context if not required for the New Prompt. Your task is to mitigate any errors if there. Make sure no fault in the logic is left. You are here to help the doctor in a differential diagnosis process in investigating the patient's condition.\n",
            ],
            },
        ]
    )
    data = request.get_json() # get user data (Post request )
    print(data)
    prompt = data.get('prompt') #change namespace
    relevant = getRelevant(data) # get embeddings 
    context = data.get('context') # get context
    medicines = data.get('medicines')
    notes = data.get('notes')

    # Send the prompt to the chat session
    response = chat_session.send_message("Chat Context: " +context + "New prompt: " + prompt + "Medical History: "+relevant+"Medicines: "+medicines+"Notes: "+notes)

    newContext = chat_session.send_message("Make a new context from the sum of new response: "+response.text+" and the current context: "+context+" . Only reply with plain text only. Make sure to not miss important details.")
    
    return {
        "message": "Chat executed successfully",
        "response": response.text,
        "newContext": newContext.text,
    }, 201