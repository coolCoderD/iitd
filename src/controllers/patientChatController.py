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
  "top_p": 0.95,
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
            "This is a patient you are talking to, answer all upcoming prompts accordingly. With every call I will send you 5 things: 1. `Medical History`, this is from his/her medical history, 2. `Chat Context`, tht is the context of the chat that has happened yet, 3. `New prompt` i.e. the prompt that the patient just gave, 4. `Medical List`, this is the list of medicines the patient is taking, 5. `Doctor Notes`, this is the list of notes the doctor has written for the patient. Give the answers in short, precise yet easy normal language. Also no need to mention Related Text, Chat Context if not required for the New Prompt\n",
          ],
        },
      ]
    )
    data = request.get_json()
    prompt = data.get('prompt')
    relevant = getRelevant(data) # get embeddings 
    context = data.get('context')
    medicines = data.get('medicines')
    notes = data.get('notes')

    # Send the prompt to the chat session
    response = chat_session.send_message("Chat Context: " +context + "New prompt: " + prompt + "Medical History: "+relevant + "Medical List: "+medicines + "Doctor Notes: "+notes)

    newContext = chat_session.send_message("Make a new context from the sum of new response: "+response.text+" and the current context: "+context+" . Only reply with plain text only. Make sure to not miss important details.")
    
    return {
        "message": "Chat executed successfully",
        "response": response.text,
        "newContext": newContext.text,
    }, 201