from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ctransformers import AutoModelForCausalLM

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
llm = AutoModelForCausalLM.from_pretrained('TheBloke/Llama-2-7B-Chat-GGML', model_file='llama-2-7b-chat.ggmlv3.q4_K_S.bin')

class InputText(BaseModel):
    text: str

@app.post("/get_joke/")
def get_joke(input_text: InputText):
    text = input_text.text
    response = ""
    for word in llm(text, stream=True):
        response += word
    return {"joke": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
