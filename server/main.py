from fastapi import FastAPI
import uvicorn

from settings import Config


config = Config()
app = FastAPI()


@app.get("/health")
async def health_check():
    return 
    # Simple inference example
   

@app.post("/completion")
async def completion(question):
    output = config.LLM.create_chat_completion(
         messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Feel free to answer any question the user may ask. Always answer in the langauge \
                of the user's question",
        },
        {"role": "user", "content": f"{question}"},
    ],
        max_tokens=1024,  # Generate up to 1024 tokens  # Whether to echo the prompt
    )
    output_msg = output["choices"][0]["message"]["content"]

    return {"message": output_msg}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)