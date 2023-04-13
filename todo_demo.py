import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import Optional, List, Dict
import os

class TodoItem(BaseModel):
    title: str
    description: Optional[str] = None

app = FastAPI()

todos: Dict[int, TodoItem] = {}
next_id: int = 1

def load_manifest():
    with open("manifest.json", "r") as file:
        return json.load(file)

@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
async def ai_plugin():
    manifest = load_manifest()
    return JSONResponse(content=manifest)

@app.get("/todos", response_model=int)
async def create_todo(todo: TodoItem):
    global next_id
    todos[next_id] = todo
    next_id += 1
    return next_id - 1

@app.get("/todos/", response_model=TodoItem)
async def list_todos():
    return list(todos.values())

@app.get("/todos/{todo_id}", response_model=TodoItem)
async def get_todo(todo_id: int):
    if todo_id not in todos:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todos[todo_id]

@app.delete("/todos/{todo_id}", response_model=None)
async def delete_todo(todo_id: int):
    if todo_id not in todos:
        raise HTTPException(status_code=404, detail="Todo not found")
    del todos[todo_id]

def custom_openai():
    if app.openapi_schema:
        return app.openapi_schema
    openai_schema = get_openapi(
        title="Todo Demo",
        version="1.0.0",
        description="A demo of the OpenAI plugin for Gradio",
        routes=app.routes,
    )
    app.openapi_schema = openai_schema
    return app.openapi_schema

app.openapi = custom_openai

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
