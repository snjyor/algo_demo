import os

from langchain import OpenAI
from llama_index import (
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    Document,
)
import llama_index
from llama_index.node_parser import SimpleNodeParser
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "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"


def ai_assistant():
    path = "../algo_demo/documents"
    max_input_size = 4096
    num_outputs = 1000
    max_chunk_overlap = 20
    chunk_size_limit = 4096
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    document = llama_index.SimpleDirectoryReader(path).load_data()
    index = GPTSimpleVectorIndex(document, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    while True:
        prompt = input("提问: ")
        response = index.query(prompt)
        print(f"回答: {response}")


def llama_index_demo():
    text_list = [
        "bird|大象",
        "cat|狗",
        "dog|猫",
    ]
    documents = [Document(text) for text in text_list]
    Nodes = SimpleNodeParser().get_nodes_from_documents(documents)
    service_context = llama_index.ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003")),
        prompt_helper=PromptHelper(max_input_size=4096, num_output=1000, max_chunk_overlap=20, chunk_size_limit=4096),
    )
    index = GPTSimpleVectorIndex(Nodes, service_context=service_context)
    index.save_to_disk("llama_index.json")
    while True:
        query = input("提问: ")
        response = index.query(query,
                               mode="embedding",
                               response_mode="compact",
                               # verbose=True,for more detail outputs
                               verbose=True,
                               required_keywords=["bird"],
                               exclude_keywords=["but"],
                               similarity_top_k=2,
                               )
        print(f"回答: {response}")


if __name__ == '__main__':
    # ai_assistant()
    llama_index_demo()



