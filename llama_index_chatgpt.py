import os
from llama_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper
import llama_index
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


if __name__ == '__main__':
    ai_assistant()


