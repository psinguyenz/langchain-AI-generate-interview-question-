from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from src.prompt import *
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# OpenAI authentication 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def file_processing(file_path):
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content
    
    splitter_ques_gen = TokenTextSplitter(
        model_name = 'gpt-4.1-mini',
        chunk_size = 10000,
        chunk_overlap = 200
    )

    chunk_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunk_ques_gen]

    splitter_ans_gen = TokenTextSplitter(
        model_name = 'gpt-4.1-mini',
        chunk_size = 256,
        chunk_overlap = 25
    )

    document_ans_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_ans_gen

def llm_pipeline(file_path):
    document_ques_gen, document_ans_gen = file_processing(file_path)

    llm_ques_gen_pipeline = ChatOpenAI(
        temperature = 0.3,
        model = "gpt-4.1-mini"
    )

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=['text'])

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    initial_chain = (
        {"text": RunnablePassthrough()} 
        | PROMPT_QUESTIONS 
        | llm_ques_gen_pipeline 
        | StrOutputParser()
    )

    ques_gen_chain = (
        {
            "existing_answer": initial_chain, 
            "text": RunnablePassthrough()
        }
        | REFINE_PROMPT_QUESTIONS
        | llm_ques_gen_pipeline
        | StrOutputParser()
    )

    ques = ques_gen_chain.invoke(document_ques_gen)

    embeddings = OpenAIEmbeddings()

    vector_stores = FAISS.from_documents(document_ans_gen, embeddings)

    llm_answer_gen = ChatOpenAI(temperature=0.1, model="gpt-4.1-mini")

    ques_list = ques.split("\n")

    filtered_questions = [line for line in ques_list if line.strip() and line.strip()[0].isdigit()]

    retriever = vector_stores.as_retriever()

    return retriever, filtered_questions 