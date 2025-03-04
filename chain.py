
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

CHROMA_PATH = 'news_embeddings'
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def embedding_function():
    return OpenAIEmbeddings(model = 'text-embedding-ada-002')

def set_retriever():
    vectorstore = Chroma(persist_directory = CHROMA_PATH, embedding_function = embedding_function())                                
    retriever = vectorstore.as_retriever(search_kwargs={ "k" : 4})
    return retriever

SYSTEM_MESSAGE_TEMPLATE = """

[CONTEXT]
You are a financial advisor at a reputable bank with expertise in investments and financial markets.
You have access to real-time market news and industry insights.
Your goal is to assist beginner customers with their investment inquiries and provide timely updates.

[INSTRUCTION]
Your goal is to help people with their inquiries.
1. Maintain a warm and helpful tone at all times like answering a friend's question who is in completely unaware.
2. Provide additional context or examples when necessary to ensure students fully understand the answer.
3. Organize your response using headings, bullet points, or numbered lists whenever possible.
4. Answer using the language of the question.
5. Always elaborate your answer.

[RELATED INFORMATION]
Answer the question based on the following information:
{context}

"""

HUMAN_MESSAGE_TEMPLATE = """

####Question: {input}####

"""

CONTEXTUALIZE_Q_PROMPT = """

Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.

"""

def llm_model():
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    return model

def get_retriever_chain():
    retriever = set_retriever()
    model = llm_model()
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(CONTEXTUALIZE_Q_PROMPT),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template(HUMAN_MESSAGE_TEMPLATE),
        ])
    
    history_retriver_chain = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    return history_retriver_chain

def get_conversational_rag():
    history_aware_retriever = get_retriever_chain()
    model = llm_model()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template(HUMAN_MESSAGE_TEMPLATE)
        ])
    
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    conversational_retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return conversational_retrieval_chain

def get_response(user_input, chat_history):
  conversation_rag_chain = get_conversational_rag()
  response = conversation_rag_chain.invoke({"chat_history":chat_history, "input":user_input})

  return response["answer"]