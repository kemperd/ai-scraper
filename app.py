import streamlit as st
import streamlit_authenticator as stauth
from bs4 import BeautifulSoup
import logging
import requests
from urllib.parse import urljoin, urlparse
import glob
import os
from typing_extensions import List, TypedDict
import pandas as pd
import yaml
from yaml.loader import SafeLoader

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from langchain_openai import OpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = 'gpt-4.1-mini'

PANDAS_DATA_FILE = 'data.csv'

COMPANIES = {
    'Coca-Cola': 'https://www.coca-cola.com/gb/en',
    'Unilever': 'https://www.unilever.com',
}
    
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO)

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

class Crawler:
    def __init__(self, company=''):
        self.visited_urls = []
        self.urls_to_visit = [COMPANIES[company]]
        self.company = company
        self.root_url = COMPANIES[company]
        self.domain = urlparse(COMPANIES[company]).netloc
        self.pages = []

    def download_url(self, url, content_type):
        if content_type is not None:
            if content_type.startswith('text/html'):
                return requests.get(url).text
            else:
                return requests.get(url).content

    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(str(html), 'html.parser')
        #print('soup: ', soup)
        for link in soup.find_all('a'):
            #print('found link: ', link)
            path = link.get('href')
            if path and path.startswith('/'):
                path = urljoin(url, path)
            yield path

    def add_url_to_visit(self, url):
        if url not in self.visited_urls and url not in self.urls_to_visit:
            self.urls_to_visit.append(url)

    def crawl(self, url):
        logging.info(f'Crawling: {url}')
        content_type = requests.head(url).headers.get('content-type')

        #print('content_type: ', content_type.split(';')[0].strip() )
        # Only allow HTML and PDF
        if content_type.split(';')[0].strip() in ['text/html', 'application/pdf']:

            body = self.download_url(url, content_type)

            self.pages.append(url)

            for url in self.get_linked_urls(url, body):
                #print('process url: ', url)
                url_domain = urlparse(url).netloc
                # Only add URLs in site domain
                if url is not None:
                    #print('url_domain: ', url_domain)
                    #if ( url_domain == self.domain ) and 'mailto' not in url:
                    if ( url.startswith(self.root_url) and 'mailto' not in url ):
                        self.add_url_to_visit(url)   

    def get_pages(self):
        return self.pages

    def run(self):
        while self.urls_to_visit:
            url = self.urls_to_visit.pop(0)
            try:
                self.crawl(url)
            except Exception:
                logging.exception(f'Failed to crawl: {url}')
            finally:
                self.visited_urls.append(url)

def extract_website(company):
    # Crawl valid site URLs
    crawler = Crawler(company=company)
    crawler.run()    

    # Load URLs into vector database
    loader = UnstructuredURLLoader(urls=crawler.get_pages())

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    if client.collection_exists(collection_name=company):
        client.delete_collection(collection_name=company)

    client.create_collection(
        collection_name = company,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    qdrant = QdrantVectorStore(
        client=client,
        collection_name=company,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.DENSE,
    )
    qdrant.add_documents(documents=all_splits)

def run_llm(company, question):    
    llm = init_chat_model(MODEL_NAME, model_provider="openai")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=company,
        embedding=OpenAIEmbeddings(),
        retrieval_mode=RetrievalMode.DENSE,
    )

    # Define state for application
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"], k=5)
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = '\n\n'
        for doc in state['context']:
            docs_content += 'Source: ' + str(doc.metadata['source'] + '\n')
            docs_content += 'Contents: ' + doc.page_content + '\n'

        logging.info(docs_content)

        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    template = """Use the following context to answer the question at the end.
    If you don not know the answer, simply state you don't know and do not try to make up an answer.
    If a document is given in the context, fully use this to answer the question.
    Also provide the title and if known the file name of the document where you found the answer.
    If there are headers in the text which can be used to answer a question with "Yes", also use these.
    Start the answer with "Yes" or "No", followed by the summary of what you found.

    {context}

    Question: {question}

    Helpful answer:"""

    prompt = PromptTemplate.from_template(template)

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph.invoke({"question": question})

def main_screen():
    st.set_page_config(layout='wide')
    #st.title('CSO deelnemersanalyse')

    tab_extract_site, tab_analyze_site, tab_reporting = st.tabs(['Extract website', 'Run analysis', 'Reporting'])

    if os.path.exists(PANDAS_DATA_FILE):
        df = pd.read_csv(PANDAS_DATA_FILE)
    else:
        df = pd.DataFrame(columns=['Company', 'Question', 'Answer'])

    company = st.sidebar.selectbox('Company', COMPANIES.keys())

    with tab_extract_site:
        st.write('Select company website to extract on the left')

        if st.button('Extract website'):
            print('Extract website ', COMPANIES[company])
            with st.spinner('Extracting...'):
                extract_website(company)
                st.success('Done!')

    with tab_analyze_site:
        cb_privacy = st.checkbox('Privacy statement', True)
        cb_accesibility = st.checkbox('Accessibility statement', True)

        topic_list = []
        if cb_privacy: topic_list.append('Privacy statement')
        if cb_accesibility: topic_list.append('Accessibility statement')

        if st.button('Analyze site'):
            if client.collection_exists(collection_name=company):
                table_data = []
                full_text = ''
                with st.status('Analyzing with AI....') as status:
                    for topic in topic_list:
                        question = topic
                        st.write(question)
                        answer = run_llm(company, question)
                        table_data.append([company, topic, question, answer['answer']])

                df_local = pd.DataFrame(data=table_data, columns=['Company', 'Topic', 'Question', 'Answer'])
                df.drop(df[df['Company'] == company].index, inplace=True)
                df = pd.concat([df, df_local])

                df.to_csv(PANDAS_DATA_FILE, index=False)
                st.dataframe(df_local)

                for index, row in df_local.iterrows():
                    st.header(row['Topic'], divider='grey')
                    st.markdown('**Is de following information present: ' + row['Question'] + '?**')
                    st.write(row['Answer'])
            else:
                st.error(f'No website data found for {company}, please extract website first')

    with tab_reporting:
        if os.path.exists(PANDAS_DATA_FILE):
            df = pd.read_csv(PANDAS_DATA_FILE)
            st.dataframe(df)

def main():
    main_screen()

if __name__ == '__main__':
    main()