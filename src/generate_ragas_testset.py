from dotenv import load_dotenv

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_cohere import ChatCohere
from langchain_cohere.llms import Cohere
from langchain_cohere import CohereEmbeddings

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

loader = DirectoryLoader("./data/papers", glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()

# generator with openai models
generator_llm = Cohere()
critic_llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
embeddings = CohereEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=15, distributions={simple: 0.2, reasoning: 0.4, multi_context: 0.4}, raise_exceptions=False)

# save testset as pandas dataframe
testset.to_pandas().to_csv("./data/papers/papers.csv", index=False)
testset.to_pandas().to_pickle("./data/papers/papers.pkl")

print(testset.to_pandas())