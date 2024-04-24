import dotenv

from langchain.globals import set_verbose, set_debug
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from chain import memoir_qa_chain, multiple_query_generate_chain, multiple_query_retriever_chain, llm_rerank_chain, \
    reformatting

dotenv.load_dotenv()
set_debug(True)
set_verbose(True)

texts = [
    Document(page_content="My name is sejong choi."),
    Document(page_content="Paris and Seoul are beautiful cities."),
    Document(page_content="The sky is clear today, perfect for an afternoon walk."),
    Document(page_content="Reading books is a good way to expand your knowledge."),
    Document(page_content="I often visit the local library to find new reading material."),
    Document(page_content="My favorite subject is computational linguistics."),
    Document(page_content="AI technology has evolved significantly over the past decade."),
    Document(page_content="Many companies are now investing in autonomous driving solutions."),
    Document(page_content="Quantum computing could revolutionize multiple industries."),
    Document(page_content="Blockchain technology provides a new layer of security."),
    Document(page_content="Healthy eating is crucial for maintaining physical fitness."),
    Document(page_content="Regular exercise can help reduce the risk of chronic diseases."),
    Document(page_content="The Python programming language is known for its versatility."),
    Document(page_content="JavaScript is essential for web development."),
    Document(page_content="Java remains a popular choice among enterprise applications."),
    Document(page_content="The rise of virtual reality has changed the gaming landscape."),
    Document(page_content="Augmented reality offers new ways to interact with digital content."),
    Document(page_content="Social media has become a powerful tool for marketing."),
    Document(page_content="E-commerce sales have grown exponentially in recent years."),
    Document(page_content="Cybersecurity is more important than ever."),
    Document(page_content="Data science involves complex statistical analyses."),
    Document(page_content="Machine learning models can predict customer behavior."),
    Document(page_content="Natural language processing helps computers understand human language."),
    Document(page_content="The Internet of Things connects everyday devices to the web."),
    Document(page_content="Cloud computing provides scalable IT infrastructure."),
    Document(page_content="Renewable energy sources are becoming more cost-effective."),
    Document(page_content="Global warming poses a serious threat to our environment."),
    Document(page_content="Conservation efforts are crucial for protecting biodiversity."),
    Document(page_content="Plastic pollution is a pressing environmental issue."),
    Document(page_content="Mental health is just as important as physical health."),
    Document(page_content="Telemedicine provides easier access to healthcare."),
    Document(page_content="The COVID-19 pandemic has impacted global economies."),
    Document(page_content="Vaccines are effective in preventing viral infections."),
    Document(page_content="Genetic research offers new insights into human diseases."),
    Document(page_content="Stem cell therapy shows promise for regenerative medicine."),
    Document(page_content="Public transportation is a sustainable travel option."),
    Document(page_content="Urban planning can significantly affect a city's liveability."),
    Document(page_content="Real estate markets fluctuate based on economic conditions."),
    Document(page_content="Investing in stocks requires careful analysis."),
    Document(page_content="Cryptocurrencies are volatile digital assets."),
    Document(page_content="Budgeting helps individuals manage their finances better."),
    Document(page_content="Learning multiple languages enhances cognitive abilities."),
    Document(page_content="Cultural exchange programs promote global understanding."),
    Document(page_content="Remote work has become a norm in many sectors."),
    Document(page_content="Artificial intelligence can automate routine tasks."),
    Document(page_content="Drones are used in various applications, from delivery to surveillance."),
    Document(page_content="Robotics engineering combines multiple technical disciplines."),
    Document(page_content="Innovation is key to staying competitive in tech industries."),
    Document(page_content="Learning history helps us understand our past."),
    Document(page_content="Cooking at home can be both fun and economical."),
    Document(page_content="Photography is a powerful form of artistic expression."),
    Document(page_content="Classical music has influenced many contemporary genres."),
    Document(page_content="Gardening is a rewarding hobby that also benefits the environment."),
    Document(page_content="Volunteering is a great way to give back to the community."),
    Document(page_content="Studying astronomy expands our understanding of the universe."),
    Document(page_content="Bird watching is a peaceful and educational activity."),
    Document(page_content="Practicing yoga can improve both mental and physical health."),
    Document(page_content="Playing chess develops strategic thinking and problem-solving skills."),
    Document(page_content="Cycling is an efficient and eco-friendly mode of transportation."),
    Document(page_content="Documentaries can provide deep insights into different cultures and issues."),
    Document(page_content="Learning to play a musical instrument enhances cognitive abilities."),
    Document(page_content="Creative writing fosters imagination and expressive skills."),
    Document(page_content="The study of philosophy can challenge and expand one's thinking."),
    Document(page_content="Adopting pets from shelters promotes animal welfare."),
    Document(page_content="The art of pottery requires patience and creativity."),
    Document(page_content="Traveling exposes one to new ideas and perspectives."),
    Document(page_content="The study of genetics is crucial for medical advances."),
    Document(page_content="Investing in education is vital for a country's development."),
    Document(page_content="The fashion industry significantly impacts global trends."),
    Document(page_content="Watching films can be both entertaining and educational."),
    Document(page_content="Building model kits is a popular hobby that teaches precision."),
    Document(page_content="Practicing martial arts can teach discipline and self-defense."),
    Document(page_content="Reading poetry can provide solace and inspiration."),
    Document(page_content="Skiing is a thrilling and physically demanding sport."),
    Document(page_content="Drawing and painting are accessible forms of artistic expression."),
    Document(page_content="The development of digital currencies is reshaping finance."),
    Document(page_content="Playing video games can improve hand-eye coordination."),
    Document(page_content="Studying foreign languages opens up new cultural horizons."),
    Document(page_content="The preservation of wildlife habitats is essential for biodiversity."),
    Document(page_content="Participating in team sports builds leadership and teamwork skills."),
    Document(page_content="Urban gardening can help cities become more sustainable."),
    Document(page_content="Making homemade crafts can be a therapeutic activity."),
    Document(page_content="The evolution of smartphones has changed how we communicate."),
    Document(page_content="Participating in science fairs can stimulate interest in STEM fields."),
    Document(page_content="The study of meteorology is essential for predicting weather."),
    Document(page_content="Exploring caves can be adventurous but requires proper safety measures."),
    Document(page_content="Architectural design influences how people experience spaces."),
    Document(page_content="Collecting stamps is a popular way to learn about world history."),
    Document(page_content="The role of NGOs is crucial in addressing global issues."),
    Document(page_content="The study of economics helps understand market dynamics."),
    Document(page_content="Home automation is becoming increasingly popular for convenience."),
    Document(page_content="The practice of mindfulness can reduce stress and anxiety."),
    Document(page_content="Jogging regularly can significantly improve cardiovascular health."),
    Document(page_content="Learning about art history can enhance appreciation for visual arts."),
    Document(page_content="Scuba diving allows exploration of underwater ecosystems."),
    Document(page_content="Maintaining a blog is a way to share knowledge and personal experiences."),
    Document(page_content="The development of AI is impacting various aspects of life."),
    Document(page_content="Understanding climate change is critical for future planning.")
]

db = Chroma().from_documents(texts, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

retriever_chain = (
        multiple_query_generate_chain(llm) |
        multiple_query_retriever_chain(db.as_retriever(search_type="similarity", search_kwargs={"k": 3})) |
        reformatting
)

query = "What is my name?"
docs = retriever_chain.invoke(input={"query": query})
contexts = llm_rerank_chain(llm).invoke(input={"documents": docs, "query": query}).content
resp = memoir_qa_chain(llm).invoke(input={"contexts": contexts, "query": query, "messages": ChatMessageHistory().messages})

print(resp)
