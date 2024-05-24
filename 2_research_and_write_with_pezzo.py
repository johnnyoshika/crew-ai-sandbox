import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from crewai import Agent, Task, Crew

# Load environment variables from .env file
load_dotenv()

# Proxy through Pezzo, by using the "Connect to any LLM" technique: https://docs.crewai.com/how-to/LLM-Connections/
# This seems to be working, as commenting out X-Pezzo-Api-Key or X-Pezzo-Project-Id results in an error,
# but for some reason the requests don't appear in Pezzo's Console.
openai = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    openai_api_base="https://proxy.pezzo.ai/openai/v1",
    default_headers={
        "X-Pezzo-Api-Key": os.getenv('PEZZO_API_KEY'),
        "X-Pezzo-Project-Id": os.getenv('PEZZO_PROJECT_ID'),
        "X-Pezzo-Environment": "Production"
    }
)

os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'

planner = Agent(
    llm=openai,
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You're working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    llm=openai,
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    llm=openai,
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    allow_delegation=False,
    verbose=True
)

plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
        "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
        "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
        "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
    "with an outline, audience analysis, "
    "SEO keywords, and resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
        "blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named "
        "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
        "engaging introduction, insightful body, "
        "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
        "alignment with the brand's voice.\n"
    ),
    expected_output="A well-written blog post "
    "in markdown format, ready for publication, "
    "each section should have 2 or 3 paragraphs.",
    agent=writer,
)

edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.",
    agent=editor
)

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2
)

result = crew.kickoff(
    inputs={"topic": "How Efficient is the Financial Market?"})

print(result)
