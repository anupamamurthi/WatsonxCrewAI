from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_ibm import WatsonxLLM
import os

api_key = os.environ.get("WATSONX_API_KEY")
project_id = os.environ.get("PRJ_ID")
url = os.environ.get("WATSONX_URL")

# Parameters
parameters = {"decoding_method": "greedy", "max_new_tokens": 500}

# Create the first LLM
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-70b-instruct",
    url=url,
    params=parameters,
    project_id=project_id,
    apikey=api_key,
)

# Create the function calling llm
code_generation_llm = WatsonxLLM(
    model_id="ibm/granite-20b-code-instruct",
    url=url,
    params=parameters,
    project_id=project_id,
    apikey=api_key,
)

# Tools
#search = SerperDevTool()

# Create the agent
product_owner = Agent(
    llm=llm,
    code_generation_llm=code_generation_llm,
    role="Product Owner",
    goal="Market analysis and feature prioritization",
    backstory="",
    allow_delegation=False,
    tools=[],
    verbose=1,
)

# Create a task
product_owner_task = Task(
    description="",
    expected_output="",
    output_file="product_owner_output.txt",
    agent=product_owner,
)

# Create the second agent
scrum_master = Agent(
    llm=llm,
    role="Scrum Master",
    goal="Organize sprints ...",
    backstory="",
    allow_delegation=False,
    verbose=1,
)

# Create a task
scrum_master_task = Task(
    description="",
    expected_output="",
    output_file="scrum_master_output.txt",
    agent=scrum_master,
)

# Create the third agent
developer = Agent(
    llm=llm,
    role="Developer",
    goal="",
    backstory="",
    allow_delegation=False,
    verbose=1,
)

# Create a task
developer_task = Task(
    description="",
    expected_output="",
    output_file="developer_output.txt",
    agent=developer,
)

# Create the fourth agent
tester = Agent(
    llm=llm,
    role="Tester",
    goal="",
    backstory="",
    allow_delegation=False,
    verbose=1,
)

# Create a task
tester_task = Task(
    description="",
    expected_output="",
    output_file="tester_output.txt",
    agent=tester,
)

# Put all together with the crew
crew = Crew(agents=[product_owner, scrum_master, developer, tester], 
            tasks=[product_owner_task, scrum_master_task, developer_task, tester_task], verbose=1)
print(crew.kickoff())
