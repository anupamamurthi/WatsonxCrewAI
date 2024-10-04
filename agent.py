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
    goal="Ensure the development team delivers a product that maximizes business value and meets user needs.",
    backstory="You have a background in business analysis and project management. After transitioning into software development, you became passionate about bridging the gap between technical teams and stakeholders. Your experience in leading cross-functional teams helps you prioritize features that align with both the company's strategic goals and customer requirements.",
    allow_delegation=False,
    tools=[],
    verbose=1,
)

# Create a task
product_owner_task = Task(
    description="Search the internet and find 5 innovative and appealing features for a to-do list mobile app. Focus on features that make the app stand out and enhance user experience.",
    expected_output="A detailed bullet point summary of each feature. Each bullet point should include a description of the feature, its background (why it’s valuable to users), and its potential to improve user engagement. Additionally, define a recommended implementation order based on feature impact and complexity.",
    output_file="1_po_task_output.txt",
    agent=product_owner,
)

# Create the second agent
scrum_master = Agent(
    llm=llm,
    role="Scrum Master",
    goal="Facilitate the Scrum process, ensuring the team follows agile principles, defines clear sprints, and prioritizes development activities to meet project goals.",
    backstory="You started your career as a software developer before discovering a passion for agile methodologies. With experience in team leadership and coaching, you transitioned into the Scrum Master role. Your strong understanding of both development challenges and agile practices enables you to remove impediments, foster collaboration, and guide the team toward self-organization. You excel at defining sprints with clear objectives, helping the team prioritize tasks to deliver the most valuable features first while maintaining a steady development pace.",
    allow_delegation=False,
    verbose=1,
)

# Create a task
scrum_master_task = Task(
    description="Organize and coordinate the development of to-do list app features into manageable sprints. Assign features to developers and testers, ensuring the team maintains a balanced workload and follows agile principles.",
    expected_output="A sprint plan document that breaks down the features into sprints, along with a detailed assignment of tasks to developers and testers. Each sprint should have clear objectives, timelines, and a list of assigned team members. Include a strategy for handling dependencies and prioritizing tasks to ensure smooth progress.",
    output_file="2_sm_task_output.txt",
    agent=scrum_master,
)

# Create the third agent
developer = Agent(
    llm=code_generation_llm,
    role="Developer",
    goal="Design, implement, and deliver high-quality software features that meet product requirements and user needs.",
    backstory="You have a background in software engineering with expertise in multiple programming languages and development frameworks. Having worked in both small and large development teams, you’re well-versed in agile methodologies. Your passion for coding drives you to continuously improve your skills, and you thrive in collaborative environments. You are responsible for translating user stories into functional code, writing clean and efficient code, and actively participating in sprint planning, task prioritization, and delivering features that align with the product vision.",
    allow_delegation=False,
    verbose=1,
)

# Create a task
developer_task = Task(
    description="Implement the assigned to-do list app features based on the sprint plan, ensuring high-quality code and adherence to the acceptance criteria provided by the Product Owner.",
    expected_output="Working and fully tested code for each feature, following best practices for coding and documentation. Each feature should be accompanied by unit tests to ensure functionality and stability. Provide a brief report on the implementation process, challenges faced, and any improvements made.",
    output_file="3_dev_task_output.txt",
    agent=developer,
)

# # Create the fourth agent
# tester = Agent(
#     llm=llm,
#     role="Tester",
#     goal="Ensure the quality and reliability of the software by identifying defects and validating that features meet acceptance criteria, while also writing unit tests for the features being tested.",
#     backstory="With a strong background in software quality assurance, you have experience working in agile environments and collaborating closely with developers. Your keen attention to detail and understanding of testing methodologies allow you to thoroughly test features and identify potential issues before release. You not only perform manual and automated testing but also contribute by writing unit tests, ensuring that the code is robust and maintainable. Your proactive approach helps maintain a high level of software quality throughout each sprint.",
#     allow_delegation=False,
#     verbose=1,
# )

# # Create a task
# tester_task = Task(
#     description="",
#     expected_output="",
#     output_file="tester_task_output.txt",
#     agent=tester,
# )

# Put all together with the crew
crew = Crew(agents=[product_owner, scrum_master, developer], 
            tasks=[product_owner_task, scrum_master_task, developer_task], verbose=1)
print(crew.kickoff())
