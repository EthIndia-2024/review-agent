# from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from checkreviewscore import CheckReviewHelpfulnessInput
from checkreviewscore import calculate_review_helpfulness
from cdp_langchain.tools import CdpTool
import os
from dotenv import load_dotenv
from incentivecalculation import CalculateIncentiveInput
from incentivecalculation import calculate_incentive

wallet_data_file = "wallet_data.txt"



# Load environment variables from .env file
load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")

# I have review as my input which is coming from the user are "user_wallet_address, user_review, service_id"
# In my workflow I just want to call the review summarizer, send its output to the service
# Run a review scorer using some chaining of agents to get a score and then calculate the amount of incentive to be given to the user
# Store the incentives, user_wallet_id, service_id in the database
# Send the batched transactions after they have crossed some certain threshold



def initialize_agent(state_modifier):
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatXAI( xai_api_key=XAI_API_KEY, model="grok-beta")

    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        # state_modifier=(
        #     "You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit. "
        #     "You are empowered to interact onchain using your tools. If you ever need funds, you can request "
        #     "them from the faucet if you are on network ID 'base-sepolia'. If not, you can provide your wallet "
        #     "details and request funds from the user. Before executing your first action, get the wallet details "
        #     "to see what network you're on. If there is a 5XX (internal) HTTP error code, ask the user to try "
        #     "again later. If someone asks you to do something you can't do with your currently available tools, "
        #     "you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit, "
        #     "recommend they go to docs.cdp.coinbase.com for more information. Be concise and helpful with your "
        #     "responses. Refrain from restating your tools' descriptions unless it is explicitly requested."
        # ),
        state_modifier=state_modifier,

    ), config


def initialize_scorer_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatXAI( xai_api_key=XAI_API_KEY, model="grok-beta")

    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()
    
    HELPFULNESS_PROMPT = """
    This tool evaluates how helpful a review is to a company by analyzing its descriptiveness, sentiment, actionability, uniqueness, specificity, and length adequacy.
    """
    
    CALCULATE_INCENTIVE_PROMPT = """
    This tool calculates the incentive to pay to a reviewer based on their review score. 
    The score ranges between 1 and 100, and the incentive is a value between 10^-6 and 10^-4.
    """
    
    calculateIncentiveTool = CdpTool(
    name="calculate_incentive",
    description=CALCULATE_INCENTIVE_PROMPT,
    cdp_agentkit_wrapper=agentkit,
    args_schema=CalculateIncentiveInput,
    func=calculate_incentive,
    )
    
    checkReviewHelpfulnessTool = CdpTool(
    name="check_review_helpfulness",
    description=HELPFULNESS_PROMPT,
    cdp_agentkit_wrapper=agentkit,
    args_schema=CheckReviewHelpfulnessInput,
    func=calculate_review_helpfulness,
    )
    
    tools.append(checkReviewHelpfulnessTool)
    tools.append(calculateIncentiveTool)

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=(
            "Your job is to score the given review and calculate the incentive to be given to the user. "
            "You have the access to the tool to get the different scores of the review submitted by the user and final score also"
            "You have the access to the tool to calculate the incentive to be given to the user based on the score of the review"
            "And your job is to calculate the incentive that is to be given to the reviewer based on how much good (how better score) he has given"
            # "Then finally store the tuple of (customer_wallet_id, service_id, incentive) in the database"
        ),

    ), config




# Future scope in this function is to give it to model iteratively again and again
def review_summarizer(user_review, service_template):
    """Summarize the user review."""
    print("Starting Review Summarizer...")
    state_modifier = ("You are a helpful agent that can summarize reviews to match a service template. "
                      "You are empowered to summarize reviews to match the service template. "
                      "Your sole job is to keep the review as informative as possible for the company to improve themselves."
                      "You should remove any abusive language or irrelevant information from the review.")
    agent_executor, config = initialize_agent(state_modifier=state_modifier)
    
    
    summarized_review = ""
    for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content='User review is: '+user_review+'\n And given service template is: '+service_template)]}, config
            ):
        summarized_review += chunk["agent"]["messages"][0].content
    return summarized_review


def review_score_and_transaction_log(user_review, service_id, customer_wallet_id):
    """Give a certain score to a review the summarized review. And then calculate the incentive which is to be given to the user.
    Store the incentive, user_wallet_id, service_id in the database."""
    print("Starting Review Scorer...")
    agent_executor, config = initialize_scorer_agent()
    scorer_output = ""
    for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_review)]}, config
            ):
        if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                    scorer_output += chunk["agent"]["messages"][0].content
        elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                    scorer_output += chunk["tools"]["messages"][0].content
        print("-------------------")
        
    return scorer_output
    

# s = review_summarizer("I love the service, it was amazing", "Critique: ..., Praise: ...")
# print(s)

scorer_output = review_score_and_transaction_log("I love the service, it was amazing", "service_id", "customer_wallet_id")
print('*********************************')
print(scorer_output)


# def main():
#     """Start the chatbot agent."""
#     agent_executor, config = initialize_agent()

#     mode = choose_mode()
#     if mode == "chat":
#         run_chat_mode(agent_executor=agent_executor, config=config)
#     elif mode == "auto":
#         run_autonomous_mode(agent_executor=agent_executor, config=config)


# if __name__ == "__main__":
#     print("Starting Agent...")
#     main()