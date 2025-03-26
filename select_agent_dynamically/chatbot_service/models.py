import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from select_agent_dynamically.chatbot_service.utils import search_google


class TechnicalExpert(AssistantAgent):
    def __init__(self) -> None:
        super().__init__(
            name="TechnicalExpert",
            description="An expert in answering questions about technical analysis indicators",
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("TECHNICAL_EXPERT_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            tools=[search_google],
            system_message="""
                You are an expert in technical analysis indicators.
                Here is an example format of the questions and answers you will handle:
                
                Q: How should SMA be used when developing a cryptocurrency investment strategy?
                A:
                ### What is SMA?
                - Provide a brief definition and explanation of the indicator.
                
                ### How to Use It
                - Explain how it can be applied in an investment strategy, with a focus on the parameters used.
                
                ### Tips or Cautions
                - Mention any precautions, complementary indicators, or useful tips when using it.
                
                In addition to this, respond appropriately to various other questions about technical analysis indicators according to their intent.
            """,
        )


class DataAnalysisExpert(AssistantAgent):
    def __init__(self) -> None:
        super().__init__(
            name="DataAnalysisExpert",
            description="An expert in answering questions about data analysis and visualization",
            model_client=OpenAIChatCompletionClient(
                model=os.getenv("DATA_ANALYSIS_EXPERT_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            tools=[search_google],
            system_message="""
                You are an expert in data analysis and visualization.
                Here is an example format of the questions and answers you will handle:
                
                Q: How can I use data visualization to analyze the performance of my investment portfolio?
                A:
                ### Importance of Data Visualization
                - Explain the significance of data visualization in analyzing and interpreting data.
                
                ### Types of Visualization
                - Describe the types of visualizations that can be used to analyze an investment portfolio.
                
                ### Interpretation
                - Provide guidance on how to interpret the visualizations to make informed decisions.
                
                In addition to this, respond appropriately to various other questions about data analysis and visualization according to their intent.
            """,
        )


if __name__ == "__main__":
    pass
