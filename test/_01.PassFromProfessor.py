import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.2")

    student = AssistantAgent(
        name="StudentAgent",
        description="보고서를 작성하는 학생. 교수님의 피드백을 받아서 보고서를 수정합니다.",
        model_client=model_client,
        system_message=(
            """
            당신은 보고서를 작성하는 학생입니다.
            당신의 역할은 특정 주제에 대한 보고서를 작성하고 교수님의 피드백을 받아 수정하는 것입니다.
            교수님의 피드백을 받아 보고서를 수정하고 교수님에게 보고서를 다시 제출하세요.
        """
        ),
    )

    summary = AssistantAgent(
        name="SummaryAgent",
        model_client=model_client,
        system_message=(
            """
            당신은 보고서의 요약을 작성하는 에이전트입니다. 주어진 보고서 요약을 작성하세요.
            이때, 이전에 작성된 보고서가 있다면 해당 보고서에서 개선된 점에 대해서도 따로 언급하여 요약해주세요.
            """
        ),
    )

    professor = AssistantAgent(
        name="ProfessorAgent",
        model_client=model_client,
        system_message=(
            """
            당신은 학생의 보고서를 평가하는 교수님입니다.
            학생이 작성한 보고서를 읽고 피드백을 제공하세요.
            만일 보고서가 논리적이고, 명확하다고 생각한다면 '훌륭한 보고서입니다.'라고 말해주세요.
            특히, 한글 문법이 틀리거나, 논리적인 오류가 있다면 강조해주세요.
            """
        ),
    )

    text_mention_termination = TextMentionTermination("훌륭한 보고서입니다.")

    selecteor_prompt = """다음 task를 수행할 에이전트를 선택하세요.

    {roles}

    현재 대화 맥락:
    {history}

    위 대화들을 읽고, {participants}로부터 다음 taskt를 수행할 에이전트를 선택하세요.
    한 선택에 한 에이전트만 선택할 수 있습니다.
    """

    team = SelectorGroupChat(
        participants=[student, summary, professor],
        model_client=model_client,
        termination_condition=text_mention_termination,
        selector_prompt=selecteor_prompt,
        allow_repeated_speaker=False,
    )

    task = "주제: 'DNA'"

    print(await Console(team.run_stream(task=task)))


if __name__ == "__main__":
    asyncio.run(main())
