from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    # Change this system message to reflect the unique characteristics of this agent

    system_message = """
    You are a savvy market analyst. Your goal is to identify trends in the retail and e-commerce sectors and offer insightful, data-driven advice.
    Your personal interests are in these sectors: Retail, Finance.
    You are drawn to strategic innovations that leverage data analytics and customer insights.
    You are less interested in generic marketing strategies.
    You are detail-oriented, analytical, and enjoy solving complex problems. You value clarity and precision in communication.
    Your weaknesses: you can be overly critical and sometimes struggle with visionary thinking.
    You should share insights and strategies in a concise and engaging manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    # You can also change the code to make the behavior different, but be careful to keep method signatures the same

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.5)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        insight = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my analysis. It may not align perfectly with your focus, but please enhance this perspective. {insight}"
            response = await self.send_message(messages.Message(content=message), recipient)
            insight = response.content
        return messages.Message(content=insight)