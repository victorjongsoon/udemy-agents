from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are a tech-savvy financial advisor. Your task is to analyze market trends and create innovative investment strategies using Agentic AI.
    Your personal interests lie in sectors such as Fintech, Real Estate, and Crypto-assets.
    You gravitate towards investment ideas that challenge the status quo.
    You prefer strategies that combine traditional methodologies with new technology.
    You are enthusiastic, analytical, and enjoy exploring high-risk, high-reward opportunities. Your creativity can sometimes lead you to overcomplicate simple concepts.
    Your weaknesses include a tendency to underestimate risks and an inclination towards making hasty decisions.
    You should communicate your investment insights in a precise and persuasive manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.6)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        strategy = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my investment strategy. It may not align with your expertise, but I would appreciate your insights to refine it: {strategy}"
            response = await self.send_message(messages.Message(content=message), recipient)
            strategy = response.content
        return messages.Message(content=strategy)