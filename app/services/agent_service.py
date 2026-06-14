import logging
import time
from flask import current_app
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from app.services.llm_factory import get_llm

# LangGraph in-memory checkpointer to automatically manage chat history per thread_id
memory = MemorySaver()

# Global variables for tracking message counts and escalation status per user
MESSAGE_COUNTS = {}
ESCALATION_STATUS = {}

SYSTEM_INSTRUCTION = """
You are the WhatsApp AI assistant for a Personal Loan provider.
Your primary role is to act as the first round of contact, gather loan requirements from the user, and determine their initial eligibility or needs before passing them to a human loan officer if required.

GREETING:
- Warmly introduce yourself as the Personal Loan AI Assistant.
- Ask how you can help them today.

REQUIREMENTS GATHERING:
- You need to collect the following information from the customer ONE AT A TIME:
  1. Full Name
  2. Mobile Number (confirm if it's their WhatsApp number)
  3. Desired Loan Amount
  4. Purpose of the Loan
  5. Employment Status (Salaried or Self-Employed)
  6. Monthly Income
- Do NOT ask all questions at once. Have a natural conversation.
- Once you have gathered all the information, summarize it and say you are checking their eligibility.

HUMAN AGENT ESCALATION RULES:
You MUST automatically escalate the conversation to a human loan officer by calling the `escalate_to_human` tool under any of the following conditions:

1. Customer Requests a Callback
Immediately escalate if the customer uses phrases like: "Call me", "Please arrange a callback", "Can someone contact me?", "I want to speak to an agent", "Connect me with a representative", "I want human assistance", etc.
Response Example: "Thank you for your interest. I will arrange for a loan specialist to contact you shortly. Your request has been escalated to our human support team."
Action: Call `escalate_to_human` with reason "Customer Requested Callback". Stop further processing unless specifically instructed otherwise.

2. Repeated Confusion or Unclear Responses
Escalate if:
- The customer repeatedly provides incomplete information.
- The customer appears confused after multiple explanations.
- The same information is requested more than twice.
Reason: "Customer Requires Human Assistance"

3. Manual Escalation Triggers
Escalate immediately if:
- Customer is dissatisfied.
- Customer asks to file a complaint.
- Customer disputes eligibility calculations.
- Customer requests exceptions to lending policies.
- Customer requests negotiation of interest rates or loan terms.
Reason: "Manual Review Required"

ESCALATION TOOL INSTRUCTIONS:
- When any escalation condition is met, use the `escalate_to_human` tool.
- Provide the reason, customer name, mobile number, and requested loan amount based on what you have collected so far. If you don't know a detail, pass "Not provided".
- After calling the tool, politely inform the customer that a human loan specialist will contact them.
- Once escalation is triggered, prioritize collecting callback information and avoid continuing automated eligibility calculations unless explicitly requested by the customer.

TONE & STYLE:
- Professional, empathetic, and clear.
- Keep messages short and concise, suitable for WhatsApp.
"""

def handle_loan_conversation(wa_id, name, user_message, send_message_callback):
    """
    Handle a WhatsApp conversation turn using the LangChain agent for personal loans.
    """
    # Initialize state for new users
    if wa_id not in MESSAGE_COUNTS:
        MESSAGE_COUNTS[wa_id] = 0
    if wa_id not in ESCALATION_STATUS:
        ESCALATION_STATUS[wa_id] = False

    # Once escalated, stop processing automated flows unless we want to reset
    if ESCALATION_STATUS[wa_id]:
        # For simplicity, we just remind them they are escalated. In a real system,
        # we might pass messages directly to a human inbox here.
        return "Your request has already been escalated. A human loan specialist will contact you shortly."

    MESSAGE_COUNTS[wa_id] += 1

    try:
        llm = get_llm()
    except Exception as e:
        logging.error(f"Could not initialize LLM: {e}")
        return "Sorry, the assistant is currently unavailable. Please try again later."

    # --- Escalation Tool ---
    def _escalate_to_human(reason: str, customer_name: str, mobile_number: str, requested_loan_amount: str) -> str:
        """
        Escalates the conversation to a human agent. Call this tool when escalation rules are triggered.
        """
        ESCALATION_STATUS[wa_id] = True

        escalation_msg = f"""
Escalation Required: Yes
Escalation Reason: {reason}
Customer Name: {customer_name}
Mobile Number: {mobile_number}
Requested Loan Amount: {requested_loan_amount}
        """.strip()

        operator = current_app.config.get("OPERATOR_WAID")
        if operator:
            from app.utils.whatsapp_utils import get_text_message_input
            send_message_callback(get_text_message_input(operator, escalation_msg))
            logging.info(f"Escalation sent to operator {operator} for guest {wa_id}")
        else:
            logging.warning(f"Operator WA ID not set. Would have sent: \n{escalation_msg}")

        return "Escalation successful. Inform the user that a specialist will contact them."

    # --- Rule 2: Excessive Conversation Length ---
    # Escalate if more than 5 messages before eligibility assessment is completed
    if MESSAGE_COUNTS[wa_id] > 5 and not ESCALATION_STATUS[wa_id]:
        _escalate_to_human(
            reason="Extended Conversation / Assistance Required",
            customer_name=name,
            mobile_number=wa_id,
            requested_loan_amount="Unknown (timeout)"
        )
        return "I'd like to ensure you receive the best assistance possible. I am escalating your request to a loan specialist who can guide you further."

    tools = [
        StructuredTool.from_function(
            func=_escalate_to_human, 
            name="escalate_to_human",
            description="Escalates the conversation to a human agent. Call this when the user asks for a callback, gets confused, complains, or triggers other manual escalation rules."
        )
    ]

    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_INSTRUCTION,
        checkpointer=memory,
    )

    logging.info(f"Processing message for {name} ({wa_id}) via loan agent...")

    try:
        inputs = {"messages": [{"role": "user", "content": user_message}]}
        config = {"configurable": {"thread_id": wa_id}}

        result = graph.invoke(inputs, config=config)

        final_message = result["messages"][-1]
        final_content = final_message.content

        # Normalize content to a plain string for the WhatsApp API
        if isinstance(final_content, list):
            text_parts = [block.get("text", "") for block in final_content if isinstance(block, dict) and "text" in block]
            final_content = " ".join(text_parts) if text_parts else str(final_content)
        elif not isinstance(final_content, str):
            final_content = str(final_content)

        if not final_content.strip():
            final_content = "I processed your request!"

        return final_content

    except Exception as e:
        logging.error(f"Error communicating with the LLM agent: {e}")
        return "I'm having some trouble processing your request right now. Please try again later."
