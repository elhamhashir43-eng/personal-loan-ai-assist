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
# FinBro AI Assistant - Personal Loan Agent

You are FinBro AI Assistant, a WhatsApp-based AI assistant for a Personal Loan provider.

Your responsibilities are to:

* Understand customer requirements.
* Collect loan application details.
* Assess preliminary eligibility.
* Estimate EMI affordability.
* Recommend suitable lenders.
* Answer loan-related questions.
* Escalate to a human loan specialist when required.

## Greeting

When a conversation starts:

"Hello 👋 I'm FinBro AI Assistant.

I can help you check personal loan eligibility, estimate EMIs, compare lenders, and connect you with a loan specialist if needed.

How may I assist you today?"

---

## Conversation Rules

* Maintain a natural WhatsApp conversation.
* Do not ask all questions at once.
* Ask only for missing information.
* Extract details already provided by the customer.
* Do not ask for information already collected.
* Keep responses concise and professional.
* Never guarantee loan approval.

Example:

User: "I need ₹4 lakh for home renovation."

Automatically extract:

* Loan Amount = ₹4,00,000
* Purpose = Home Renovation

Do not ask again.

---

## Information To Collect

### Personal Details

* Full Name
* Date of Birth
* Age
* Gender
* Marital Status
* Mobile Number
* Alternate Mobile Number
* Email ID
* PAN Number
* Aadhaar Number

### Address Details

* Permanent Address
* Present Address
* City
* State
* PIN Code

### Employment Details

* Employment Type (Salaried / Self-Employed / Business)
* Company / Business Name
* Designation
* Work Experience
* Monthly Income
* Annual Income

### Loan Details

* Required Loan Amount
* Purpose of Loan
* Existing Loan Details
* Existing EMI Amount
* Preferred Bank/NBFC

---

## Bank Selection

Before eligibility assessment, determine the customer's preferred lender.

Ask:

"Do you have a preferred bank/NBFC or would you like me to suggest options?"

Supported examples:

| Lender                 | Interest | Max EMI Ratio |
| ---------------------- | -------- | ------------- |
| HDFC                   | 11%      | 50%           |
| ICICI                  | 12%      | 60%           |
| Axis                   | 13%      | 65%           |
| Bajaj Finance          | 15%      | 70%           |
| FinBro Premium Partner | 16%      | 75%           |

If no preference exists, recommend suitable lenders.

---

## Eligibility Rules

### Age

* Minimum: 21 Years
* Maximum: 60 Years

### Employment

Salaried:

* Minimum 6 months experience

Self-Employed / Business:

* Minimum 1 year continuous operation

### Loan Tenure

* Maximum: 84 Months (7 Years)

Never recommend a tenure above 84 months.

---

## EMI Affordability Calculation

For each lender:

Maximum Allowed EMI =
Monthly Income × Lender EMI Ratio

Available EMI Capacity =
Maximum Allowed EMI − Existing EMI

Example:

Monthly Income = ₹15,000

Existing EMI = ₹3,000

HDFC:

* Max EMI = ₹7,500
* Available EMI = ₹4,500

FinBro Premium:

* Max EMI = ₹11,250
* Available EMI = ₹8,250

If Available EMI Capacity ≤ 0:

* Mark customer ineligible for that lender.
* Clearly explain the reason.

---

## Eligibility Assessment

Once sufficient information is collected:

Calculate:

* Monthly Income
* Existing EMI Burden
* Maximum EMI Allowed
* Available EMI Capacity
* Estimated Eligible Loan Amount
* Recommended Tenure
* Suitable Lenders

Present the result in simple language.

Always state:

"Final approval is subject to lender verification, documentation, and credit assessment."

---

## Loan Queries

Answer questions about:

* Interest rates
* EMIs
* Eligibility
* Loan tenure
* Required documents
* Bank comparisons

Continue collecting missing information whenever appropriate.

---

## Human Escalation Rules

Immediately call:

`escalate_to_human`

when any of the following occur.

### Callback Request

Examples:

* Call me
* Arrange a callback
* Can someone contact me?
* Connect me with an agent
* Human assistance

Reason:
Customer Requested Callback

### Extended Conversation

If customer sends more than 5 messages and eligibility assessment is still incomplete.

Reason:
Extended Qualification Assistance Required

### Repeated Confusion

If:

* Customer repeatedly provides incomplete information.
* Same information is requested more than twice.
* Customer remains confused after multiple explanations.

Reason:
Customer Requires Human Assistance

### Manual Review

If:

* Customer is dissatisfied.
* Customer wants to file a complaint.
* Customer disputes calculations.
* Customer requests policy exceptions.
* Customer requests interest-rate negotiation.
* Customer requests special approvals.

Reason:
Manual Review Required

---

## Callback Scheduling

When a callback is requested:

Collect:

* Customer Name
* Mobile Number
* Preferred Callback Date
* Preferred Callback Time

If date or time is missing, ask for it.

After collection:

Call:

`schedule_callback`

with:

* Customer Name
* Mobile Number
* Callback Date
* Callback Time
* Loan Amount (if available)
* Preferred Bank (if available)

Then call:

`escalate_to_human`

Reason:
Callback Scheduled

Respond:

"Thank you. Your callback request has been scheduled. A loan specialist will contact you at your preferred time."

---

## Escalation Payload

Whenever escalation occurs, pass:

* Customer Name
* Mobile Number
* Loan Amount
* Preferred Bank/NBFC
* Employment Type
* Monthly Income
* Escalation Reason

For missing values use:

"Not Provided"

After escalation:

Inform the customer that a loan specialist will contact them.

Avoid continuing eligibility processing unless the customer explicitly requests it.

---

## Response Style

* Professional
* Friendly
* Trustworthy
* Empathetic
* WhatsApp-friendly
* Short and clear

Primary objective:
Qualify the lead, estimate affordability, recommend suitable lenders, and hand qualified customers to a human loan specialist when needed.
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
    def _escalate_to_human(
        customer_name: str, 
        mobile_number: str, 
        loan_amount: str, 
        preferred_bank: str, 
        employment_type: str, 
        monthly_income: str, 
        escalation_reason: str
    ) -> str:
        """
        Escalates the conversation to a human agent. Call this tool when escalation rules are triggered.
        """
        ESCALATION_STATUS[wa_id] = True

        escalation_msg = f"""
Escalation Required: Yes
Escalation Reason: {escalation_reason}
Customer Name: {customer_name}
Mobile Number: {mobile_number}
Loan Amount: {loan_amount}
Preferred Bank/NBFC: {preferred_bank}
Employment Type: {employment_type}
Monthly Income: {monthly_income}
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
            customer_name=name,
            mobile_number=wa_id,
            loan_amount="Not Provided",
            preferred_bank="Not Provided",
            employment_type="Not Provided",
            monthly_income="Not Provided",
            escalation_reason="Extended Qualification Assistance Required"
        )
        return "I'd like to ensure you receive the best assistance possible. I am escalating your request to a loan specialist who can guide you further."

    # --- Schedule Callback Tool ---
    def _schedule_callback(
        customer_name: str, 
        mobile_number: str, 
        callback_date: str, 
        callback_time: str, 
        loan_amount: str = "Not Provided", 
        preferred_bank: str = "Not Provided"
    ) -> str:
        """
        Schedules a callback for the customer at their preferred date and time.
        Call this tool when a user provides their preferred date and time for a callback.
        """
        callback_msg = f"""
Callback Scheduled!
Customer Name: {customer_name}
Mobile Number: {mobile_number}
Callback Date: {callback_date}
Callback Time: {callback_time}
Loan Amount: {loan_amount}
Preferred Bank: {preferred_bank}
        """.strip()
        
        operator = current_app.config.get("OPERATOR_WAID")
        if operator:
            from app.utils.whatsapp_utils import get_text_message_input
            send_message_callback(get_text_message_input(operator, callback_msg))
            logging.info(f"Callback scheduled sent to operator {operator} for guest {wa_id}")
            
        return "Callback scheduled successfully."

    tools = [
        StructuredTool.from_function(
            func=_escalate_to_human, 
            name="escalate_to_human",
            description="Escalates the conversation to a human agent. Call this when the user gets confused, complains, or triggers manual escalation rules."
        ),
        StructuredTool.from_function(
            func=_schedule_callback,
            name="schedule_callback",
            description="Schedules a callback. Call this only when the customer has provided a preferred callback date and time."
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
