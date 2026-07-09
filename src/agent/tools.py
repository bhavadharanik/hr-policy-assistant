from typing import Optional
from langchain_core.tools import tool


@tool
def lookup_employee_record(employee_id: str) -> dict:
    """Look up an employee record by ID. Returns basic employee information."""
    # Stubbed employee database
    employees = {
        "EMP001": {"name": "Alice Johnson", "department": "Engineering", "start_date": "2022-03-15", "leave_balance": 18},
        "EMP002": {"name": "Bob Smith", "department": "Marketing", "start_date": "2021-07-01", "leave_balance": 22},
        "EMP003": {"name": "Carol White", "department": "HR", "start_date": "2020-01-10", "leave_balance": 25},
    }
    return employees.get(employee_id, {"error": f"Employee {employee_id} not found"})


@tool
def escalate_to_hr(reason: str, question: str, employee_id: Optional[str] = None) -> dict:
    """Escalate a question to a human HR representative when the agent cannot confidently answer."""
    ticket_id = f"HR-{hash(question) % 10000:04d}"
    print(f"\n[ESCALATION] Ticket {ticket_id} created")
    print(f"Reason: {reason}")
    print(f"Question: {question}")
    return {
        "ticket_id": ticket_id,
        "status": "escalated",
        "message": f"Your question has been escalated to HR. Reference: {ticket_id}. You will hear back within 2 business days."
    }
