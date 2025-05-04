# calendar_helpers.py - Helper functions for calendar operations
import re
import datetime
import dateparser
from typing import List, Optional, Dict, Any, Tuple

# ------------------------------
# INTENT DETECTION
# ------------------------------
def detect_calendar_intent(message: str) -> Optional[str]:
    """Detect if a message contains calendar-related intents"""
    message = message.lower()
    
    # Check availability patterns
    availability_patterns = [
        r"(free|available|open).*?(slot|time|appointment)",
        r"when.*?(free|available)",
        r"check.*?(calendar|schedule|availability)",
        r"what.*?(time|date).*?(available|free)",
        r"available.*?(time|slot|date)"
    ]
    
    # Book appointment patterns
    booking_patterns = [
        r"book.*?(appointment|meeting|call|demo)",
        r"schedule.*?(appointment|meeting|call|demo)",
        r"set up.*?(appointment|meeting|call|demo)",
        r"reserve.*?(time|slot)",
        r"(can|could).*?(schedule|book|have).*?(call|meeting|appointment|demo)"
    ]
    
    # Cancel appointment patterns
    cancel_patterns = [
        r"cancel.*?(appointment|meeting|call|demo)",
        r"reschedule.*?(appointment|meeting|call|demo)",
        r"delete.*?(appointment|meeting|call|demo)"
    ]
    
    # List appointments patterns
    list_patterns = [
        r"(show|list|what).*?(appointment|meeting|schedule|calendar)",
        r"upcoming.*?(appointment|meeting|event)",
        r"my.*?(schedule|calendar|appointment)"
    ]
    
    # Check for matches
    for pattern in availability_patterns:
        if re.search(pattern, message):
            return "check_availability"
    
    for pattern in booking_patterns:
        if re.search(pattern, message):
            return "book_appointment"
    
    for pattern in cancel_patterns:
        if re.search(pattern, message):
            return "cancel_appointment"
    
    for pattern in list_patterns:
        if re.search(pattern, message):
            return "list_appointments"
    
    return None

# ------------------------------
# DATA EXTRACTION
# ------------------------------
def extract_dates(message: str) -> List[datetime.datetime]:
    """Extract date and time information from a message with preference for future dates"""
    # Common date patterns
    date_patterns = [
        r"(today|tomorrow|next week|this week)",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}",
        r"\d{1,2}/\d{1,2}(/\d{2,4})?",
        r"\d{1,2}-\d{1,2}(-\d{2,4})?"
    ]
    
    # Time patterns
    time_patterns = [
        r"\d{1,2}:\d{2}\s*([ap]m)?",
        r"\d{1,2}\s*([ap]m)",
        r"(morning|afternoon|evening)"
    ]
    
    # Extract potential date and time strings
    date_matches = []
    for pattern in date_patterns:
        matches = re.finditer(pattern, message.lower())
        for match in matches:
            date_matches.append(match.group(0))
    
    time_matches = []
    for pattern in time_patterns:
        matches = re.finditer(pattern, message.lower())
        for match in matches:
            time_matches.append(match.group(0))
    
    # Try to parse each match into an actual date
    parsed_dates = []
    for date_str in date_matches:
        # For weekday names, add "next" to prefer future dates
        if date_str.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            # Try with "next" prefix first to get upcoming date
            parsed_date = dateparser.parse(f"next {date_str}", settings={'PREFER_DATES_FROM': 'future'})
            if not parsed_date:
                # Fall back to regular parsing
                parsed_date = dateparser.parse(date_str)
        else:
            # For other dates, use regular parsing with future preference
            parsed_date = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'future'})
        
        if parsed_date:
            parsed_dates.append(parsed_date)
    
    # If we have both date and time, try to combine them
    if parsed_dates and time_matches:
        date_with_time = []
        for time_str in time_matches:
            parsed_time = dateparser.parse(time_str)
            if parsed_time:
                # Create combined datetime
                combined = datetime.datetime.combine(
                    parsed_dates[0].date(),
                    parsed_time.time()
                )
                date_with_time.append(combined)
        
        # If we successfully combined date and time, use those instead
        if date_with_time:
            return date_with_time
    
    return parsed_dates

def extract_meeting_purpose(message: str) -> Optional[str]:
    """Extract the purpose or topic of a meeting from a message"""
    purpose_patterns = [
        r"(about|regarding|for|to discuss) (.+?)(?:\.|,|$)",
        r"(topic|subject|matter|purpose) (?:is|will be) (.+?)(?:\.|,|$)",
        r"(?:a|an) (consultation|discussion|meeting|call) (?:about|regarding|on) (.+?)(?:\.|,|$)",
        r"(?:want to|would like to) (talk about|discuss) (.+?)(?:\.|,|$)"
    ]
    
    for pattern in purpose_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(2).strip()
    
    return None

def extract_email(message: str) -> Optional[str]:
    """Extract email addresses from a message"""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    match = re.search(email_pattern, message)
    if match:
        return match.group(0)
    
    return None

# ------------------------------
# CALENDAR INTENT PROCESSING
# ------------------------------
def process_calendar_intent(intent: str, message: str) -> Tuple[str, Dict[str, Any]]:
    """
    Process a calendar intent from a user message
    
    Args:
        intent: The detected calendar intent
        message: The user's message
        
    Returns:
        Tuple[str, dict]: (instruction for KILA, extracted data)
    """
    # Extract data from the message
    extracted_data = {}
    
    # Extract dates
    dates = extract_dates(message)
    if dates:
        extracted_data["dates"] = [d.isoformat() for d in dates]
        formatted_dates = [d.strftime("%A, %B %d at %I:%M %p") if d.hour != 0 else d.strftime("%A, %B %d") 
                          for d in dates]
        extracted_data["formatted_dates"] = formatted_dates
    
    # Extract purpose
    purpose = extract_meeting_purpose(message)
    if purpose:
        extracted_data["purpose"] = purpose
    
    # Extract email
    email = extract_email(message)
    if email:
        extracted_data["email"] = email
    
    # Prepare instructions based on intent
    instruction = ""
    
    if intent == "check_availability":
        if dates:
            date_str = ", ".join(extracted_data["formatted_dates"])
            instruction = f"The user is asking about calendar availability for {date_str}. You should inform them that you'll check the calendar for available times."
        else:
            instruction = "The user is asking about calendar availability but didn't specify a date. Ask for which date they'd like to check."
    
    elif intent == "book_appointment":
        instruction = "The user wants to book an appointment."
        if dates:
            instruction += f" They mentioned {', '.join(extracted_data['formatted_dates'])}."
        else:
            instruction += " Ask them what date and time they prefer."
            
        if purpose:
            instruction += f" The purpose appears to be: {purpose}."
        else:
            instruction += " Ask about the purpose of the meeting if they haven't mentioned it."
            
        instruction += " Let them know that you'll check availability and get back after approval."
    
    elif intent == "cancel_appointment":
        if dates:
            date_str = ", ".join(extracted_data["formatted_dates"])
            instruction = f"The user wants to cancel an appointment on {date_str}. Ask for confirmation before proceeding."
        else:
            instruction = "The user wants to cancel an appointment. Ask which appointment they'd like to cancel."
    
    elif intent == "list_appointments":
        instruction = "The user wants to know about upcoming appointments. Let them know you'll check the calendar."
    
    else:
        instruction = "The user mentioned something calendar-related, but the intent isn't clear. Ask for clarification."
    
    return instruction, extracted_data
