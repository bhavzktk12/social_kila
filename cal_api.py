# calendar_api.py - Handles Google Calendar operations for KILA
from google.oauth2 import service_account
from googleapiclient.discovery import build
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import datetime
import pytz
import re
import dateparser
import os
import json
import tempfile

# ------------------------------
# CONFIGURATION
# ------------------------------
# Check for service account file
possible_paths = [
    "./kila-456404-d406b0474c4f.json",  # Original path
    "/etc/secrets/kila-456404-d406b0474c4f.json",  # Render secrets path
    os.path.join(os.getcwd(), "kila-456404-d406b0474c4f.json")  # Absolute path
]

# Find the first path that exists
SERVICE_ACCOUNT_FILE = None
for path in possible_paths:
    if os.path.exists(path):
        SERVICE_ACCOUNT_FILE = path
        print(f"[CALENDAR] Found service account file at: {path}")
        break

# If file not found, try to create it from environment variable
if not SERVICE_ACCOUNT_FILE:
    print("[CALENDAR] Service account file not found, checking environment variable")
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    
    if service_account_json:
        # Create a temporary file with the JSON content
        try:
            fd, SERVICE_ACCOUNT_FILE = tempfile.mkstemp(suffix='.json')
            with os.fdopen(fd, 'w') as f:
                f.write(service_account_json)
            print(f"[CALENDAR] Created service account file from environment variable at: {SERVICE_ACCOUNT_FILE}")
        except Exception as e:
            print(f"[CALENDAR] Error creating service account file: {str(e)}")
            # Fall back to default path
            SERVICE_ACCOUNT_FILE = "./kila-456404-d406b0474c4f.json"
    else:
        print("[CALENDAR] WARNING: No service account file or environment variable found!")
        # Fall back to default path in case something else handles it
        SERVICE_ACCOUNT_FILE = "./kila-456404-d406b0474c4f.json"

CALENDAR_ID = "ceo@agently-ai.com"
TIMEZONE = "America/Chicago"  # Adjust to your timezone

# ------------------------------
# MODELS
# ------------------------------
class AvailabilityRequest(BaseModel):
    date: str
    duration_minutes: Optional[int] = 60

class TimeSlot(BaseModel):
    start: str
    end: str

class EventRequest(BaseModel):
    summary: str
    start_time: str
    end_time: str
    description: Optional[str] = None
    attendees: Optional[List[str]] = None

class EventResponse(BaseModel):
    id: str
    summary: str
    start: str
    end: str
    link: Optional[str] = None

class CalendarResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# ------------------------------
# CALENDAR SERVICE
# ------------------------------
def get_calendar_service():
    """Get an authorized Google Calendar API service"""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/calendar"],
            subject="ceo@agently-ai.com"  # Added to impersonate the calendar owner
        )
        return build("calendar", "v3", credentials=credentials)
    except Exception as e:
        print(f"[CALENDAR] Error getting calendar service: {str(e)}")
        raise

# ------------------------------
# ROUTER
# ------------------------------
router = APIRouter(prefix="/calendar", tags=["calendar"])

@router.get("/availability", response_model=CalendarResponse)
async def check_availability(date: str, duration_minutes: int = 30):
    """Check calendar availability for a specific date"""
    try:
        service = get_calendar_service()
        
        # Parse the date string
        try:
            # Use future preference for dates
            parsed_date = dateparser.parse(date, settings={'PREFER_DATES_FROM': 'future'})
            if not parsed_date:
                return CalendarResponse(
                    success=False,
                    message=f"Could not parse date: {date}",
                    data={}
                )
            target_date = parsed_date.date()
        except Exception as e:
            return CalendarResponse(
                success=False,
                message=f"Error parsing date: {str(e)}",
                data={}
            )
        
        # Set time range for the day
        timezone = pytz.timezone(TIMEZONE)
        start_of_day = timezone.localize(datetime.datetime.combine(
            target_date, datetime.datetime.min.time()))
        end_of_day = timezone.localize(datetime.datetime.combine(
            target_date, datetime.datetime.max.time()))
        
        # Define business hours (9 AM to 5 PM)
        business_start = timezone.localize(datetime.datetime.combine(
            target_date, datetime.time(9, 0)))
        business_end = timezone.localize(datetime.datetime.combine(
            target_date, datetime.time(17, 0)))
        
        # Get busy periods from calendar
        body = {
            "timeMin": start_of_day.isoformat(),
            "timeMax": end_of_day.isoformat(),
            "items": [{"id": CALENDAR_ID}]
        }
        
        busy_periods = service.freebusy().query(body=body).execute()
        
        # Extract busy time slots
        busy_slots = []
        for calendar_id, busy_data in busy_periods.get('calendars', {}).items():
            for busy_slot in busy_data.get('busy', []):
                start = datetime.datetime.fromisoformat(
                    busy_slot['start'].replace('Z', '+00:00'))
                end = datetime.datetime.fromisoformat(
                    busy_slot['end'].replace('Z', '+00:00'))
                busy_slots.append((start, end))
        
        # Generate available slots with requested duration
        available_slots = []
        slot_start = business_start
        
        while slot_start < business_end:
            slot_end = slot_start + datetime.timedelta(minutes=duration_minutes)
            if slot_end > business_end:
                break
                
            # Check if slot overlaps with any busy period
            is_available = True
            for busy_start, busy_end in busy_slots:
                if (slot_start < busy_end and slot_end > busy_start):
                    is_available = False
                    break
            
            if is_available:
                # Convert to local time for display
                local_start = slot_start.astimezone(timezone)
                local_end = slot_end.astimezone(timezone)
                available_slots.append(TimeSlot(
                    start=local_start.isoformat(),
                    end=local_end.isoformat()
                ))
            
            # Move to next slot (30-minute increments)
            slot_start = slot_start + datetime.timedelta(minutes=30)
        
        friendly_date = target_date.strftime("%A, %B %d, %Y")
        return CalendarResponse(
            success=True,
            message=f"Found {len(available_slots)} available slots for {friendly_date}",
            data={
                "date": friendly_date,
                "available_slots": [{"start": slot.start, "end": slot.end} for slot in available_slots]
            }
        )
    
    except Exception as e:
        return CalendarResponse(
            success=False,
            message=f"Error checking availability: {str(e)}",
            data={}
        )

@router.post("/events", response_model=CalendarResponse)
async def create_event(event_data: EventRequest):
    """Create a new calendar event"""
    try:
        service = get_calendar_service()
        
        # Parse start and end times
        try:
            start_time = dateparser.parse(event_data.start_time, settings={'PREFER_DATES_FROM': 'future'})
            end_time = dateparser.parse(event_data.end_time, settings={'PREFER_DATES_FROM': 'future'})
            
            if not start_time or not end_time:
                return CalendarResponse(
                    success=False,
                    message="Could not parse start or end time",
                    data={}
                )
                
            # Add timezone if not present
            timezone = pytz.timezone(TIMEZONE)
            if start_time.tzinfo is None:
                start_time = timezone.localize(start_time)
            if end_time.tzinfo is None:
                end_time = timezone.localize(end_time)
                
        except Exception as e:
            return CalendarResponse(
                success=False,
                message=f"Error parsing dates: {str(e)}",
                data={}
            )
        
        # Create event
        event = {
            'summary': event_data.summary,
            'description': event_data.description or '',
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': TIMEZONE,
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': TIMEZONE,
            },
        }
        
        # Add attendees if provided
        if event_data.attendees:
            event['attendees'] = [{'email': email} for email in event_data.attendees]
        
        # Insert the event
        created_event = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        
        return CalendarResponse(
            success=True,
            message=f"Event '{event_data.summary}' created successfully",
            data={
                "event": {
                    "id": created_event['id'],
                    "summary": created_event['summary'],
                    "start": created_event['start']['dateTime'],
                    "end": created_event['end']['dateTime'],
                    "link": created_event.get('htmlLink')
                }
            }
        )
    
    except Exception as e:
        return CalendarResponse(
            success=False,
            message=f"Error creating event: {str(e)}",
            data={}
        )

@router.get("/events", response_model=CalendarResponse)
async def list_events(max_results: int = 10):
    """List upcoming calendar events"""
    try:
        service = get_calendar_service()
        
        # Get current time
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        
        # Call the Calendar API
        events_result = service.events().list(
            calendarId=CALENDAR_ID,
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Format the events
        formatted_events = []
        for event in events:
            try:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                
                formatted_events.append({
                    "id": event['id'],
                    "summary": event.get('summary', '(No title)'),
                    "start": start,
                    "end": end,
                    "link": event.get('htmlLink')
                })
            except Exception as e:
                # If there's an error with one event, continue with others
                continue
        
        return CalendarResponse(
            success=True,
            message=f"Found {len(formatted_events)} upcoming events",
            data={"events": formatted_events}
        )
    
    except Exception as e:
        return CalendarResponse(
            success=False,
            message=f"Error listing events: {str(e)}",
            data={}
        )

@router.delete("/events/{event_id}", response_model=CalendarResponse)
async def cancel_event(event_id: str):
    """Cancel/delete a calendar event"""
    try:
        service = get_calendar_service()
        
        # Get the event details before deleting
        try:
            event = service.events().get(calendarId=CALENDAR_ID, eventId=event_id).execute()
            summary = event.get('summary', 'Unknown event')
        except Exception:
            return CalendarResponse(
                success=False,
                message=f"Event with ID {event_id} not found",
                data={}
            )
        
        # Delete the event
        service.events().delete(calendarId=CALENDAR_ID, eventId=event_id).execute()
        
        return CalendarResponse(
            success=True,
            message=f"Event '{summary}' cancelled successfully",
            data={"event_id": event_id}
        )
    
    except Exception as e:
        return CalendarResponse(
            success=False,
            message=f"Error cancelling event: {str(e)}",
            data={}
        )
