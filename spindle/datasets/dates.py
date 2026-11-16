"""Generated rows that need the current date: the system prompt states "now", the request says
"tomorrow at 9" or "next Tuesday", and the answer holds the worked-out date in the format the
tool asks for.

Meaning of the phrases (so every row has one right answer):
  today / tonight / tomorrow / the day after tomorrow / in N days / in N weeks: counted from now.
  this <weekday>, next <weekday>, on <weekday>: the first such day after today.
  on <Month> <day>: that day in the current year (only days still ahead are generated).
"""
import random
from datetime import datetime, timedelta

from spindle.convert import call, tool

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']

# How the system prompt states the current moment.
NOW = [lambda n: f'Current date and time: {n:%Y-%m-%dT%H:%M:%S} ({n:%A})',
       lambda n: f'Today is {n:%A}, {MONTHS[n.month - 1]} {n.day}, {n.year}. The time is {n:%H:%M}.',
       lambda n: f'Date: {n:%Y-%m-%d}\nWeekday: {n:%A}\nTime: {n:%H:%M}',
       lambda n: f'You are a helpful assistant. Now: {n:%a} {n:%Y-%m-%d} {n:%H:%M:%S}.',
       lambda n: f'It is {n:%H:%M} on {n.day} {MONTHS[n.month - 1]} {n.year}, a {n:%A}.']

PEOPLE = ['Dr. Patel', 'Maria', 'the landlord', 'Kenji', 'Aisha', 'my manager', 'the plumber', 'Sam and Lee',
          'the design team', 'Grandma', 'Prof. Okafor', 'the accountant', 'Jonas', 'the vet', 'Chloe']
THINGS = ['dentist appointment', 'project kickoff', 'budget review', 'yoga class', 'team lunch', 'car service',
          'parent-teacher meeting', 'flight check-in', 'book club', 'quarterly planning', 'piano lesson',
          'gym session', 'product demo', 'haircut', 'standup', 'client call', 'movie night', 'physio session']
PLACES = ['Luigi\'s', 'Blue Lotus', 'The Corner Bistro', 'Sakura House', 'Casa Verde', 'Harbor Grill']

# (tool names, description, title field names, phrasing of the request). {t} is the title, {w} the time phrase.
ACTIONS = [
    (['create_event', 'add_calendar_event', 'schedule_event', 'new_event'], 'Add an event to the calendar.',
     ['title', 'name', 'summary'],
     ['Schedule {t} {w}.', 'Put {t} on my calendar {w}.', 'Add an event called "{t}" {w}.',
      'Can you create a calendar entry for {t} {w}?', 'I have {t} {w}, please add it.']),
    (['set_reminder', 'remind_me', 'create_reminder'], 'Create a reminder.', ['text', 'message', 'note'],
     ['Remind me about {t} {w}.', 'Set a reminder for {t} {w}.', 'I need a reminder {w}: {t}.',
      'Please remind me {w} about {t}.']),
    (['schedule_meeting', 'book_meeting', 'create_meeting'], 'Schedule a meeting.', ['topic', 'subject', 'title'],
     ['Set up a meeting about {t} {w}.', 'Book a meeting for {t} {w}.', 'We need to meet about {t} {w}.']),
    (['add_task', 'create_todo', 'new_task'], 'Add a task with a due date.', ['task', 'name', 'title'],
     ['Add a task: {t}, due {w}.', 'I have to finish {t} {w}, add it to my list.', 'Create a to-do for {t} {w}.']),
    (['book_table', 'reserve_table', 'make_reservation'], 'Reserve a table at a restaurant.',
     ['restaurant', 'place', 'venue'],
     ['Book a table at {t} {w}.', 'Reserve a table at {t} {w}.', 'Get us a table at {t} {w}, please.']),
]

DISTRACTORS = [('get_weather', 'Get the weather forecast for a city.', {'city': 'string'}),
               ('send_message', 'Send a text message.', {'to': 'string', 'body': 'string'}),
               ('play_music', 'Play a song or artist.', {'query': 'string'}),
               ('search_web', 'Search the web.', {'query': 'string'}),
               ('toggle_wifi', 'Turn Wi-Fi on or off.', {'enabled': 'boolean'}),
               ('get_directions', 'Get directions to a place.', {'destination': 'string'}),
               ('convert_currency', 'Convert an amount between currencies.',
                {'amount': 'number', 'source': 'string', 'target': 'string'})]


def _clock(rng):
    """(hour, minute, text) for a spoken time."""
    hour, minute = rng.randint(6, 22), rng.choice([0, 0, 0, 15, 30, 30, 45])
    if (hour, minute) == (12, 0) and rng.random() < 0.5:
        return hour, minute, 'noon'
    style = rng.random()
    if style < 0.25:
        return hour, minute, f'{hour:02d}:{minute:02d}'
    suffix = rng.choice([' PM', 'pm', ' p.m.']) if hour >= 12 else rng.choice([' AM', 'am', ' a.m.'])
    short = hour - 12 if hour > 12 else hour
    return hour, minute, f'{short}{suffix}' if minute == 0 and style < 0.7 else f'{short}:{minute:02d}{suffix}'


def _day(now, rng):
    """(date, phrase) for a day named relative to now."""
    kind = rng.random()
    if kind < 0.14:
        return now, rng.choice(['today', 'later today'])
    if kind < 0.34:
        return now + timedelta(days=1), 'tomorrow'
    if kind < 0.40:
        return now + timedelta(days=2), 'the day after tomorrow'
    if kind < 0.50:
        days = rng.randint(2, 12)
        return now + timedelta(days=days), f'in {days} days'
    if kind < 0.56:
        weeks = rng.randint(1, 4)
        return now + timedelta(weeks=weeks), 'in a week' if weeks == 1 else f'in {weeks} weeks'
    if kind < 0.84:
        ahead = rng.randint(1, 7)
        day = now + timedelta(days=ahead)
        # A week from today is only ever "next <weekday>": "this Wednesday" said on a Wednesday is unclear.
        forms = ['next {d}'] if ahead == 7 else ['next {d}', 'this {d}', 'on {d}', 'this coming {d}']
        return day, rng.choice(forms).format(d=DAYS[day.weekday()])
    day = now + timedelta(days=rng.randint(3, 120))
    if day.year != now.year:
        day = now + timedelta(days=3)
    name = MONTHS[day.month - 1]
    return day, rng.choice([f'on {name} {day.day}', f'on {day.day} {name}', f'on {DAYS[day.weekday()]}, {name} {day.day}',
                            f'on {name[:3]} {day.day}'])


def _when(now, rng):
    """(datetime, phrase): a relative day plus a time of day."""
    day, phrase = _day(now, rng)
    if phrase in ('today', 'later today') and rng.random() < 0.5 and now.hour < 19:
        hour = rng.randint(max(now.hour + 1, 18), 22)
        moment = day.replace(hour=hour, minute=0, second=0)
        return moment, f'tonight at {hour - 12}'
    hour, minute, clock = _clock(rng)
    moment = day.replace(hour=hour, minute=minute, second=0)
    return moment, rng.choice(['{p} at {c}', 'at {c} {p}']).format(p=phrase, c=clock)


def _row(rng):
    now = datetime(2023, 1, 1) + timedelta(days=rng.randint(0, 365 * 5), hours=rng.randint(6, 21),
                                           minutes=rng.randint(0, 59), seconds=rng.randint(0, 59))
    names, description, title_fields, phrasings = rng.choice(ACTIONS)
    moment, phrase = _when(now, rng)
    title = (rng.choice(PLACES) if 'table' in description else
             rng.choice(THINGS) if rng.random() < 0.6 else f'{rng.choice(["call", "lunch", "meeting", "coffee"])} with '
                                                             f'{rng.choice(PEOPLE)}')
    name, title_field = rng.choice(names), rng.choice(title_fields)
    properties = {title_field: {'type': 'string'}}
    arguments = {title_field: title}
    if rng.random() < 0.7:  # one field with date and time
        field = rng.choice(['datetime', 'start', 'when', 'time', 'start_time', 'due'])
        properties[field] = {'type': 'string', 'description': rng.choice([
            'Date and time in YYYY-MM-DDTHH:MM:SS format', 'ISO 8601 date and time, e.g. 2025-03-09T14:30:00',
            'When it starts (YYYY-MM-DDTHH:MM:SS)'])}
        arguments[field] = moment.strftime('%Y-%m-%dT%H:%M:%S')
        resolved = f"{field} '{phrase}' relative to system date '{now:%Y-%m-%dT%H:%M:%S}' -> '{arguments[field]}'"
    else:  # the date and the time in separate fields
        date_field, time_field = rng.choice([('date', 'time'), ('day', 'start_time'), ('due_date', 'due_time')])
        properties[date_field] = {'type': 'string', 'description': 'Date as YYYY-MM-DD'}
        properties[time_field] = {'type': 'string', 'description': '24-hour time as HH:MM'}
        arguments[date_field], arguments[time_field] = moment.strftime('%Y-%m-%d'), moment.strftime('%H:%M')
        resolved = (f"{date_field} '{phrase}' relative to system date '{now:%Y-%m-%d}' -> '{arguments[date_field]}'; "
                    f"{time_field} '{arguments[time_field]}'")
    fields = list(properties.items())
    rng.shuffle(fields)
    tools = [tool(name, description, {'type': 'object', 'properties': dict(fields), 'required': list(properties)})]
    for other, text, spec in rng.sample(DISTRACTORS, rng.randint(0, 3)):
        tools.append(tool(other, text, {'type': 'object', 'properties': {k: {'type': v} for k, v in spec.items()}}))
    rng.shuffle(tools)
    return {'query': rng.choice(phrasings).format(t=title, w=phrase), 'system': rng.choice(NOW)(now),
            'tools': tools, 'answers': [call(name, arguments)], 'task': 'tools', 'source': 'dates',
            'reasoning': f"{name}: {title_field} '{title}' from query; {resolved}."}


def generate(count=8000, seed=5):
    rng = random.Random(seed)
    for _ in range(count):
        yield _row(rng)


SOURCES = {'dates': generate}
