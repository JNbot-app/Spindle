"""Generated rows for app-style actions, in the shapes small models tend to decline wrongly.

  record    view / edit / delete a record named by a URI ("content://notes/42"); edits pass a
            free-form object with only the stated fields
  files     pick files by type (MIME list + multiple) or create a file (MIME type + name)
  search    imperative research requests ("Explore ...", "Discover ...") -> search(query)
  lookup    one field (phone, email, address, link) of a named person
  timer     countdown with a spelled-out duration ("two hours") and a label
  bare      tools without arguments ("list the alarm sounds") -> {}

Every request also comes as a no-call row: the same shapes, but no listed tool does the job. Tool
names use four naming styles, and 0-3 unrelated tools are added so list length gives no hint.
Rows carry `teacher: False`: the call / no-call distillation leaves them out (see
spindle/training/decision.py), since the labels here are exact.
"""
import random

N = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
PEOPLE = ['Marcus Lee', 'Priya Raman', 'Olivia', 'Dr. Kowalski', 'Hannah Becker', 'Tomás', 'Grace Kim', 'Leo',
          'Fatima Noor', 'Ingrid Svensson', 'Mateo Rossi', 'Yuki Tanaka', 'Ben Carter', 'Amara Obi', 'Noah',
          'Sofia Alvarez', 'Ethan Brooks', 'Mia Chen', 'the landlord', 'our plumber', 'Coach Rivera', 'Aunt Rosa']
PURPOSES = ['for the quarterly report', 'for my thesis', 'for the client pitch', 'for this weekend',
            'for the design review', 'for our team offsite', 'before the audit', 'for the new onboarding flow',
            'for my portfolio', 'for the marketing launch', '', '', '', '']


def tool(name, description, properties=None, required=()):
    return {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties or {}, 'required': list(required)}}


def string(description):
    return {'type': 'string', 'description': description}


def call_row(query, tools, calls, kind):
    return {'query': query, 'tools': tools, 'answers': calls, 'task': 'tools', 'source': f'app_{kind}',
            'teacher': False}


def styled(rng, base):
    """A tool name in one of the naming styles apps use: VIEW_NOTE, view_note, viewNote, ACTION_VIEW_NOTE."""
    words = base.split('_')
    style = rng.random()
    if style < 0.3:
        return base.upper()
    if style < 0.45:
        return 'INTENT_' + base.upper() if rng.random() < 0.5 else 'APP_' + base.upper()
    if style < 0.6:
        return words[0] + ''.join(w.title() for w in words[1:])
    return base


# ---------------------------------------------------------------- record by URI
# (entity, uri forms, editable fields with example values)
ENTITIES = [
    ('note', ['content://notes/{n}', 'content://notes/items/{n}', 'notes://item/{n}'],
     {'title': ['Grocery plan', 'Trip ideas', 'Meeting recap'], 'body': ['Buy oat milk and rice', 'Call the bank on Monday'],
      'color': ['yellow', 'blue', 'green'], 'pinned': [True, False]}),
    ('event', ['content://calendar/events/{n}', 'content://com.calendar/events/{n}', 'calendar://event/{n}'],
     {'title': ['Board meeting', 'Dentist', 'Sprint review'], 'location': ['Room 4B', '22 Baker Street', 'Online'],
      'description': ['Bring the slides', 'Agenda in the shared folder']}),
    ('task', ['content://tasks/{n}', 'app://tasks/{n}', 'todo://task/{n}'],
     {'title': ['File taxes', 'Renew passport', 'Fix the login bug'], 'priority': ['high', 'low', 'medium'],
      'status': ['done', 'in progress', 'blocked'], 'assignee': ['Grace Kim', 'Leo', 'Mateo Rossi']}),
    ('playlist', ['content://media/playlists/{n}', 'music://playlist/{n}'],
     {'name': ['Road Trip', 'Focus Mix', 'Sunday Morning'], 'description': ['Songs for long drives', 'Calm piano'],
      'public': [True, False]}),
    ('album', ['content://media/albums/{n}', 'content://photos/albums/{n}'],
     {'name': ['Lisbon 2024', 'Wedding', 'Garden'], 'cover_photo': ['IMG_2041.jpg', 'IMG_0007.jpg'],
      'shared': [True, False]}),
    ('bookmark', ['content://browser/bookmarks/{n}', 'bookmarks://item/{n}'],
     {'title': ['Recipe blog', 'Tax portal', 'Train times'], 'url': ['https://example.org/recipes', 'https://tickets.example.com'],
      'folder': ['Cooking', 'Work', 'Travel']}),
    ('customer', ['crm://customers/{n}', 'content://crm/customers/{n}'],
     {'email': ['j.ortega@northwind.io', 'sales@brightlabs.co'], 'phone': ['617-555-0192', '+44 20 7946 0321'],
      'company': ['Northwind', 'Bright Labs', 'Acme Corp'], 'address': ['12 Elm Road, Leeds', '900 Market St, Denver']}),
    ('employee', ['hr://employees/{n}', 'content://directory/people/{n}'],
     {'title': ['Senior Analyst', 'Team Lead'], 'department': ['Finance', 'Research', 'Support'],
      'phone': ['212-555-0148', '303-555-0177'], 'office': ['Floor 3, desk 12', 'Building C']}),
    ('ticket', ['support://tickets/{n}', 'content://helpdesk/tickets/{n}'],
     {'status': ['closed', 'open', 'waiting'], 'priority': ['urgent', 'normal'], 'assignee': ['Hannah Becker', 'Ben Carter'],
      'summary': ['Printer jams on page 2', 'Cannot reset password']}),
    ('recipe', ['content://recipes/{n}', 'cookbook://recipe/{n}'],
     {'name': ['Lemon tart', 'Green curry'], 'servings': [4, 6], 'notes': ['Use less sugar', 'Add lime at the end']}),
]
VIEW_ASKS = ['Show the {e} at {u}.', "Open the {e} referenced by '{u}'.", 'Display the {e} details for {u}.',
             "Can you pull up the {e} stored at '{u}'?", 'Let me see the {e} with the URI {u}.',
             "Fetch and show the {e} located at '{u}', I need to check something.", 'Bring up {u} for me.',
             'Present the {e} information found at {u}.', "I'd like to look at the {e} under '{u}'.",
             'Retrieve and display the {e} from the URI {u} in the app.', 'Go to the {e} page for {u}.']
EDIT_ASKS = ["Update the {e} at '{u}' so that {c}.", 'For the {e} at {u}, {c}.', "Change the {e} '{u}': {c}.",
             "Could you edit {u} and make sure {c}?", "I need the {e} located at '{u}' changed: {c}.",
             "Quickly modify the {e} {u} so {c}.", "Please adjust the {e} at {u} — {c}."]
DELETE_ASKS = ["Delete the {e} at '{u}'.", 'Remove {u}, I no longer need it.', 'Get rid of the {e} stored at {u}.',
               "Please erase the {e} '{u}'."]


def _change_text(field, value):
    if isinstance(value, bool):
        return f'{field.replace("_", " ")} is {"on" if value else "off"}' if field != 'pinned' else (
            'it is pinned' if value else 'it is not pinned')
    return f"the {field.replace('_', ' ')} is '{value}'" if isinstance(value, str) else f'{field} is {value}'


def record_row(rng):
    entity, uris, fields = rng.choice(ENTITIES)
    uri = rng.choice(uris).format(n=rng.randint(1, 9999))
    uri_field = rng.choice(['uri', f'{entity}_uri', 'content_uri', 'item_uri'])
    action = rng.choices(['view', 'edit', 'delete'], [0.45, 0.4, 0.15])[0]
    if action == 'view':
        name = styled(rng, f'view_{entity}')
        tools = [tool(name, rng.choice([f'Open a {entity} in the app and show its details.',
                                        f'Display the {entity} identified by a content URI.']),
                      {uri_field: string(f'Content URI of the {entity}, e.g. content://...')}, [uri_field])]
        query, arguments = rng.choice(VIEW_ASKS).format(e=entity, u=uri), {uri_field: uri}
    elif action == 'delete':
        name = styled(rng, f'delete_{entity}')
        tools = [tool(name, f'Delete a {entity} by its URI.', {uri_field: string(f'URI of the {entity}')}, [uri_field])]
        query, arguments = rng.choice(DELETE_ASKS).format(e=entity, u=uri), {uri_field: uri}
    else:
        name = styled(rng, f'edit_{entity}')
        object_field = rng.choice([f'{entity}_info', 'changes', 'fields', 'updates'])
        keys = ', '.join(fields)
        tools = [tool(name, f'Edit an existing {entity}. Only the given fields change.',
                      {uri_field: string(f'URI of the {entity} to edit'),
                       object_field: {'type': 'object', 'description': f'Fields to change. Possible keys: {keys}.'}},
                      [uri_field, object_field])]
        chosen = rng.sample(list(fields), rng.choice([1, 1, 1, 2, 2, 3]) if len(fields) > 2 else 1)
        changes = {f: rng.choice(fields[f]) for f in chosen}
        text = ' and '.join(_change_text(f, v) for f, v in changes.items())
        query, arguments = rng.choice(EDIT_ASKS).format(e=entity, u=uri, c=text), {uri_field: uri, object_field: changes}
    if rng.random() < 0.35:  # the same app also offers the other actions on this entity
        others = [a for a in ('view', 'edit', 'delete', 'share') if a != action]
        for other in rng.sample(others, rng.randint(1, 2)):
            tools.append(tool(styled(rng, f'{other}_{entity}'), f'{other.title()} a {entity} by its URI.',
                              {'uri': string(f'URI of the {entity}')}, ['uri']))
    return call_row(query[0].upper() + query[1:], tools, [{'name': name, 'arguments': arguments}], 'record')


# ---------------------------------------------------------------- files
# (spoken file kind, MIME type, file extension)
FILE_TYPES = [('PDF', 'application/pdf', 'pdf'), ('Word', 'application/msword', 'doc'),
              ('PNG image', 'image/png', 'png'), ('JPEG', 'image/jpeg', 'jpg'), ('GIF', 'image/gif', 'gif'),
              ('MP3', 'audio/mpeg', 'mp3'), ('WAV audio', 'audio/wav', 'wav'), ('MP4 video', 'video/mp4', 'mp4'),
              ('plain text', 'text/plain', 'txt'), ('CSV', 'text/csv', 'csv'), ('HTML', 'text/html', 'html'),
              ('JSON', 'application/json', 'json'), ('XML', 'application/xml', 'xml'),
              ('ZIP archive', 'application/zip', 'zip'), ('Excel spreadsheet', 'application/vnd.ms-excel', 'xls'),
              ('SVG', 'image/svg+xml', 'svg'), ('WebM video', 'video/webm', 'webm'), ('Markdown', 'text/markdown', 'md'),
              ('PowerPoint', 'application/vnd.ms-powerpoint', 'ppt'), ('YAML', 'application/x-yaml', 'yaml')]
FILE_NAMES = ['budget_2025', 'notes', 'inventory', 'Guide', 'export', 'Q3-summary', 'roadmap', 'config', 'invoice_114',
              'playlist', 'ProjectPlan', 'survey_results', 'backup', 'Changelog']
PICK_ONE = ['Let me pick a {k} file {p}.', 'Open the file picker so I can choose one {k} {p}.',
            'I need to attach a single {k} file {p}.', 'Select a {k} from my storage {p}.']
PICK_MANY = ['Help me open several {k} files {p}.', 'Begin importing {k} files {p}.', 'Fetch some {k} files {p}.',
             'I want to select a few {k} files {p}.', 'Give me access to multiple {k} files {p}.',
             'Let me choose a bunch of {k} files {p}.']
PICK_TWO = ['Fetch some {k} and {k2} files {p}.', 'Help me select {k} and {k2} files {p}.',
            'Open several {k} or {k2} files {p}.']
CREATE = ["Create a new {k} file named '{n}' {p}.", "Start a {k} document called '{n}' {p}.",
          "Make an empty {k} file, name it '{n}'.", "Set up a fresh {k} file titled '{n}' {p}.",
          "Develop a {k} file for me with the name '{n}'."]


def files_row(rng):
    if rng.random() < 0.6:
        name = styled(rng, rng.choice(['pick_files', 'choose_files', 'open_file_picker', 'select_documents']))
        types_field, multi_field = rng.choice([('mime_types', 'allow_multiple'), ('types', 'multiple'),
                                               ('file_types', 'select_many')])
        tools = [tool(name, 'Let the user pick existing files of the given MIME types.',
                      {types_field: {'type': 'array', 'items': {'type': 'string'},
                                     'description': 'MIME types to show, e.g. image/png'},
                       multi_field: {'type': 'boolean', 'description': 'Whether several files may be picked'}},
                      [types_field])]
        purpose = rng.choice(PURPOSES)
        shape = rng.random()
        if shape < 0.3:
            kind = rng.choice(FILE_TYPES)
            query, arguments = rng.choice(PICK_ONE).format(k=kind[0], p=purpose), {types_field: [kind[1]]}
            if rng.random() < 0.5:
                arguments[multi_field] = False
        elif shape < 0.75:
            kind = rng.choice(FILE_TYPES)
            query, arguments = rng.choice(PICK_MANY).format(k=kind[0], p=purpose), {types_field: [kind[1]], multi_field: True}
        else:
            a, b = rng.sample(FILE_TYPES, 2)
            query, arguments = rng.choice(PICK_TWO).format(k=a[0], k2=b[0], p=purpose), {types_field: [a[1], b[1]],
                                                                                        multi_field: True}
    else:
        name = styled(rng, rng.choice(['create_file', 'new_document', 'create_document', 'make_file']))
        type_field, name_field = rng.choice([('mime_type', 'file_name'), ('type', 'title'), ('mime_type', 'initial_name')])
        tools = [tool(name, 'Create a new empty file and open it.',
                      {type_field: string('MIME type of the new file'), name_field: string('Name of the file')},
                      [type_field, name_field])]
        kind = rng.choice(FILE_TYPES)
        file_name = f'{rng.choice(FILE_NAMES)}.{kind[2]}'
        query = rng.choice(CREATE).format(k=kind[0], n=file_name, p=rng.choice(PURPOSES))
        arguments = {type_field: kind[1], name_field: file_name}
    return call_row(' '.join(query.split()).replace(' .', '.'), tools, [{'name': name, 'arguments': arguments}], 'files')


# ---------------------------------------------------------------- search
TOPICS = ['the history of the Silk Road', 'healthy breakfast ideas for kids', 'how solar panels work',
          'the best hiking trails near Denver', 'tips for learning Spanish quickly', 'the rules of cricket',
          'current mortgage rates in Canada', 'how to start a vegetable garden', 'the causes of inflation',
          'cheap flights from Boston to Lisbon', 'the plot of Moby Dick', 'reviews of electric scooters',
          'how to repot a cactus', 'upcoming jazz concerts in Chicago', 'the benefits of cold showers',
          'remote work best practices', 'how to file a small claims case', 'the tallest buildings in Asia',
          'kid-friendly museums in London', 'how to brew cold coffee at home', 'symptoms of vitamin D deficiency',
          'the latest news on electric cars', 'how vaccines are tested', 'beginner woodworking projects',
          'the population of Nigeria', 'how to negotiate a salary', 'farmers markets open this Saturday',
          'the meaning of the word serendipity', 'ways to reduce food waste', 'how to train a puppy to sit',
          'the best time to visit Iceland', 'how blockchain works', 'easy vegan dinner recipes',
          'local volunteering opportunities', 'how to apply for a business license in Texas',
          'the life of Marie Curie', 'common interview questions for nurses', 'the weather pattern El Niño',
          'how to clean a cast iron pan', 'popular board games for adults', 'the steps to open a bakery',
          'trends in renewable energy', 'how to fix a leaking tap', 'what causes northern lights',
          'the top-rated sushi places in Seattle', 'how to write a cover letter', 'stretching routines for runners']
SEARCH_ASKS = ['Explore {t}.', 'Discover {t}.', 'Look for {t}.', 'Can you show me {t}?', 'Look up {t}.',
               'Find out about {t}.', 'Research {t} for me.', 'I want to learn about {t}.', 'Search {t}.',
               'Dig up some information on {t}.', 'Show me what you can find on {t}.', 'Get me details on {t}.',
               'Could you look into {t}?', 'Help me find {t}.']


def search_row(rng):
    name = styled(rng, rng.choice(['web_search', 'search_online', 'internet_search', 'search', 'find_online']))
    field = rng.choice(['query', 'q', 'search_terms', 'keywords'])
    tools = [tool(name, rng.choice(['Search the web and show the results.', 'Run an internet search.',
                                    'Look something up online.']), {field: string('What to search for')}, [field])]
    topic = rng.choice(TOPICS)
    query = rng.choice(SEARCH_ASKS).format(t=topic)
    return call_row(query[0].upper() + query[1:], tools, [{'name': name, 'arguments': {field: topic}}], 'search')


# ---------------------------------------------------------------- lookup one field
LOOKUP = {'phone': ['the phone number of {p}', "{p}'s number", 'a number to reach {p}', 'how to call {p}'],
          'email': ["{p}'s email address", 'the email of {p}', 'an address to email {p}'],
          'address': ["{p}'s home address", 'where {p} lives', 'the mailing address for {p}'],
          'link': ['the link to edit {p} in the directory', "{p}'s profile link", 'the record link for {p}']}
LOOKUP_ASKS = ['Get me {x}.', 'I need {x}.', 'Find {x}, please.', 'Can you look up {x}?', 'What is {x}?',
               'Pull {x} from my directory.', 'Set me up with {x}.']


def lookup_row(rng):
    name = styled(rng, rng.choice(['get_person_detail', 'lookup_directory', 'find_person_field', 'get_member_info']))
    key = rng.choice(['field', 'key', 'detail'])
    tools = [tool(name, 'Look up one detail of a person in the saved directory.',
                  {'name': string('Name of the person'),
                   key: {'type': 'string', 'enum': list(LOOKUP), 'description': 'Which detail to return'}},
                  ['name', key])]
    field = rng.choice(list(LOOKUP))
    person = rng.choice([p for p in PEOPLE if not p.startswith(('the ', 'our '))])
    query = rng.choice(LOOKUP_ASKS).format(x=rng.choice(LOOKUP[field]).format(p=person))
    return call_row(query, tools, [{'name': name, 'arguments': {'name': person, key: field}}], 'lookup')


# ---------------------------------------------------------------- timer
LABELS = ['pizza dough', 'the laundry', 'study break', 'the roast chicken', 'my plank workout', 'parking meter',
          'bread proofing', 'meditation', 'the kids screen time', 'tea steeping', 'deep work session', 'the slow cooker']
TIMER_ASKS = ['Set a {d} timer for {l}.', 'Start a countdown of {d} for {l}.', 'Run a timer for {d} — {l}.',
              'Can you time {d} for {l}?', 'Organize a {d} timer for {l}.', 'Begin a {d} countdown for {l}.']


def _duration(rng):
    hours, minutes = rng.choice([(0, rng.choice([1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 45, 50])),
                                 (rng.randint(1, 12), 0), (rng.randint(1, 3), rng.choice([15, 30, 45]))])
    parts, spoken = [], []
    for amount, unit in ((hours, 'hour'), (minutes, 'minute')):
        if amount:
            parts.append(f'{amount} {unit}{"s" if amount > 1 else ""}')
            word = N[amount - 1] if amount <= 12 and rng.random() < 0.5 else str(amount)
            spoken.append(f'{word} {unit}{"s" if amount > 1 else ""}')
    return ' '.join(parts), ' and '.join(spoken)


def timer_row(rng):
    name = styled(rng, rng.choice(['start_timer', 'countdown', 'set_countdown', 'new_timer']))
    duration_field, label_field = rng.choice([('duration', 'label'), ('length', 'name'), ('duration', 'message')])
    tools = [tool(name, 'Start a countdown timer.',
                  {duration_field: string("How long, e.g. '1 hour 30 minutes'"), label_field: string('Optional label')},
                  [duration_field])]
    value, spoken = _duration(rng)
    label = rng.choice(LABELS)
    if rng.random() < 0.25:
        query, arguments = rng.choice(['Set a timer for {d}.', 'Count down {d}.', 'Start a {d} timer.']).format(d=spoken), {
            duration_field: value}
    else:
        query, arguments = rng.choice(TIMER_ASKS).format(d=spoken, l=label), {duration_field: value, label_field: label}
    return call_row(query, tools, [{'name': name, 'arguments': arguments}], 'timer')


# ---------------------------------------------------------------- tools without arguments
BARE = [('list_alarm_sounds', 'Show the sounds that can be used for alarms.',
         ['Which sounds can I use for my alarm?', 'Present the list of alarm sounds.', 'Show me the alarm tones.']),
        ('open_wallpaper_picker', 'Open the wallpaper chooser.',
         ['I want to change my wallpaper.', 'Let me pick a new background picture.', 'Open the wallpaper chooser.']),
        ('show_battery_usage', 'Show which apps use the most battery.',
         ['What is draining my battery?', 'Show battery usage per app.', 'Display my battery stats.']),
        ('scan_qr_code', 'Open the camera to scan a QR code.',
         ['Scan this QR code for me.', 'I need to read a QR code on a poster.', 'Open the QR scanner.']),
        ('take_screenshot', 'Capture the current screen.',
         ['Take a screenshot.', 'Grab what is on my screen right now.', 'Capture the screen, please.']),
        ('record_voice_memo', 'Start recording a voice memo.',
         ['Record a voice note.', 'I want to dictate a quick memo.', 'Start a voice recording.']),
        ('open_storage_manager', 'Show used and free storage.',
         ['How much storage do I have left?', 'Show me what is filling up my phone.', 'Open storage info.'])]


def bare_row(rng):
    base, description, asks = rng.choice(BARE)
    name = styled(rng, base)
    return call_row(rng.choice(asks), [tool(name, description)], [{'name': name, 'arguments': {}}], 'bare')


SHAPES = [(record_row, 0.35), (files_row, 0.2), (search_row, 0.15), (lookup_row, 0.1), (timer_row, 0.1), (bare_row, 0.1)]
FILLER = [tool('get_weather', 'Get the weather forecast for a city.', {'city': string('City name')}, ['city']),
          tool('play_music', 'Play a song, album or artist.', {'query': string('What to play')}, ['query']),
          tool('toggle_flashlight', 'Turn the flashlight on or off.', {'on': {'type': 'boolean'}}, ['on']),
          tool('send_text', 'Send a text message.', {'to': string('Phone number'), 'body': string('Message')}, ['to', 'body']),
          tool('set_volume', 'Set the media volume.', {'level': {'type': 'integer', 'description': '0-100'}}, ['level']),
          tool('get_directions', 'Directions to a place.', {'destination': string('Where to go')}, ['destination']),
          tool('translate_text', 'Translate text.', {'text': string('Text'), 'target': string('Language code')},
               ['text', 'target'])]


def _shape(rng):
    maker = rng.choices([m for m, _ in SHAPES], [w for _, w in SHAPES])[0]
    return maker(rng)


def _pad(row, rng, most=3):
    """Add 0-`most` tools that have nothing to do with the request, so list length is no shortcut."""
    names = {t['name'].lower() for t in row['tools']}
    extra = [t for t in rng.sample(FILLER, rng.randint(0, most)) if t['name'] not in names]
    row['tools'] = row['tools'] + extra
    rng.shuffle(row['tools'])
    return row


def calls(count, seed=11):
    """`count` call rows; about one in seven asks for two of these actions at once."""
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        row = _shape(rng)
        if rng.random() < 0.15:
            second = _shape(rng)
            if {t['name'] for t in second['tools']} & {t['name'] for t in row['tools']}:
                continue
            link = rng.choice([' Also, ', ' Then ', ' And ', ' After that, '])
            b = second['query']
            lower = link.endswith(', ') and not b.startswith(('I ', "I'"))
            row['query'] = row['query'] + link + (b[0].lower() + b[1:] if lower else b)
            row['tools'] += second['tools']
            row['answers'] += second['answers']
        out.append(_pad(row, rng))
    return out


def decline(count, seed=12):
    """`count` no-call rows: an app-style request whose tool is missing; the listed tools are
    other app tools (often from the same shapes), so declining depends on reading them."""
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        wanted, other = _shape(rng), _shape(rng)
        needed = wanted['answers'][0]['name'].lower()
        if any(t['name'].lower() == needed for t in other['tools']) or wanted['source'] == other['source'] == 'app_bare':
            continue
        # Records of another entity read alike but do not do the asked thing; the same entity might.
        entity = next((e for e, _, _ in ENTITIES if e in needed), None)
        if entity and any(entity in t['name'].lower() for t in other['tools']):
            continue
        if wanted['source'] == other['source'] and wanted['source'] in ('app_search', 'app_lookup', 'app_timer', 'app_files'):
            continue
        row = {'query': wanted['query'], 'tools': other['tools'], 'answers': [], 'task': 'no_call',
               'source': wanted['source'] + '_decline', 'teacher': False}
        out.append(_pad(row, rng))
    return out


def generate(calls_count=8000, declines_count=3000, seed=11):
    yield from calls(calls_count, seed)
    yield from decline(declines_count, seed + 1)


SOURCES = {'app_actions': generate}
