// Everything the demo shows: five tasks, each with tool definitions and example inputs.
// A "task" is just a set of tools plus a way to display the calls the model returns.

const str = (description, extra = {}) => ({ type: 'string', description, ...extra });
const int = (description, extra = {}) => ({ type: 'integer', description, ...extra });
const num = (description, extra = {}) => ({ type: 'number', description, ...extra });
const bool = (description) => ({ type: 'boolean', description });
const tool = (name, description, properties, required = []) => ({
  name, description, parameters: { type: 'object', properties, required },
});

// ---------------------------------------------------------------- tool calling
const smartHome = [
  tool('set_light', 'Turn a light on or off, optionally dimming it.',
    { room: str('Room name'), state: str('', { enum: ['on', 'off'] }), brightness: int('Brightness in percent, 0-100') },
    ['room', 'state']),
  tool('set_thermostat', 'Set the target temperature of the heating.',
    { temperature: num('Target temperature'), unit: str('', { enum: ['celsius', 'fahrenheit'] }) }, ['temperature']),
  tool('lock_door', 'Lock or unlock a door.', { door: str('Which door'), action: str('', { enum: ['lock', 'unlock'] }) }, ['door', 'action']),
  tool('play_music', 'Play music by an artist, song or genre.', { query: str('Artist, song or genre'), room: str('Room to play in') }, ['query']),
  tool('set_timer', 'Start a countdown timer.', { minutes: int('Length in minutes'), label: str('Optional name for the timer') }, ['minutes']),
];

const phone = [
  tool('send_message', 'Send a text message to a contact.', { contact: str('Name of the contact'), text: str('Message text') }, ['contact', 'text']),
  tool('set_alarm', 'Set an alarm.', { time: str('Time of day, e.g. 07:30'), label: str('Optional name for the alarm') }, ['time']),
  tool('get_weather', 'Get the current weather for a city.', { city: str('City name') }, ['city']),
  tool('create_reminder', 'Create a reminder.', { text: str('What to be reminded of'), time: str('When to remind, e.g. tomorrow 9am') }, ['text']),
  tool('call_contact', 'Start a phone call with a contact.', { contact: str('Name of the contact') }, ['contact']),
];

const travel = [
  tool('search_flights', 'Search for flights between two cities.',
    { origin: str('Departure city'), destination: str('Arrival city'), date: str('Departure date'), passengers: int('Number of passengers') },
    ['origin', 'destination']),
  tool('book_hotel', 'Book a hotel room.',
    { city: str('City'), check_in: str('Check-in date'), nights: int('Number of nights'), guests: int('Number of guests') }, ['city', 'nights']),
  tool('rent_car', 'Rent a car.', { city: str('Pick-up city'), days: int('Number of days'), type: str('Car type', { enum: ['economy', 'suv', 'luxury'] }) }, ['city', 'days']),
  tool('get_exchange_rate', 'Get the exchange rate between two currencies.', { from: str('Currency code, e.g. EUR'), to: str('Currency code, e.g. USD') }, ['from', 'to']),
];

const strings = (description) => ({ type: 'array', items: { type: 'string' }, description });

const work = [
  tool('send_email', 'Send an email.', { to: str('Recipient name or address'), subject: str('Subject line'), body: str('Email text') }, ['to', 'subject']),
  tool('send_message', 'Send a chat message to a colleague.', { contact: str('Name of the colleague'), text: str('Message text') }, ['contact', 'text']),
  tool('create_event', 'Add a meeting or event to the calendar.',
    { title: str('Event title'), date: str('Date'), time: str('Start time'), attendees: strings('People to invite') }, ['title', 'date']),
  tool('create_reminder', 'Create a personal reminder.', { text: str('What to be reminded of'), time: str('When to remind') }, ['text']),
  tool('search_files', 'Search documents by keywords.',
    { query: str('Keywords'), file_type: str('File type', { enum: ['pdf', 'doc', 'sheet', 'slides'] }) }, ['query']),
  tool('translate_text', 'Translate text into another language.', { text: str('Text to translate'), target_language: str('Language to translate into') }, ['text', 'target_language']),
  tool('convert_currency', 'Convert an amount between currencies.', { amount: num('Amount'), from: str('Currency code, e.g. EUR'), to: str('Currency code, e.g. USD') }, ['amount', 'from', 'to']),
  tool('calculate_tip', 'Split a bill and add a tip.', { total: num('Bill total'), people: int('Number of people'), tip_percent: num('Tip in percent') }, ['total']),
  tool('set_status', 'Set your chat status.',
    { status: str('', { enum: ['available', 'busy', 'away', 'do_not_disturb'] }), message: str('Optional status text') }, ['status']),
];

// ---------------------------------------------------------------- extraction
const record = (name, description, properties, required = []) => [tool(name, description, properties, required)];

const eventRecord = record('record_event', 'Save the details of an event from an invitation.', {
  title: str('Name of the event'),
  date: str('Date of the event'),
  start_time: str('Start time'),
  end_time: str('End time'),
  location: str('Where it takes place'),
  organizer: str('Person who organises it'),
}, ['title']);

const invoiceRecord = record('record_invoice', 'Save the details of an invoice.', {
  vendor: str('Company that sent the invoice'),
  invoice_number: str('Invoice number'),
  issue_date: str('Date the invoice was issued'),
  due_date: str('Date the invoice has to be paid'),
  total: num('Total amount to pay'),
  currency: str('Currency code', { enum: ['EUR', 'USD', 'GBP'] }),
}, ['vendor', 'total']);

const contactRecord = record('record_contact', 'Save a person\'s contact details.', {
  full_name: str('Full name of the person'),
  email: str('Email address'),
  phone: str('Phone number'),
  company: str('Company name'),
}, ['full_name']);

const flightRecord = record('record_flight', 'Save a flight booking from a confirmation email.', {
  passenger: str('Passenger name'),
  airline: str('Airline'),
  flight_number: str('Flight number'),
  origin: str('Departure city or airport'),
  destination: str('Arrival city or airport'),
  departure_date: str('Date of departure'),
  departure_time: str('Time of departure'),
}, ['passenger', 'flight_number']);

// ---------------------------------------------------------------- sentiment
const sentiment = [tool('classify_sentiment', 'Classify the sentiment of a text.', {
  sentiment: str('Overall sentiment of the text', { enum: ['positive', 'negative', 'neutral'] }),
}, ['sentiment'])];

// ---------------------------------------------------------------- search filters
const products = [tool('search_products', 'Search the shop with filters.', {
  category: str('Product category', { enum: ['shoes', 'laptops', 'phones', 'jackets', 'headphones'] }),
  max_price: num('Maximum price in euros'),
  color: str('Color'),
  sort_by: str('Sort order', { enum: ['cheapest', 'most_expensive', 'best_rated', 'newest'] }),
}, ['category'])];

const restaurants = [tool('find_restaurants', 'Find restaurants with filters.', {
  city: str('City'),
  cuisine: str('Kind of food, e.g. italian or thai'),
  max_distance_km: num('Maximum distance in kilometers'),
  open_now: bool('Only places that are open right now'),
  vegetarian: bool('Only places with vegetarian options'),
}, ['city'])];

const e = (label, text) => ({ label, text });

export const TASKS = [
  {
    id: 'tools',
    title: 'Tool calling',
    blurb: 'Pick the right tool and fill in its arguments. Ask for two things and you get two calls, in order.',
    mode: 'calls',
    sets: [
      { name: 'Smart home', tools: smartHome, examples: [
        e('Dim a light', 'Dim the living room lights to 30 percent'),
        e('Two actions', 'Lock the front door and turn off the kitchen light'),
        e('Music', 'Play some jazz in the bedroom'),
        e('Temperature', 'Set the heating to 21 degrees celsius'),
        e('Timer', 'Start a 15 minute timer called laundry'),
        e('Unlock', 'Unlock the back door'),
        e('Three actions', 'Turn off the lights in the kitchen and the hallway, then lock the front door'),
        e('Indirect', "It's way too cold in here, put the heating on 23"),
      ] },
      { name: 'Phone assistant', tools: phone, examples: [
        e('Message', 'Text Anna that I will be ten minutes late'),
        e('Alarm', 'Set an alarm for 7:30 called gym'),
        e('Weather', 'Do I need an umbrella in Paris today?'),
        e('Reminder', 'Remind me to buy milk tomorrow at 9am'),
        e('Call', 'Call my brother Tom'),
        e('Wake up', 'Wake me up at 6:15 tomorrow for the airport'),
        e('Two people', 'Ring my sister Eva and text Jonas that dinner is at 8'),
        e('Text + reminder', "Tell Mia I'm running late and remind me to buy flowers at 6pm"),
        e('No fitting tool', 'Take a photo'),
      ] },
      { name: 'Travel', tools: travel, examples: [
        e('Flights', 'Find a flight from Brussels to Lisbon on March 3rd for 2 people'),
        e('Hotel + car', 'Book a hotel in Rome for 4 nights and rent a luxury car for 3 days there'),
        e('Exchange rate', 'What is the EUR to USD exchange rate?'),
        e('Car rental', 'Rent an SUV in Denver for 5 days'),
        e('No fitting tool', 'Book a train to Paris for Friday'),
      ] },
      { name: 'Work assistant', tools: work, examples: [
        e('Find + email', 'Find the budget spreadsheet and email it to Omar with subject Budget'),
        e('Chat', 'Ping Tom on chat that the build is green'),
        e('Reminder + chat', "Remind me to send the invoice on Friday and let Jules know on chat that it's coming"),
        e('Files', 'Find the onboarding slides'),
        e('Translate', "Translate 'where is the train station' into Japanese"),
        e('Split a bill', 'Split a 84.50 bill between 3 people with a 15% tip'),
        e('No fitting tool', 'Delete all my emails from last week'),
      ] },
    ],
  },
  {
    id: 'extract',
    title: 'Extraction',
    blurb: 'Pull structured fields (names, dates, amounts) out of messy text. The output is guaranteed to be valid JSON that matches the schema.',
    mode: 'record',
    sets: [
      { name: 'Event invitation', tools: eventRecord, examples: [
        e('Team offsite', 'Hi all, Sarah Mitchell here. Join us for the Q3 Planning Offsite on 14 October 2026, from 9:30 to 16:00 at the Harbour Hotel in Antwerp. Lunch is provided.'),
        e('Annual meeting', 'Please join the Annual General Meeting on 5 November 2026, 14:00 to 16:30, in the Grand Hall, Rotterdam. Chair: Priya Nair.'),
        e('Webinar', 'Reminder: "Intro to Rust for Python Developers" webinar, hosted by David Kim, on 3 December 2026 from 18:00 to 19:30 at the Ghent Tech Hub.'),
      ] },
      { name: 'Invoice', tools: invoiceRecord, examples: [
        e('Design studio', 'Blue Pixel Studio - invoice no. BP-7731 - 12 Jan 2026. Logo design and brand guide. Amount due: USD 2000 by 11 Feb 2026.'),
        e('Hosting', 'ACME Hosting Ltd, issued 2026-03-02. Cloud servers for March. Total due: 480.00 EUR, to be paid by 2026-03-30.'),
        e('Utilities', 'GreenPower, 1 May 2026. Electricity for April. Amount due: 142.55 GBP by 15 May 2026.'),
      ] },
      { name: 'Contact', tools: contactRecord, examples: [
        e('Email signature', 'Best regards,\nJulia Fernandez, Head of Marketing at Northwind Traders\nEmail: julia.fernandez@northwind.example\nPhone: +32 470 12 34 56'),
        e('Short intro', 'Contact me: Sofia Lindqvist, sofia@lindqvist.example, 070-123 45 67'),
        e('Chat message', 'hey, you should talk to Liam O\'Connor at Brightwave. his phone number is 0471 98 76 54 and mail is liam@brightwave.example'),
      ] },
      { name: 'Flight booking', tools: flightRecord, examples: [
        e('Confirmation', 'Booking confirmed! Passenger: Noah Peeters. Brussels Airlines SN 3121 from Brussels (BRU) to Barcelona (BCN), departing 22 May 2026 at 07:45.'),
        e('Short note', 'Your flight: KL1234 Amsterdam to New York JFK, 3 September 2026 at 11:20, passenger Emma Jansen, operated by KLM.'),
      ] },
    ],
  },
  {
    id: 'sentiment',
    title: 'Sentiment',
    blurb: 'Classification is extraction with a closed set of answers: the model can only reply with one of the allowed labels.',
    mode: 'record',
    sets: [
      { name: 'Sentiment', tools: sentiment, examples: [
        e('Happy review', 'Absolutely love this vacuum! It picked up everything and the battery lasts for hours. Best purchase of the year.'),
        e('Angry review', 'Terrible service. The package arrived two weeks late and the box was crushed. Customer service never replied. Never ordering again.'),
        e('Neutral', 'The meeting is scheduled for Tuesday at 10am in room 4B.'),
        e('Sarcasm', 'Oh wonderful, the update deleted all my settings again. Just what I needed today, terrible.'),
        e('App update', 'Loving the new update, everything feels so much faster!'),
        e('Restaurant', 'Waited 45 minutes for cold pasta and the waiter was rude. Not coming back.'),
        e('Movie', 'The plot was predictable but the soundtrack and visuals were stunning.'),
        e('Support', 'Thanks for the quick fix, the problem is solved and everything works perfectly now.'),
      ] },
    ],
  },
  {
    id: 'search',
    title: 'Search filters',
    blurb: 'Turn a plain-language search into the structured filters your database or API expects.',
    mode: 'record',
    sets: [
      { name: 'Online shop', tools: products, examples: [
        e('Shoes', 'cheapest black running shoes below 80 euro'),
        e('Laptop', 'show me the best rated laptops below 900 euro'),
        e('Jacket', 'newest jackets under 120 euros'),
        e('Headphones', 'white headphones under 150 euros, best rated first'),
        e('Phone', 'cheapest phones under 300 euro'),
      ] },
      { name: 'Restaurants', tools: restaurants, examples: [
        e('Open now', 'italian food in Ghent, open right now'),
        e('Vegetarian', 'vegetarian places in Utrecht within 2 km'),
        e('Chinese', 'chinese food in Leuven that is open now'),
        e('Combined', 'thai food in Brussels, open now, vegetarian, max 3 km away'),
      ] },
    ],
  },
  {
    id: 'custom',
    title: 'Your own tools',
    blurb: 'Write your own tool definitions (JSON) and a request. The model sees exactly what you write here.',
    mode: 'calls',
    editable: true,
    sets: [
      { name: 'Custom', tools: [
        tool('create_event', 'Add an event to the calendar.', { title: str('Event title'), date: str('Date'), time: str('Time') }, ['title']),
        tool('send_email', 'Send an email.', { to: str('Recipient'), subject: str('Subject'), body: str('Email body') }, ['to', 'subject']),
      ], examples: [
        e('Calendar', 'Put dentist appointment on my calendar for next Friday at 2pm'),
        e('Email', 'Email Priya with subject "Slides" and tell her they are attached'),
      ] },
    ],
  },
];
