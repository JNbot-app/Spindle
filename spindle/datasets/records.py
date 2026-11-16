"""Generated extraction rows: a messy text and a record schema, answer = the fields the text states.

Built so the text and the schema do not line up one to one:
  * the text states facts the schema does not ask for,
  * the schema has fields the text does not state (they are left out of the answer),
  * field names, tool names, layouts and value formats vary.
"""
import random

from spindle.convert import call, tool

FIRST = ('Anna Ben Carla David Elena Farid Grace Hugo Ines Jonas Keiko Liam Maya Noah Olga Pedro Quinn Rosa Sam '
         'Tara Umar Vera Wout Xena Yusuf Zoe Amir Bianca Chidi Daria Emeka Freya Goran Hana Ivo Jade').split()
LAST = ('Adams Baker Costa Dubois Evans Fischer Garcia Haddad Ivanov Jansen Kim Lopez Martin Nakamura Okafor '
        'Peeters Quist Rossi Silva Tanaka Usman Visser Weber Xu Yilmaz Zhang Novak Moreau Lindqvist Petrov').split()
COMPANY = ('Northwind Traders|Blue Harbor Logistics|Alder & Finch|Kestrel Software|Lumen Dental|Brightside Catering|'
           'Halcyon Print|Redwood Hardware|Marlow Legal|Nimbus Hosting|Copper Kettle Coffee|Atlas Movers|'
           'Verde Landscaping|Pioneer Tools|Sable Studio|Orchard Health|Tidewater Marine|Quill Stationery|'
           'Ember Heating|Juniper Travel').split('|')
CITY = ('Lisbon Ghent Kraków Austin Osaka Nairobi Toronto Seville Leeds Bergen Denver Porto Lyon Utrecht Tallinn '
        'Dublin Graz Malmö Perth Tampa Bologna Leipzig Cork Split Vilnius').split()
PRODUCT = ('desk lamp|standing desk|wireless mouse|office chair|monitor arm|laptop sleeve|espresso machine|'
           'bike helmet|running shoes|rain jacket|water filter|phone case|noise-cancelling headphones|'
           'cast iron pan|yoga mat|backpack|keyboard|tent|air purifier|drill set').split('|')
VENUE = ('the Riverside Hall|room 204|the main auditorium|Café Aurora|the rooftop terrace|the community centre|'
         'the old library|conference room B|the Harbour Hotel|the city park pavilion').split('|')
EVENT = ('team offsite|quarterly review|book club|product launch|charity run|pottery workshop|alumni dinner|'
         'design sprint|board meeting|open mic night|parents evening|hack day').split('|')
AIRLINE = 'Aerovia|Nordwind Air|Pacific Crest|SkyLark|Meridian Airways|Coastal Jet'.split('|')
MONTH = 'January February March April May June July August September October November December'.split()
ROLE = 'data analyst|nurse|site manager|backend engineer|sales lead|accountant|chef|UX designer'.split('|')


def _name(rng):
    return f'{rng.choice(FIRST)} {rng.choice(LAST)}'


def _date(rng):
    day, month, year = rng.randint(1, 28), rng.randrange(12), rng.choice((2024, 2025, 2026))
    return rng.choice((f'{MONTH[month]} {day}, {year}', f'{day} {MONTH[month]} {year}',
                       f'{year}-{month + 1:02d}-{day:02d}', f'{day:02d}/{month + 1:02d}/{year}',
                       f'{MONTH[month][:3]} {day}'))


def _time(rng):
    hour, minute = rng.randint(6, 21), rng.choice((0, 15, 30, 45))
    return rng.choice((f'{hour:02d}:{minute:02d}', f'{hour % 12 or 12}:{minute:02d} {"am" if hour < 12 else "pm"}',
                       f'{hour % 12 or 12}{"am" if hour < 12 else "pm"}' if not minute else f'{hour}h{minute:02d}'))


def _money(rng):
    value = round(rng.choice((rng.uniform(5, 99), rng.uniform(100, 999), rng.uniform(1000, 9000))), rng.choice((0, 2)))
    value = int(value) if value == int(value) else value
    shown = rng.choice((f'{value:,}', f'{value}'))
    return value, shown


def _email(name, rng):
    first, last = name.lower().split()
    return f'{rng.choice((first, first[0], f"{first}.{last}"))}{rng.choice(("", ".", "_")) if rng.random() < .3 else ""}' \
           f'{last if rng.random() < .6 else ""}@{rng.choice(("mail.com", "example.org", "inbox.net", "work.io"))}'


def _phone(rng):
    digits = [rng.randint(0, 9) for _ in range(9)]
    a, b, c = ''.join(map(str, digits[:3])), ''.join(map(str, digits[3:6])), ''.join(map(str, digits[6:]))
    return rng.choice((f'+{rng.randint(1, 99)} {a} {b} {c}', f'({a}) {b}-{c}', f'0{a} {b} {c}', f'{a}-{b}-{c}'))


def _code(rng, prefix, digits=5):
    return f'{prefix}{rng.choice(("-", "", " "))}{rng.randrange(10 ** (digits - 1), 10 ** digits)}'


# A fact: key -> (field names, JSON type, description, phrasings). `{v}` is the value as shown.
def _invoice(rng):
    total, shown = _money(rng)
    currency = rng.choice(('EUR', 'USD', 'GBP'))
    symbol = {'EUR': '€', 'USD': '$', 'GBP': '£'}[currency]
    return ('record_invoice save_invoice log_bill'.split(), 'Save the details of an invoice.', [
        ('vendor', 'vendor supplier seller'.split(), 'string', 'Company that sent the invoice', rng.choice(COMPANY),
         None, ['Invoice from {v}', 'From: {v}', '{v} has sent you a new invoice.', 'Billed by {v}']),
        ('number', 'invoice_number invoice_id reference'.split(), 'string', 'Invoice number', _code(rng, 'INV'),
         None, ['Invoice no. {v}', 'Invoice #{v}', 'Ref: {v}', 'Please quote {v} when paying.']),
        ('total', 'total amount_due total_amount'.split(), 'number', 'Total amount to pay', total,
         f'{symbol}{shown}' if rng.random() < .5 else f'{shown} {currency}',
         ['Total due: {v}', 'Amount payable: {v}', 'The total comes to {v}.', 'TOTAL {v}']),
        ('currency', ['currency'], ('enum', ['EUR', 'USD', 'GBP']), 'Currency of the amounts', currency, None, []),
        ('issued', 'issue_date invoice_date issued_on'.split(), 'string', 'Date the invoice was issued', _date(rng),
         None, ['Issued: {v}', 'Date of issue {v}', 'Dated {v}.']),
        ('due', 'due_date pay_by payment_due'.split(), 'string', 'Date the invoice has to be paid', _date(rng),
         None, ['Due date: {v}', 'Payment is due by {v}.', 'Please pay before {v}', 'Pay by {v}']),
        ('customer', 'customer bill_to client'.split(), 'string', 'Who the invoice is addressed to', _name(rng),
         None, ['Bill to: {v}', 'Customer: {v}', 'Dear {v},']),
    ], 'currency' if True else None)


def _contact(rng):
    name = _name(rng)
    return ('record_contact save_contact add_person'.split(), "Save a person's contact details.", [
        ('name', 'full_name name contact_name'.split(), 'string', 'Full name of the person', name, None,
         ['{v} here.', 'Name: {v}', 'You can reach {v} any weekday.', 'My name is {v}', '— {v}']),
        ('email', 'email email_address'.split(), 'string', 'Email address', _email(name, rng), None,
         ['Email: {v}', 'drop me a line at {v}', 'e-mail {v}', 'mail: {v}']),
        ('phone', 'phone phone_number mobile'.split(), 'string', 'Phone number', _phone(rng), None,
         ['Phone: {v}', 'call me on {v}', 'tel. {v}', 'mobile {v}']),
        ('company', 'company employer organization'.split(), 'string', 'Company the person works for',
         rng.choice(COMPANY), None, ['Company: {v}', 'I work at {v}.', '{v}', 'on behalf of {v}']),
        ('role', 'job_title role position'.split(), 'string', 'Job title', rng.choice(ROLE), None,
         ['Title: {v}', 'I am the {v} there.', 'Position: {v}']),
        ('city', 'city location based_in'.split(), 'string', 'City where the person is based', rng.choice(CITY), None,
         ['Based in {v}.', 'Location: {v}', 'Our office is in {v}']),
    ], None)


def _event(rng):
    return ('record_event create_event save_event'.split(), 'Save the details of an event.', [
        ('title', 'title event_name name'.split(), 'string', 'Name of the event', rng.choice(EVENT), None,
         ['You are invited to the {v}.', 'Event: {v}', 'Reminder: {v}', 'Join us for the {v}!']),
        ('date', 'date event_date day'.split(), 'string', 'Date of the event', _date(rng), None,
         ['Date: {v}', 'It takes place on {v}.', 'on {v}', 'Save the date: {v}']),
        ('start', 'start_time starts_at begin'.split(), 'string', 'Start time', _time(rng), None,
         ['Starts at {v}.', 'Start: {v}', 'Doors open {v}', 'from {v}']),
        ('end', 'end_time ends_at finish'.split(), 'string', 'End time', _time(rng), None,
         ['Ends at {v}.', 'End: {v}', 'We wrap up around {v}', 'until {v}']),
        ('location', 'location venue place'.split(), 'string', 'Where it takes place', rng.choice(VENUE), None,
         ['Location: {v}', 'We meet at {v}.', 'Venue: {v}', 'Held in {v}']),
        ('organizer', 'organizer host contact_person'.split(), 'string', 'Person who organises it', _name(rng), None,
         ['Organised by {v}.', 'Host: {v}', 'Questions? Ask {v}.', 'RSVP to {v}']),
        ('attendees', 'attendees expected_guests headcount'.split(), 'integer', 'Number of people expected',
         rng.randint(4, 300), None, ['About {v} people are expected.', 'Guests: {v}', 'Capacity {v}']),
    ], None)


def _order(rng):
    price, shown = _money(rng)
    return ('record_order save_order log_purchase'.split(), 'Save the details of an online order.', [
        ('order_id', 'order_id order_number reference'.split(), 'string', 'Order number', _code(rng, 'ORD', 6), None,
         ['Order {v} is confirmed.', 'Order no: {v}', 'Your order ({v}) has shipped.', 'Ref {v}']),
        ('item', 'item product article'.split(), 'string', 'What was ordered', rng.choice(PRODUCT), None,
         ['Item: {v}', 'You bought a {v}.', '1x {v}', 'Product: {v}']),
        ('quantity', 'quantity units count'.split(), 'integer', 'How many were ordered', rng.randint(1, 12), None,
         ['Quantity: {v}', 'Qty {v}', '{v} units', 'Number of items: {v}']),
        ('price', 'price total amount'.split(), 'number', 'Amount paid', price, shown,
         ['Total: {v}', 'You paid {v}.', 'Charged {v} to your card', 'Price: {v}']),
        ('delivery', 'delivery_date arrives_on expected_delivery'.split(), 'string', 'Expected delivery date',
         _date(rng), None, ['Delivery: {v}', 'It should arrive on {v}.', 'Expected {v}', 'Arriving {v}']),
        ('city', 'city ship_to destination'.split(), 'string', 'City it ships to', rng.choice(CITY), None,
         ['Ships to {v}.', 'Destination: {v}', 'Delivery address: 12 Main St, {v}']),
        ('status', ['status'], ('enum', ['processing', 'shipped', 'delivered', 'cancelled']), 'State of the order',
         rng.choice(['processing', 'shipped', 'delivered', 'cancelled']), None,
         ['Status: {v}', 'The order is currently {v}.', 'State: {v}']),
    ], None)


def _flight(rng):
    origin, destination = rng.sample(CITY, 2)
    return ('record_flight save_booking log_trip'.split(), 'Save a flight booking.', [
        ('passenger', 'passenger traveller passenger_name'.split(), 'string', 'Passenger name', _name(rng), None,
         ['Passenger: {v}', 'Dear {v}, your booking is confirmed.', 'Traveller {v}', 'Booked for {v}']),
        ('airline', 'airline carrier'.split(), 'string', 'Airline', rng.choice(AIRLINE), None,
         ['Airline: {v}', 'Thank you for flying {v}.', 'Operated by {v}', 'Carrier {v}']),
        ('flight', 'flight_number flight flight_no'.split(), 'string', 'Flight number',
         f'{rng.choice("ABKLQSTX")}{rng.choice("ACEFHMRZ")}{rng.choice(("", " "))}{rng.randint(100, 9999)}', None,
         ['Flight {v}', 'Flight no. {v}', 'You are on {v}.', 'Flt: {v}']),
        ('origin', 'origin from departure_city'.split(), 'string', 'Departure city', origin, None,
         ['From: {v}', 'Departing {v}', 'Leaves from {v}.', 'Origin {v}']),
        ('destination', 'destination to arrival_city'.split(), 'string', 'Arrival city', destination, None,
         ['To: {v}', 'Arriving in {v}', 'Destination: {v}', 'Lands in {v}.']),
        ('date', 'departure_date date travel_date'.split(), 'string', 'Date of departure', _date(rng), None,
         ['Date: {v}', 'Departure on {v}.', 'Travel date {v}', 'on {v}']),
        ('time', 'departure_time departs_at boarding_time'.split(), 'string', 'Time of departure', _time(rng), None,
         ['Departs {v}', 'Departure time: {v}', 'Take-off at {v}.', 'Leaves {v}']),
        ('seat', 'seat seat_number'.split(), 'string', 'Seat', f'{rng.randint(1, 38)}{rng.choice("ABCDEF")}', None,
         ['Seat {v}', 'Seat: {v}', 'You are in seat {v}.']),
    ], None)


KINDS = (_invoice, _contact, _event, _order, _flight)
OPENERS = ('Hi,', 'Hello!', 'FYI:', 'Quick note.', 'See below.', 'Forwarding this:', '', '', '')
CLOSERS = ('Thanks!', 'Best regards', 'Let me know if anything is off.', 'Sent from my phone', '', '', '')
NOISE = ('Please do not reply to this message.', 'Our office is closed on public holidays.',
         'Keep this for your records.', 'Terms and conditions apply.', 'This is an automated message.',
         'Call 0800 555 019 with any questions.', 'Page 1 of 2', 'Thank you for your business.')


def generate(count, seed=0):
    """Yields (query, tools, answers)."""
    rng = random.Random(seed)
    for _ in range(count):
        names, description, facts, _ = rng.choice(KINDS)(rng)
        stated = [f for f in facts if f[6] and rng.random() < 0.75]            # facts the text states
        if not stated:
            stated = [next(f for f in facts if f[6])]
        asked = [f for f in facts if (f in stated and rng.random() < 0.8) or (f not in stated and rng.random() < 0.45)]
        if not any(f in stated for f in asked):
            asked.append(stated[0])
        rng.shuffle(asked)
        lines = [rng.choice(f[6]).format(v=f[5] if f[5] is not None else f[4]) for f in stated]
        if rng.random() < 0.6:
            rng.shuffle(lines)
        lines += rng.sample(NOISE, rng.randint(0, 2))
        if rng.random() < 0.5:
            lines[1:] = rng.sample(lines[1:], len(lines) - 1)
        body = (rng.choice(('\n', '\n', ' ', ' | '))).join(lines)
        text = ' '.join(part for part in (rng.choice(OPENERS), body, rng.choice(CLOSERS)) if part)
        properties, arguments = {}, {}
        for key, field_names, kind, field_description, value, _, _ in asked:
            field = rng.choice(field_names)
            if field in properties:
                continue
            spec = {'type': 'string', 'enum': kind[1]} if isinstance(kind, tuple) else {'type': kind}
            if rng.random() < 0.8:
                spec['description'] = field_description
            properties[field] = spec
            # An enum fact has no sentence of its own when its phrasings are empty (currency: read off the amount).
            shown = next((f for f in stated if f[0] == key), None) or (
                key == 'currency' and any(f[0] == 'total' for f in stated))
            if shown:
                arguments[field] = value
        if not arguments:
            continue
        name = rng.choice(names)
        yield text, [tool(name, description, {'type': 'object', 'properties': properties})], [call(name, arguments)]
