"""Turn the loose tool schemas of public datasets into plain JSON Schema tool definitions."""

TYPES = {'dict': 'object', 'float': 'number', 'double': 'number', 'int': 'integer', 'long': 'integer',
         'str': 'string', 'bool': 'boolean', 'list': 'array', 'tuple': 'array', 'arraylist': 'array',
         'hashmap': 'object', 'char': 'string', 'bigint': 'integer', 'any': None, 'set': 'array',
         'object': 'object', 'string': 'string', 'integer': 'integer', 'number': 'number',
         'boolean': 'boolean', 'array': 'array', 'null': 'null'}
KEEP = ('description', 'enum', 'required')


def json_type(name):
    """'str', 'STRING', 'List[int]', 'int, optional', 'Optional[Dict[str, Any]]' -> a JSON Schema type
    (None when unknown)."""
    name = str(name).lower().strip()
    if name.startswith('optional[') and name.endswith(']'):
        name = name[9:-1]
    name = name.split(',')[0].strip()
    return TYPES.get(name.split('[')[0].strip())


def to_schema(node, strings_only=False):
    """A JSON Schema from the dialects public sets use (dict, float, STRING, ...)."""
    if not isinstance(node, dict):
        return {}
    out = {k: node[k] for k in KEEP if k in node and (k != 'required' or isinstance(node[k], list))}
    kind = json_type(node['type']) if isinstance(node.get('type'), str) else None
    if strings_only and kind not in ('object', None):
        return {**{k: v for k, v in out.items() if k == 'description'}, 'type': 'string'}
    if kind:
        out['type'] = kind
    if isinstance(node.get('properties'), dict):
        out['properties'] = {k: to_schema(v, strings_only) for k, v in node['properties'].items()}
        out.setdefault('type', 'object')
    if isinstance(node.get('items'), dict):
        out['items'] = to_schema(node['items'], strings_only)
    if 'enum' in out and not (isinstance(out['enum'], list) and out['enum']):
        del out['enum']
    if not isinstance(out.get('description', ''), str):
        del out['description']
    return out


def tool(name, description, parameters, strings_only=False):
    schema = to_schema(parameters or {'type': 'object', 'properties': {}}, strings_only)
    schema['type'] = 'object'
    schema.setdefault('properties', {})
    return {'name': name, 'description': (description or '').strip(), 'parameters': schema}


def call(name, arguments):
    return {'name': name, 'arguments': arguments}
