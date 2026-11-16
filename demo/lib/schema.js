// A small JSON Schema validator (the subset tool schemas use). It answers one question:
// does this value satisfy this schema?

const typeMatches = (value, type) => {
  switch (type) {
    case 'string': return typeof value === 'string';
    case 'number': return typeof value === 'number' && Number.isFinite(value);
    case 'integer': return typeof value === 'number' && Number.isInteger(value);
    case 'boolean': return typeof value === 'boolean';
    case 'null': return value === null;
    case 'array': return Array.isArray(value);
    case 'object': return value !== null && typeof value === 'object' && !Array.isArray(value);
    default: return false;
  }
};

export const deepEqual = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export function isValid(value, schema) {
  if (schema === true || schema === undefined) return true;
  if (schema === false) return false;
  if ('type' in schema) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((t) => typeMatches(value, t))) return false;
  }
  if ('const' in schema && !deepEqual(value, schema.const)) return false;
  if (schema.enum && !schema.enum.some((v) => deepEqual(v, value))) return false;
  if (typeof value === 'number') {
    if (schema.minimum !== undefined && value < schema.minimum) return false;
    if (schema.maximum !== undefined && value > schema.maximum) return false;
    if (schema.exclusiveMinimum !== undefined && value <= schema.exclusiveMinimum) return false;
    if (schema.exclusiveMaximum !== undefined && value >= schema.exclusiveMaximum) return false;
    if (schema.multipleOf !== undefined && Math.abs(value / schema.multipleOf - Math.round(value / schema.multipleOf)) > 1e-9) return false;
  }
  if (typeof value === 'string') {
    const length = [...value].length;
    if (schema.minLength !== undefined && length < schema.minLength) return false;
    if (schema.maxLength !== undefined && length > schema.maxLength) return false;
    if (schema.pattern !== undefined && !new RegExp(schema.pattern, 'u').test(value)) return false;
  }
  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) return false;
    if (schema.maxItems !== undefined && value.length > schema.maxItems) return false;
    if (schema.uniqueItems && new Set(value.map((v) => JSON.stringify(v))).size !== value.length) return false;
    const prefix = schema.prefixItems || [];
    for (let i = 0; i < value.length; i++) {
      if (!isValid(value[i], i < prefix.length ? prefix[i] : schema.items)) return false;
    }
  }
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    const properties = schema.properties || {};
    for (const key of schema.required || []) if (!(key in value)) return false;
    for (const key of Object.keys(value)) {
      if (Object.hasOwn(properties, key)) {
        if (!isValid(value[key], properties[key])) return false;
      } else if ('additionalProperties' in schema) {
        if (!isValid(value[key], schema.additionalProperties)) return false;
      }
    }
  }
  for (const part of schema.allOf || []) if (!isValid(value, part)) return false;
  if (schema.anyOf && !schema.anyOf.some((s) => isValid(value, s))) return false;
  if (schema.oneOf && schema.oneOf.filter((s) => isValid(value, s)).length !== 1) return false;
  return true;
}
