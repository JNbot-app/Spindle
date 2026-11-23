// Token choice for tool calling: grammar-constrained greedy decoding, copy guard and enum debiasing.
//
// The answer must be `<tool_call>` + a JSON array of {"name": ..., "arguments": ...} objects + `</tool_call>`,
// where every call matches one tool's JSON schema. JsonState follows the JSON byte by byte; the decoder picks
// the most likely token whose bytes keep that state valid. Python and the browser both drive this code
// through the np_decoder_* functions in spindle.h, so they decode the same way.
#include "spindle.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------- JSON values

struct Json;
using JsonPtr = std::shared_ptr<const Json>;

struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    bool boolean = false;
    double number = 0;
    std::string text;  // a string's UTF-8 value, or a number as written
    std::vector<JsonPtr> items;
    std::vector<std::pair<std::string, JsonPtr>> fields;  // in order, keys unique

    const Json *get(const std::string &key) const {
        for (const auto &field : fields)
            if (field.first == key) return field.second.get();
        return nullptr;
    }
    JsonPtr get_ptr(const std::string &key) const {
        for (const auto &field : fields)
            if (field.first == key) return field.second;
        return nullptr;
    }
    bool has(const std::string &key) const { return get(key) != nullptr; }
    void set(const std::string &key, JsonPtr value) {
        for (auto &field : fields)
            if (field.first == key) {
                field.second = std::move(value);
                return;
            }
        fields.emplace_back(key, std::move(value));
    }
    bool is_false() const { return kind == Bool && !boolean; }
};

std::shared_ptr<Json> object() {
    auto result = std::make_shared<Json>();
    result->kind = Json::Object;
    return result;
}

JsonPtr string(const std::string &text) {
    auto result = std::make_shared<Json>();
    result->kind = Json::String;
    result->text = text;
    return result;
}

JsonPtr make_bool(bool value) {
    auto result = std::make_shared<Json>();
    result->kind = Json::Bool;
    result->boolean = value;
    return result;
}

bool equal(const Json &a, const Json &b) {
    if (a.kind != b.kind) return false;
    switch (a.kind) {
    case Json::Null: return true;
    case Json::Bool: return a.boolean == b.boolean;
    case Json::Number: return a.number == b.number;
    case Json::String: return a.text == b.text;
    case Json::Array:
        if (a.items.size() != b.items.size()) return false;
        for (size_t i = 0; i < a.items.size(); ++i)
            if (!equal(*a.items[i], *b.items[i])) return false;
        return true;
    case Json::Object:
        if (a.fields.size() != b.fields.size()) return false;
        for (const auto &field : a.fields) {
            const Json *other = b.get(field.first);
            if (!other || !equal(*field.second, *other)) return false;
        }
        return true;
    }
    return false;
}

void append_utf8(std::string &out, uint32_t code) {
    // Lone surrogates are kept as three bytes (WTF-8) so they can be written back as escapes.
    if (code < 0x80) out += char(code);
    else if (code < 0x800) {
        out += char(0xc0 | (code >> 6));
        out += char(0x80 | (code & 0x3f));
    } else if (code < 0x10000) {
        out += char(0xe0 | (code >> 12));
        out += char(0x80 | ((code >> 6) & 0x3f));
        out += char(0x80 | (code & 0x3f));
    } else {
        out += char(0xf0 | (code >> 18));
        out += char(0x80 | ((code >> 12) & 0x3f));
        out += char(0x80 | ((code >> 6) & 0x3f));
        out += char(0x80 | (code & 0x3f));
    }
}

int hex_value(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Strict UTF-8 check. `partial`: a cut-off last character is fine.
bool utf8_valid(const std::string &bytes, bool partial) {
    size_t i = 0, n = bytes.size();
    while (i < n) {
        const auto b = uint8_t(bytes[i]);
        int need = 0;
        uint8_t low = 0x80, high = 0xbf;
        if (b < 0x80) {
            ++i;
            continue;
        }
        if (b >= 0xc2 && b <= 0xdf) need = 1;
        else if (b >= 0xe0 && b <= 0xef) {
            need = 2;
            if (b == 0xe0) low = 0xa0;
            if (b == 0xed) high = 0x9f;
        } else if (b >= 0xf0 && b <= 0xf4) {
            need = 3;
            if (b == 0xf0) low = 0x90;
            if (b == 0xf4) high = 0x8f;
        } else
            return false;
        for (int k = 1; k <= need; ++k) {
            if (i + k >= n) return partial;
            const auto c = uint8_t(bytes[i + k]);
            if (c < (k == 1 ? low : 0x80) || c > (k == 1 ? high : 0xbf)) return false;
        }
        i += need + 1;
    }
    return true;
}

// The value of the JSON string body `raw` (without quotes), or nothing when it is not valid JSON.
std::optional<std::string> decode_string(const std::string &raw) {
    if (raw.find('\\') == std::string::npos) {
        if (!utf8_valid(raw, false)) return std::nullopt;
        return raw;
    }
    std::string out;
    for (size_t i = 0; i < raw.size(); ++i) {
        const auto c = uint8_t(raw[i]);
        if (c < 32) return std::nullopt;
        if (c != '\\') {
            out += char(c);
            continue;
        }
        if (++i >= raw.size()) return std::nullopt;
        switch (raw[i]) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
            auto unit = [&](size_t at) -> int {
                if (at + 4 > raw.size()) return -1;
                int value = 0;
                for (size_t k = 0; k < 4; ++k) {
                    const int h = hex_value(raw[at + k]);
                    if (h < 0) return -1;
                    value = value * 16 + h;
                }
                return value;
            };
            int code = unit(i + 1);
            if (code < 0) return std::nullopt;
            i += 4;
            if (code >= 0xd800 && code <= 0xdbff && i + 6 < raw.size() && raw[i + 1] == '\\' && raw[i + 2] == 'u') {
                const int low = unit(i + 3);
                if (low >= 0xdc00 && low <= 0xdfff) {
                    code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                    i += 6;
                }
            }
            append_utf8(out, uint32_t(code));
            break;
        }
        default: return std::nullopt;
        }
    }
    // Bytes outside escapes must still form UTF-8 (escaped lone surrogates are allowed, as in Python).
    std::string plain;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\\') {
            i += raw[i + 1] == 'u' ? 5 : 1;
            continue;
        }
        plain += raw[i];
    }
    if (!utf8_valid(plain, false)) return std::nullopt;
    return out;
}

struct Parser {
    const std::string &s;
    size_t i = 0;

    [[noreturn]] void fail() const { throw std::runtime_error("Invalid JSON at byte " + std::to_string(i)); }
    void space() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
    }
    bool literal(const char *word) {
        const size_t n = std::strlen(word);
        if (s.compare(i, n, word) != 0) return false;
        i += n;
        return true;
    }
    std::string string_body() {
        if (s[i] != '"') fail();
        const size_t start = ++i;
        while (i < s.size() && s[i] != '"') i += s[i] == '\\' ? 2 : 1;
        if (i >= s.size()) fail();
        auto value = decode_string(s.substr(start, i - start));
        if (!value) fail();
        ++i;
        return *value;
    }
    JsonPtr value() {
        space();
        if (i >= s.size()) fail();
        auto result = std::make_shared<Json>();
        const char c = s[i];
        if (c == '{') {
            result->kind = Json::Object;
            ++i;
            space();
            if (i < s.size() && s[i] == '}') {
                ++i;
                return result;
            }
            for (;;) {
                space();
                std::string key = string_body();
                space();
                if (i >= s.size() || s[i++] != ':') fail();
                result->set(key, value());
                space();
                if (i >= s.size()) fail();
                if (s[i] == ',') {
                    ++i;
                    continue;
                }
                if (s[i++] != '}') fail();
                return result;
            }
        }
        if (c == '[') {
            result->kind = Json::Array;
            ++i;
            space();
            if (i < s.size() && s[i] == ']') {
                ++i;
                return result;
            }
            for (;;) {
                result->items.push_back(value());
                space();
                if (i >= s.size()) fail();
                if (s[i] == ',') {
                    ++i;
                    continue;
                }
                if (s[i++] != ']') fail();
                return result;
            }
        }
        if (c == '"') {
            result->kind = Json::String;
            result->text = string_body();
            return result;
        }
        if (literal("true")) return make_bool(true);
        if (literal("false")) return make_bool(false);
        if (literal("null")) return result;
        const size_t start = i;
        while (i < s.size() && std::strchr("+-0123456789.eE", s[i])) ++i;
        if (start == i) fail();
        result->kind = Json::Number;
        result->text = s.substr(start, i - start);
        result->number = std::strtod(result->text.c_str(), nullptr);
        return result;
    }
};

JsonPtr parse(const std::string &text) {
    Parser parser{text};
    JsonPtr result = parser.value();
    parser.space();
    if (parser.i != text.size()) parser.fail();
    return result;
}

void dump_string(std::string &out, const std::string &value) {
    out += '"';
    for (size_t i = 0; i < value.size(); ++i) {
        const auto c = uint8_t(value[i]);
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c < 32) {
            char buffer[8];
            std::snprintf(buffer, sizeof buffer, "\\u%04x", c);
            out += buffer;
        } else if (c == 0xed && i + 2 < value.size() && uint8_t(value[i + 1]) >= 0xa0) {  // a lone surrogate
            const uint32_t code = ((c & 0x0f) << 12) | ((uint8_t(value[i + 1]) & 0x3f) << 6) | (uint8_t(value[i + 2]) & 0x3f);
            char buffer[8];
            std::snprintf(buffer, sizeof buffer, "\\u%04x", code);
            out += buffer;
            i += 2;
        } else
            out += char(c);
    }
    out += '"';
}

void dump(std::string &out, const Json &value) {
    switch (value.kind) {
    case Json::Null: out += "null"; break;
    case Json::Bool: out += value.boolean ? "true" : "false"; break;
    case Json::Number: out += value.text; break;
    case Json::String: dump_string(out, value.text); break;
    case Json::Array:
        out += '[';
        for (size_t i = 0; i < value.items.size(); ++i) {
            if (i) out += ',';
            dump(out, *value.items[i]);
        }
        out += ']';
        break;
    case Json::Object:
        out += '{';
        for (size_t i = 0; i < value.fields.size(); ++i) {
            if (i) out += ',';
            dump_string(out, value.fields[i].first);
            out += ':';
            dump(out, *value.fields[i].second);
        }
        out += '}';
        break;
    }
}

// ---------------------------------------------------------------- JSON schema helpers

/** Objects with listed properties allow no extra keys unless the schema says otherwise. */
JsonPtr default_additional_properties(const JsonPtr &schema) {
    if (schema->kind == Json::Array) {
        auto result = std::make_shared<Json>(*schema);
        for (auto &item : result->items) item = default_additional_properties(item);
        return result;
    }
    if (schema->kind != Json::Object) return schema;
    auto result = std::make_shared<Json>(*schema);
    if (const Json *properties = schema->get("properties"); properties && properties->kind == Json::Object) {
        auto copy = std::make_shared<Json>(*properties);
        for (auto &field : copy->fields) field.second = default_additional_properties(field.second);
        result->set("properties", copy);
    }
    if (JsonPtr items = schema->get_ptr("items")) result->set("items", default_additional_properties(items));
    for (const char *key : {"prefixItems", "oneOf", "anyOf", "allOf"})
        if (JsonPtr list = schema->get_ptr(key); list && list->kind == Json::Array)
            result->set(key, default_additional_properties(list));
    for (const char *key : {"$defs", "definitions"})
        if (const Json *defs = schema->get(key); defs && defs->kind == Json::Object) {
            auto copy = std::make_shared<Json>(*defs);
            for (auto &field : copy->fields) field.second = default_additional_properties(field.second);
            result->set(key, copy);
        }
    if (result->has("properties") && !result->has("additionalProperties")) result->set("additionalProperties", make_bool(false));
    if (JsonPtr extra = result->get_ptr("additionalProperties"); extra && extra->kind == Json::Object)
        result->set("additionalProperties", default_additional_properties(extra));
    return result;
}

const Json EMPTY_OBJECT = [] {
    Json value;
    value.kind = Json::Object;
    return value;
}();

/** The alternatives of a schema (oneOf / anyOf branches), each with its allOf parts folded in. */
std::vector<JsonPtr> variants(const JsonPtr &schema) {
    if (schema->kind == Json::Bool) return schema->boolean ? std::vector<JsonPtr>{std::make_shared<Json>(EMPTY_OBJECT)} : std::vector<JsonPtr>{};
    std::vector<JsonPtr> branches;
    if (const Json *list = schema->get("oneOf"); list && list->kind == Json::Array) branches = list->items;
    else if (const Json *any = schema->get("anyOf"); any && any->kind == Json::Array) branches = any->items;
    else branches = {schema};
    std::vector<JsonPtr> result;
    for (const JsonPtr &branch : branches) {
        if (branch->kind == Json::Bool) {
            if (branch->boolean) result.push_back(std::make_shared<Json>(EMPTY_OBJECT));
            continue;
        }
        if (branch->kind != Json::Object) continue;
        auto projected = std::make_shared<Json>(*branch);
        if (const Json *parts = branch->get("allOf"); parts && parts->kind == Json::Array) {
            for (const JsonPtr &raw_part : parts->items) {
                const auto options = variants(raw_part);
                if (options.size() != 1) continue;  // container completion still checks the full schema
                const Json &part = *options[0];
                for (const char *key : {"type", "items", "prefixItems", "enum", "const"})
                    if (!projected->has(key) && part.has(key)) projected->set(key, part.get_ptr(key));
                if (const Json *more = part.get("properties"); more && more->kind == Json::Object) {
                    auto properties = std::make_shared<Json>();
                    properties->kind = Json::Object;
                    if (const Json *own = projected->get("properties"); own && own->kind == Json::Object) properties->fields = own->fields;
                    for (const auto &field : more->fields) {
                        if (JsonPtr old = properties->get_ptr(field.first)) {
                            auto both = std::make_shared<Json>();
                            both->kind = Json::Object;
                            auto list = std::make_shared<Json>();
                            list->kind = Json::Array;
                            list->items = {old, field.second};
                            both->set("allOf", list);
                            properties->set(field.first, both);
                        } else
                            properties->set(field.first, field.second);
                    }
                    projected->set("properties", properties);
                }
                if (part.has("additionalProperties")) projected->set("additionalProperties", part.get_ptr("additionalProperties"));
                auto required = std::make_shared<Json>();
                required->kind = Json::Array;
                std::set<std::string> seen;
                for (const Json *list : {projected->get("required"), part.get("required")})
                    if (list && list->kind == Json::Array)
                        for (const JsonPtr &name : list->items)
                            if (name->kind == Json::String && seen.insert(name->text).second) required->items.push_back(name);
                projected->set("required", required);
            }
        }
        if (projected->has("properties") && !projected->has("additionalProperties")) projected->set("additionalProperties", make_bool(false));
        result.push_back(projected);
    }
    return result;
}

bool has_type(const Json &item, const char *kind) {
    const Json *types = item.get("type");
    if (!types) return false;
    if (types->kind == Json::String) return types->text == kind;
    if (types->kind == Json::Array)
        for (const JsonPtr &t : types->items)
            if (t->kind == Json::String && t->text == kind) return true;
    return false;
}

bool permits(const JsonPtr &schema, const char *kind) {
    for (const JsonPtr &item : variants(schema)) {
        const bool any = !item->has("type");
        if (any || has_type(*item, kind) || (std::strcmp(kind, "number") == 0 && has_type(*item, "integer"))) {
            bool all = true;
            if (const Json *parts = item->get("allOf"); parts && parts->kind == Json::Array)
                for (const JsonPtr &part : parts->items)
                    if (!permits(part, kind)) {
                        all = false;
                        break;
                    }
            if (all) return true;
        }
    }
    return false;
}

bool int_only(const JsonPtr &schema) {
    if (!permits(schema, "integer")) return false;
    for (const JsonPtr &item : variants(schema))
        if (has_type(*item, "number")) return false;
    return true;
}

size_t code_points(const std::string &value) {
    size_t count = 0;
    for (unsigned char c : value) count += (c & 0xc0) != 0x80;
    return count;
}

double number_of(const Json *value, double fallback) {
    return value && value->kind == Json::Number ? value->number : fallback;
}

/** Does `value` satisfy `schema`? The subset of JSON Schema that tool definitions use. */
bool is_valid(const Json &value, const Json &schema) {
    if (schema.kind == Json::Bool) return schema.boolean;
    if (schema.kind != Json::Object) return true;
    if (const Json *types = schema.get("type")) {
        auto matches = [&](const std::string &type) {
            switch (value.kind) {
            case Json::String: return type == "string";
            case Json::Number:
                return type == "number" ? std::isfinite(value.number)
                                        : type == "integer" && std::isfinite(value.number) && std::floor(value.number) == value.number;
            case Json::Bool: return type == "boolean";
            case Json::Null: return type == "null";
            case Json::Array: return type == "array";
            case Json::Object: return type == "object";
            }
            return false;
        };
        bool ok = false;
        if (types->kind == Json::String) ok = matches(types->text);
        else if (types->kind == Json::Array)
            for (const JsonPtr &t : types->items) ok = ok || (t->kind == Json::String && matches(t->text));
        if (!ok) return false;
    }
    if (const Json *constant = schema.get("const"); constant && !equal(value, *constant)) return false;
    if (const Json *options = schema.get("enum"); options && options->kind == Json::Array) {
        bool found = false;
        for (const JsonPtr &option : options->items) found = found || equal(value, *option);
        if (!found) return false;
    }
    if (value.kind == Json::Number) {
        const double x = value.number, inf = std::numeric_limits<double>::infinity();
        if (x < number_of(schema.get("minimum"), -inf) || x > number_of(schema.get("maximum"), inf)) return false;
        if (const Json *m = schema.get("exclusiveMinimum"); m && m->kind == Json::Number && x <= m->number) return false;
        if (const Json *m = schema.get("exclusiveMaximum"); m && m->kind == Json::Number && x >= m->number) return false;
        if (const Json *m = schema.get("multipleOf"); m && m->kind == Json::Number && m->number > 0) {
            const double q = x / m->number;
            if (std::fabs(q - std::round(q)) > 1e-9) return false;
        }
    }
    if (value.kind == Json::String) {
        const double length = double(code_points(value.text)), inf = std::numeric_limits<double>::infinity();
        if (length < number_of(schema.get("minLength"), -inf) || length > number_of(schema.get("maxLength"), inf)) return false;
        if (const Json *pattern = schema.get("pattern"); pattern && pattern->kind == Json::String) {
            try {
                if (!std::regex_search(value.text, std::regex(pattern->text, std::regex::ECMAScript))) return false;
            } catch (const std::regex_error &) {
                // A pattern this regex engine cannot read is not enforced.
            }
        }
    }
    if (value.kind == Json::Array) {
        const double count = double(value.items.size()), inf = std::numeric_limits<double>::infinity();
        if (count < number_of(schema.get("minItems"), -inf) || count > number_of(schema.get("maxItems"), inf)) return false;
        if (const Json *unique = schema.get("uniqueItems"); unique && unique->kind == Json::Bool && unique->boolean)
            for (size_t i = 0; i < value.items.size(); ++i)
                for (size_t j = i + 1; j < value.items.size(); ++j)
                    if (equal(*value.items[i], *value.items[j])) return false;
        const Json *prefix = schema.get("prefixItems");
        const Json *items = schema.get("items");
        for (size_t i = 0; i < value.items.size(); ++i) {
            const Json *child = prefix && prefix->kind == Json::Array && i < prefix->items.size() ? prefix->items[i].get() : items;
            if (child && !is_valid(*value.items[i], *child)) return false;
        }
    }
    if (value.kind == Json::Object) {
        const Json *properties = schema.get("properties");
        if (const Json *required = schema.get("required"); required && required->kind == Json::Array)
            for (const JsonPtr &key : required->items)
                if (key->kind == Json::String && !value.has(key->text)) return false;
        const Json *extra = schema.get("additionalProperties");
        for (const auto &field : value.fields) {
            const Json *child = properties && properties->kind == Json::Object ? properties->get(field.first) : nullptr;
            if (child) {
                if (!is_valid(*field.second, *child)) return false;
            } else if (extra && !is_valid(*field.second, *extra))
                return false;
        }
    }
    if (const Json *parts = schema.get("allOf"); parts && parts->kind == Json::Array)
        for (const JsonPtr &part : parts->items)
            if (!is_valid(value, *part)) return false;
    if (const Json *any = schema.get("anyOf"); any && any->kind == Json::Array) {
        bool ok = false;
        for (const JsonPtr &part : any->items) ok = ok || is_valid(value, *part);
        if (!ok) return false;
    }
    if (const Json *one = schema.get("oneOf"); one && one->kind == Json::Array) {
        int count = 0;
        for (const JsonPtr &part : one->items) count += is_valid(value, *part);
        if (count != 1) return false;
    }
    return true;
}

/** Legal JSON spellings of each character of `value`: (bytes, is a \u escape). */
std::vector<std::vector<std::pair<std::string, bool>>> string_encodings(const std::string &value) {
    std::vector<std::vector<std::pair<std::string, bool>>> result;
    for (size_t i = 0; i < value.size();) {
        const auto b = uint8_t(value[i]);
        const size_t length = b < 0x80 ? 1 : b < 0xe0 ? 2 : b < 0xf0 ? 3 : 4;
        const std::string character = value.substr(i, length);
        i += length;
        uint32_t code = b;
        if (length == 2) code = ((b & 0x1f) << 6) | (uint8_t(character[1]) & 0x3f);
        if (length == 3) code = ((b & 0x0f) << 12) | ((uint8_t(character[1]) & 0x3f) << 6) | (uint8_t(character[2]) & 0x3f);
        if (length == 4)
            code = ((b & 0x07) << 18) | ((uint8_t(character[1]) & 0x3f) << 12) | ((uint8_t(character[2]) & 0x3f) << 6) | (uint8_t(character[3]) & 0x3f);
        char buffer[16];
        std::string escape;
        if (code >= 0x10000) {
            const uint32_t v = code - 0x10000;
            std::snprintf(buffer, sizeof buffer, "\\u%04x\\u%04x", 0xd800 + (v >> 10), 0xdc00 + (v & 0x3ff));
        } else
            std::snprintf(buffer, sizeof buffer, "\\u%04x", code);
        escape = buffer;
        std::vector<std::pair<std::string, bool>> options{{escape, true}};
        static const std::map<uint32_t, std::string> escapes = {{'"', "\\\""}, {'\\', "\\\\"}, {'/', "\\/"}, {'\b', "\\b"},
                                                                {'\f', "\\f"}, {'\n', "\\n"}, {'\r', "\\r"}, {'\t', "\\t"}};
        if (auto found = escapes.find(code); found != escapes.end()) options.emplace_back(found->second, false);
        if (code >= 32 && code != '"' && code != '\\' && !(code >= 0xd800 && code <= 0xdfff)) options.emplace_back(character, false);
        result.push_back(std::move(options));
    }
    return result;
}

bool starts_with(const std::string &text, const std::string &prefix) {
    return prefix.size() <= text.size() && text.compare(0, prefix.size(), prefix) == 0;
}

/** Keep prefixes only when an allowed string can still complete them. */
bool string_prefix_matches(const std::string &raw, const std::vector<std::string> &choices) {
    if (raw.find('\\') == std::string::npos) {
        for (const std::string &choice : choices)
            if (starts_with(choice, raw)) return true;
        return false;
    }
    for (const std::string &choice : choices) {
        std::set<size_t> positions{0};
        for (const auto &encodings : string_encodings(choice)) {
            std::set<size_t> following;
            for (size_t position : positions) {
                const std::string suffix = raw.substr(position);
                for (const auto &[spelling, unicode_escape] : encodings) {
                    std::string fragment = suffix.substr(0, spelling.size());
                    if (unicode_escape)
                        for (char &c : fragment)
                            if (c >= 'A' && c <= 'F') c = char(c + 32);
                    if (starts_with(spelling, fragment)) {
                        if (suffix.size() <= spelling.size()) return true;
                        following.insert(position + spelling.size());
                    }
                }
            }
            positions = std::move(following);
            if (positions.empty()) break;
        }
        if (positions.count(raw.size())) return true;
    }
    return false;
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// Number spellings: `full` needs a complete number, otherwise a prefix that can still become one.
bool number_matches(const std::string &s, bool integer, bool full) {
    size_t i = 0, n = s.size();
    if (i < n && s[i] == '-') ++i;
    if (i == n) return false;
    if (s[i] == '0') ++i;
    else if (s[i] >= '1' && s[i] <= '9')
        while (i < n && is_digit(s[i])) ++i;
    else
        return false;
    if (i == n) return true;
    if (integer) return false;
    if (s[i] == '.') {
        size_t digits = 0;
        for (++i; i < n && is_digit(s[i]); ++i) ++digits;
        if (digits == 0) return !full && i == n;
        if (i == n) return true;
    }
    if (s[i] != 'e' && s[i] != 'E') return false;
    ++i;
    if (i < n && (s[i] == '+' || s[i] == '-')) ++i;
    size_t digits = 0;
    for (; i < n && is_digit(s[i]); ++i) ++digits;
    return i == n && (digits > 0 || !full);
}

// ---------------------------------------------------------------- incremental JSON state

struct Frame {
    JsonPtr schema;
    bool object = false;
    std::shared_ptr<Json> value;  // copied on write: clones share it until one of them changes it
    enum Stage { KeyOrEnd, ValueOrEnd, Key, Colon, Value, Delimiter } stage = KeyOrEnd;
    std::string key;
};

struct JsonState {
    enum Mode { None, String, Number, Literal } mode = None;
    JsonPtr schema;
    std::vector<Frame> stack;
    std::string raw;
    bool escape = false;
    int unicode_remaining = 0;
    JsonPtr scalar_schema;
    bool string_key = false;
    bool done = false;
    JsonPtr value;
    bool last_was_space = false;

    Json &own_value(Frame &frame) {
        if (frame.value.use_count() > 1) frame.value = std::make_shared<Json>(*frame.value);
        return *frame.value;
    }

    std::vector<JsonPtr> effective(const Frame &frame) const {
        auto options = variants(frame.schema);
        if (frame.object) {
            if (const Json *name = frame.value->get("name")) {
                std::vector<JsonPtr> selected;
                for (const JsonPtr &option : options) {
                    const Json *properties = option->get("properties");
                    const Json *wanted = properties ? properties->get("name") : nullptr;
                    const Json *constant = wanted ? wanted->get("const") : nullptr;
                    if (!constant || equal(*constant, *name)) selected.push_back(option);
                }
                if (!selected.empty()) options = std::move(selected);
            }
        }
        return options;
    }

    // Property names still allowed in `frame`, or nothing when any name is allowed.
    std::optional<std::vector<std::string>> available_keys(const Frame &frame) const {
        std::vector<std::string> keys;
        for (const JsonPtr &option : effective(frame)) {
            const Json *extra = option->get("additionalProperties");
            if (extra && !extra->is_false()) return std::nullopt;
            if (!extra && !option->has("properties")) return std::nullopt;  // free-form object: any key
            if (const Json *properties = option->get("properties"); properties && properties->kind == Json::Object)
                for (const auto &field : properties->fields)
                    if (!frame.value->has(field.first) && std::find(keys.begin(), keys.end(), field.first) == keys.end())
                        keys.push_back(field.first);
        }
        return keys;
    }

    JsonPtr expected() const {
        if (stack.empty()) return schema;
        const Frame &frame = stack.back();
        std::vector<JsonPtr> candidates;
        for (const JsonPtr &item : effective(frame)) {
            JsonPtr candidate;
            if (!frame.object) {
                const size_t position = frame.value->items.size();
                const Json *prefix = item->get("prefixItems");
                if (prefix && prefix->kind == Json::Array && position < prefix->items.size()) candidate = prefix->items[position];
                else candidate = item->has("items") ? item->get_ptr("items") : std::make_shared<Json>(EMPTY_OBJECT);
            } else {
                const Json *properties = item->get("properties");
                JsonPtr own = properties && properties->kind == Json::Object ? properties->get_ptr(frame.key) : nullptr;
                candidate = own ? own : item->has("additionalProperties") ? item->get_ptr("additionalProperties")
                          : properties ? make_bool(false) : std::make_shared<Json>(EMPTY_OBJECT);  // free-form: any value
            }
            if (!candidate->is_false()) candidates.push_back(candidate);
        }
        if (candidates.empty()) return nullptr;
        if (candidates.size() == 1) return candidates[0];
        auto any = std::make_shared<Json>();
        any->kind = Json::Object;
        auto list = std::make_shared<Json>();
        list->kind = Json::Array;
        list->items = candidates;
        any->set("anyOf", list);
        return any;
    }

    bool emit(JsonPtr item, const JsonPtr &item_schema) {
        if (!is_valid(*item, *default_additional_properties(item_schema))) return false;
        if (stack.empty()) {
            value = std::move(item);
            done = true;
            return true;
        }
        Frame &frame = stack.back();
        Json &target = own_value(frame);
        if (frame.object) target.set(frame.key, std::move(item));
        else target.items.push_back(std::move(item));
        frame.stage = Frame::Delimiter;
        return true;
    }

    bool close() {
        Frame frame = std::move(stack.back());
        stack.pop_back();
        return emit(frame.value, frame.schema);
    }

    // Take one byte; false when it makes the JSON invalid for the schema.
    bool byte(uint8_t b) {
        if (mode == String) {
            last_was_space = false;
            if (b < 32) return false;
            if (b == '"' && !escape) {
                auto text = unicode_remaining ? std::nullopt : decode_string(raw);
                if (!text) return false;
                mode = None;
                if (string_key) {
                    Frame &frame = stack.back();
                    if (frame.value->has(*text)) return false;
                    frame.key = *text;
                    frame.stage = Frame::Colon;
                    return expected() != nullptr;
                }
                auto item = std::make_shared<Json>();
                item->kind = Json::String;
                item->text = *text;
                return emit(item, scalar_schema);
            }
            if (unicode_remaining) {
                if (hex_value(b) < 0) return false;
                --unicode_remaining;
            } else if (escape) {
                if (!std::strchr("\"\\/bfnrtu", b)) return false;
                unicode_remaining = b == 'u' ? 4 : 0;
                escape = false;
            } else
                escape = b == '\\';
            raw += char(b);
            if (string_key) {
                auto keys = available_keys(stack.back());
                if (keys && !string_prefix_matches(raw, *keys)) return false;
            }
            if (raw.find('\\') == std::string::npos && !utf8_valid(raw, true)) return false;
            if (!string_key) {
                std::vector<std::string> choices;
                bool unconstrained = false;
                for (const JsonPtr &option : variants(scalar_schema)) {
                    if (const Json *values = option->get("enum"); values && values->kind == Json::Array) {
                        for (const JsonPtr &v : values->items)
                            if (v->kind == Json::String) choices.push_back(v->text);
                    } else if (const Json *constant = option->get("const")) {
                        if (constant->kind == Json::String) choices.push_back(constant->text);
                    } else
                        unconstrained = true;
                }
                if (!unconstrained && !string_prefix_matches(raw, choices)) return false;
            }
            return true;
        }
        if (mode == Number) {
            last_was_space = false;
            const bool integer = int_only(scalar_schema);
            if (is_digit(char(b)) || (!integer && std::strchr(".eE+-", b) && b)) {
                raw += char(b);
                return raw == "-" || number_matches(raw, integer, false);
            }
            if (!number_matches(raw, integer, true)) return false;
            mode = None;
            auto item = std::make_shared<Json>();
            item->kind = Json::Number;
            item->text = raw;
            item->number = std::strtod(raw.c_str(), nullptr);
            if (!emit(item, scalar_schema)) return false;
            return byte(b);
        }
        if (mode == Literal) {
            last_was_space = false;
            raw += char(b);
            static const char *words[] = {"true", "false", "null"};
            bool any = false;
            for (const char *word : words) {
                if (!starts_with(word, raw)) continue;
                any = true;
                if (raw == word) {
                    mode = None;
                    JsonPtr item = raw == "null" ? std::make_shared<Json>() : make_bool(raw == "true");
                    return emit(item, scalar_schema);
                }
            }
            return any;
        }
        if (b == ' ' || b == '\r' || b == '\n' || b == '\t') {
            if (b != ' ' || last_was_space) return false;
            last_was_space = true;
            return true;
        }
        last_was_space = false;
        if (done) return false;
        if (!stack.empty()) {
            Frame &frame = stack.back();
            switch (frame.stage) {
            case Frame::Colon:
                if (b != ':') return false;
                frame.stage = Frame::Value;
                return true;
            case Frame::Delimiter:
                if (b == (frame.object ? '}' : ']')) return close();
                if (b != ',') return false;
                if (frame.object) {
                    auto keys = available_keys(frame);
                    if (keys && keys->empty()) return false;
                }
                frame.stage = frame.object ? Frame::Key : Frame::Value;
                return true;
            case Frame::Key:
            case Frame::KeyOrEnd:
                if (b == '}' && frame.stage == Frame::KeyOrEnd) return close();
                if (b != '"') return false;
                if (auto keys = available_keys(frame); keys && keys->empty()) return false;
                mode = String;
                raw.clear();
                escape = false;
                string_key = true;
                return true;
            case Frame::ValueOrEnd:
                if (b == ']') return close();
                break;
            case Frame::Value: break;
            }
        }
        JsonPtr next = expected();
        if (!next) return false;
        if (b == '{' || b == '[') {
            const bool object = b == '{';
            if (!permits(next, object ? "object" : "array")) return false;
            Frame frame;
            frame.schema = next;
            frame.object = object;
            frame.value = std::make_shared<Json>();
            frame.value->kind = object ? Json::Object : Json::Array;
            frame.stage = object ? Frame::KeyOrEnd : Frame::ValueOrEnd;
            stack.push_back(std::move(frame));
        } else if (b == '"') {
            if (!permits(next, "string")) return false;
            mode = String;
            raw.clear();
            escape = false;
            string_key = false;
            scalar_schema = next;
        } else if (b == '-' || is_digit(char(b))) {
            if (!permits(next, "number")) return false;
            mode = Number;
            raw.assign(1, char(b));
            scalar_schema = next;
        } else if (b == 't' || b == 'f' || b == 'n') {
            if (!permits(next, b == 'n' ? "null" : "boolean")) return false;
            mode = Literal;
            raw.assign(1, char(b));
            scalar_schema = next;
        } else
            return false;
        return true;
    }

    bool feed(const std::string &bytes) {
        for (unsigned char b : bytes)
            if (!byte(b)) return false;
        return true;
    }
};

// ---------------------------------------------------------------- decoder

constexpr int MIN_COPIED = 3;     // copy guard: bytes that must already match the request
constexpr int COPY_TOP = 32;      // copy guard: candidate tokens looked at
constexpr int DEBIAS_MIN = 8;     // enum debiasing: requests seen before the correction starts
constexpr int SELECT_TOP = 32;    // tokens tried before sorting the whole vocabulary

void write_error(char *error, size_t size, const std::string &message) {
    if (error && size) {
        std::snprintf(error, size, "%s", message.c_str());
    }
}

}  // namespace

struct np_decoder {
    enum Mode { Before = 0, InJson = 1, Closed = 2, Done = 3 };
    using State = std::pair<JsonState, Mode>;

    // Vocabulary.
    std::vector<std::string> pieces;
    std::vector<uint8_t> special;
    int32_t open_id = -1, close_id = -1, think_id = -1;
    std::set<int32_t> stop_ids;

    // Tools and options.
    JsonPtr root;
    std::unordered_map<std::string, int32_t> label_tokens;  // enum label -> its first token
    std::map<std::string, std::pair<std::vector<double>, int>> priors;  // enum debiasing, kept across requests

    // The current request.
    JsonState json;
    Mode mode = Before;
    bool require_call = false, reasoning = true, debias = true;
    float copy_margin = 4;
    std::string query;
    std::unordered_map<int32_t, std::optional<State>> transitions;
    int steps = 0;
    // Call tracking: which call each generated byte belongs to, and the token log probabilities per call.
    int depth = 0, call = -1;
    bool in_string = false, string_escape = false;
    std::vector<std::vector<double>> log_probs;

    const std::optional<State> &transition(int32_t token) {
        auto found = transitions.find(token);
        if (found != transitions.end()) return found->second;
        std::optional<State> result;
        const bool is_special = special[size_t(token)] != 0;
        if (stop_ids.count(token)) {
            if (mode == Closed) result = State(json, Done);
        } else if (mode == Before) {
            if (token == open_id) result = State(json, InJson);
            else if (token != close_id && !is_special) result = State(json, Before);
        } else if (mode == Closed) {
        } else if (token == close_id) {
            if (json.done) result = State(json, Closed);
        } else if (token != open_id && !is_special) {
            JsonState next = json;
            if (next.feed(pieces[size_t(token)]) &&
                !(require_call && next.done && next.value && next.value->kind == Json::Array && next.value->items.empty()))
                result = State(std::move(next), InJson);
        }
        return transitions.emplace(token, std::move(result)).first->second;
    }

    bool allowed(int32_t token) { return transition(token).has_value(); }

    void advance(int32_t token) {
        const auto &next = transition(token);
        if (!next) throw std::runtime_error("Invalid chosen token");
        State state = *next;
        json = std::move(state.first);
        mode = state.second;
        transitions.clear();
    }

    // Tokens sorted by logit, best first; ties keep the lower id first.
    static std::vector<int32_t> top_tokens(const float *logits, size_t count, size_t k) {
        std::vector<int32_t> order(count);
        std::iota(order.begin(), order.end(), 0);
        auto better = [&](int32_t a, int32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
        k = std::min(k, count);
        std::partial_sort(order.begin(), order.begin() + ptrdiff_t(k), order.end(), better);
        order.resize(k);
        return order;
    }

    int32_t select_token(const float *logits, size_t count) {
        int32_t best = 0;
        for (size_t i = 1; i < count; ++i)
            if (logits[i] > logits[best]) best = int32_t(i);
        if (allowed(best)) return best;
        for (int32_t token : top_tokens(logits, count, SELECT_TOP))
            if (allowed(token)) return token;
        for (int32_t token : top_tokens(logits, count, count))  // rare: sort everything only when needed
            if (allowed(token)) return token;
        throw std::runtime_error("No schema-valid continuation exists");
    }

    // Enum debiasing: the debiased token at the start of an enum value, or -1 to let the grammar choose.
    int32_t debias_select(const float *logits) {
        if (mode != InJson || json.mode != JsonState::String || json.string_key || !json.raw.empty() || json.stack.empty()) return -1;
        std::vector<std::string> labels;
        for (const JsonPtr &option : variants(json.scalar_schema)) {
            std::vector<JsonPtr> values;
            if (const Json *list = option->get("enum"); list && list->kind == Json::Array) values = list->items;
            else if (JsonPtr constant = option->get_ptr("const")) values = {constant};
            else return -1;
            for (const JsonPtr &v : values)
                if (v->kind == Json::String && std::find(labels.begin(), labels.end(), v->text) == labels.end()) labels.push_back(v->text);
        }
        if (labels.size() < 2) return -1;
        std::vector<int32_t> firsts;
        for (const std::string &label : labels) {
            auto found = label_tokens.find(label);
            if (label.empty() || found == label_tokens.end()) return -1;
            if (std::find(firsts.begin(), firsts.end(), found->second) != firsts.end()) return -1;  // two labels share one
            firsts.push_back(found->second);
        }
        for (int32_t token : firsts)
            if (!allowed(token)) return -1;
        const Frame &frame = json.stack.back();
        std::string key = frame.object ? "k:" + frame.key : std::string("-");
        for (const std::string &label : labels) key += '\x01' + label;
        std::vector<double> p(firsts.size());
        double top = -std::numeric_limits<double>::infinity(), sum = 0;
        for (size_t i = 0; i < firsts.size(); ++i) top = std::max(top, double(logits[firsts[i]]));
        for (size_t i = 0; i < firsts.size(); ++i) sum += p[i] = std::exp(double(logits[firsts[i]]) - top);
        for (double &v : p) v /= sum;
        auto &total = priors.try_emplace(key, std::vector<double>(firsts.size(), 0.0), 0).first->second;
        int32_t choice = -1;
        if (total.second >= DEBIAS_MIN) {
            double best = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < p.size(); ++i) {
                const double score = std::log(p[i]) - std::log(std::max(total.first[i] / total.second, 1e-6));
                if (score > best) {
                    best = score;
                    choice = firsts[i];
                }
            }
        }
        for (size_t i = 0; i < p.size(); ++i) total.first[i] += p[i];  // this request joins the prior after its own choice
        ++total.second;
        return choice;
    }

    // Copy guard: `chosen`, or a token that keeps copying the request word for word.
    int32_t copy_guard(const float *logits, size_t count, int32_t chosen) {
        const std::string &raw = json.raw;
        if (mode != InJson || json.mode != JsonState::String || json.string_key || int(raw.size()) < MIN_COPIED ||
            raw.find('\\') != std::string::npos || query.find(raw) == std::string::npos)
            return chosen;
        const std::string &piece = pieces[size_t(chosen)];
        if ((!piece.empty() && piece[0] == '"') || query.find(raw + piece) != std::string::npos) return chosen;
        const float best = logits[chosen];
        for (int32_t token : top_tokens(logits, count, COPY_TOP)) {
            if (logits[token] < best - copy_margin) break;
            const std::string &candidate = pieces[size_t(token)];
            const bool blank = candidate.find_first_not_of(" \t\n\r\x0b\x0c") == std::string::npos;
            if (!blank && candidate.find('"') == std::string::npos && query.find(raw + candidate) != std::string::npos && allowed(token))
                return token;
        }
        return chosen;
    }

    // log p(token) renormalised over the grammar-valid tokens among the top 32 candidates.
    double constrained_log_probability(const float *logits, size_t count, int32_t token) {
        double top = -std::numeric_limits<double>::infinity(), sum = 0;
        for (size_t i = 0; i < count; ++i) top = std::max(top, double(logits[i]));
        for (size_t i = 0; i < count; ++i) sum += std::exp(double(logits[i]) - top);
        const double full = double(logits[token]) - top - std::log(sum);
        if (full > -1e-4) return full;
        std::vector<int32_t> close;
        for (size_t i = 0; i < count; ++i)
            if (double(logits[i]) >= double(logits[token]) - 12.0) close.push_back(int32_t(i));
        if (close.size() > 32) {
            std::partial_sort(close.begin(), close.begin() + 32, close.end(),
                              [&](int32_t a, int32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
            close.resize(32);
        }
        std::vector<double> values{double(logits[token])};
        for (int32_t t : close)
            if (t != token && allowed(t)) values.push_back(double(logits[t]));
        const double peak = *std::max_element(values.begin(), values.end());
        double total = 0;
        for (double v : values) total += std::exp(v - peak);
        return double(logits[token]) - peak - std::log(total);
    }

    // Which calls the bytes of a token touch (a call is a depth-2 object of the answer array).
    std::set<int> track(const std::string &bytes) {
        std::set<int> touched;
        for (unsigned char b : bytes) {
            if (in_string) {
                if (string_escape) string_escape = false;
                else if (b == '\\') string_escape = true;
                else if (b == '"') in_string = false;
            } else if (b == '"')
                in_string = true;
            else if (b == '{' || b == '[') {
                ++depth;
                if (b == '{' && depth == 2) ++call;
            } else if (b == '}' || b == ']') {
                if (depth >= 2) touched.insert(call);
                --depth;
                continue;
            }
            if (depth >= 2) touched.insert(call);
        }
        return touched;
    }

    int32_t step(const float *logits, size_t count) {
        if (count != pieces.size()) throw std::runtime_error("Invalid decoder logits");
        for (size_t i = 0; i < count; ++i)
            if (!std::isfinite(logits[i])) throw std::runtime_error("Invalid decoder logits");
        int32_t token;
        if (steps == 0) {  // the answer starts with a note (when the model prefers one) or the calls
            token = reasoning && think_id >= 0 && logits[think_id] > logits[open_id] ? think_id : open_id;
        } else {
            token = debias ? debias_select(logits) : -1;
            if (token < 0) token = select_token(logits, count);
            if (copy_margin > 0) token = copy_guard(logits, count, token);
        }
        if (mode == InJson && token != close_id) {
            const double value = constrained_log_probability(logits, count, token);
            for (int index : track(pieces[size_t(token)])) {
                if (index < 0) continue;
                if (log_probs.size() <= size_t(index)) log_probs.resize(size_t(index) + 1);
                log_probs[size_t(index)].push_back(value);
            }
        }
        advance(token);
        ++steps;
        return token;
    }

    std::string result() const {
        std::string out = "{\"complete\":";
        const bool complete = mode == Done;
        out += complete ? "true" : "false";
        out += ",\"calls\":";
        size_t calls = 0;
        if (complete && json.value) {
            dump(out, *json.value);
            calls = json.value->items.size();
        } else
            out += "[]";
        out += ",\"probabilities\":[";
        for (size_t i = 0; i < calls; ++i) {
            double mean = 0;
            if (i < log_probs.size() && !log_probs[i].empty()) {
                for (double v : log_probs[i]) mean += v;
                mean /= double(log_probs[i].size());
            }
            char buffer[32];
            std::snprintf(buffer, sizeof buffer, "%s%.17g", i ? "," : "", std::exp(mean));
            out += buffer;
        }
        out += "]}";
        return out;
    }
};

extern "C" {

np_decoder *np_decoder_new(const uint8_t *pieces, const uint32_t *offsets, const uint8_t *special, size_t count,
                           int32_t open_id, int32_t close_id, int32_t think_id, const int32_t *stop_ids, size_t stop_count,
                           char *error, size_t error_size) {
    try {
        if (!pieces || !offsets || !special || !count || open_id < 0 || close_id < 0 || size_t(open_id) >= count ||
            size_t(close_id) >= count || (think_id >= 0 && size_t(think_id) >= count))
            throw std::runtime_error("Invalid decoder vocabulary");
        auto decoder = std::make_unique<np_decoder>();
        decoder->pieces.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (offsets[i + 1] < offsets[i]) throw std::runtime_error("Invalid piece offsets");
            decoder->pieces.emplace_back(reinterpret_cast<const char *>(pieces) + offsets[i], offsets[i + 1] - offsets[i]);
        }
        decoder->special.assign(special, special + count);
        decoder->open_id = open_id;
        decoder->close_id = close_id;
        decoder->think_id = think_id;
        for (size_t i = 0; i < stop_count; ++i) decoder->stop_ids.insert(stop_ids[i]);
        decoder->root = parse("{\"type\":\"array\",\"items\":false}");
        return decoder.release();
    } catch (const std::exception &e) {
        write_error(error, error_size, e.what());
        return nullptr;
    }
}

void np_decoder_free(np_decoder *decoder) { delete decoder; }

int32_t np_decoder_set_tools(np_decoder *decoder, const char *tools_json, const char *labels_json, char *error, size_t error_size) {
    try {
        if (!decoder || !tools_json) throw std::runtime_error("Missing decoder or tools");
        JsonPtr tools = parse(tools_json);
        if (tools->kind != Json::Array) throw std::runtime_error("Tools must be a JSON list");
        // [{"name": <tool>, "arguments": <its parameters>}, ...]: one alternative per tool.
        auto alternatives = std::make_shared<Json>();
        alternatives->kind = Json::Array;
        for (const JsonPtr &tool : tools->items) {
            const JsonPtr name = tool->get_ptr("name");
            const JsonPtr parameters = tool->get_ptr("parameters");
            if (!name || name->kind != Json::String || !parameters) throw std::runtime_error("Each tool needs a name and parameters");
            auto name_schema = object();
            name_schema->set("const", name);
            auto properties = object();
            properties->set("name", name_schema);
            properties->set("arguments", parameters);
            auto call = object();
            call->set("type", string("object"));
            call->set("properties", properties);
            call->set("required", parse("[\"name\",\"arguments\"]"));
            call->set("additionalProperties", make_bool(false));
            alternatives->items.push_back(call);
        }
        auto items = object();
        items->set("oneOf", alternatives);
        auto schema = object();
        schema->set("type", string("array"));
        schema->set("items", alternatives->items.empty() ? make_bool(false) : JsonPtr(items));
        decoder->root = default_additional_properties(schema);
        decoder->label_tokens.clear();
        if (labels_json && *labels_json) {
            JsonPtr labels = parse(labels_json);
            for (const auto &field : labels->fields)
                if (field.second->kind == Json::Number) decoder->label_tokens[field.first] = int32_t(field.second->number);
        }
        return 0;
    } catch (const std::exception &e) {
        write_error(error, error_size, e.what());
        return -1;
    }
}

int32_t np_decoder_start(np_decoder *decoder, const char *query, size_t query_size, int32_t flags, float copy_margin,
                         char *error, size_t error_size) {
    if (!decoder) {
        write_error(error, error_size, "Missing decoder");
        return -1;
    }
    decoder->json = JsonState();
    decoder->json.schema = decoder->root;
    decoder->mode = np_decoder::Before;
    decoder->require_call = flags & NP_DECODER_REQUIRE_CALL;
    decoder->reasoning = flags & NP_DECODER_REASONING;
    decoder->debias = flags & NP_DECODER_DEBIAS;
    decoder->copy_margin = copy_margin;
    decoder->query.assign(query ? query : "", query ? query_size : 0);
    decoder->transitions.clear();
    decoder->steps = 0;
    decoder->depth = 0;
    decoder->call = -1;
    decoder->in_string = decoder->string_escape = false;
    decoder->log_probs.clear();
    return 0;
}

int32_t np_decoder_step(np_decoder *decoder, const float *logits, size_t count, char *error, size_t error_size) {
    try {
        if (!decoder || !logits) throw std::runtime_error("Missing decoder or logits");
        if (decoder->mode == np_decoder::Done) throw std::runtime_error("The answer is already complete");
        return decoder->step(logits, count);
    } catch (const std::exception &e) {
        write_error(error, error_size, e.what());
        return -1;
    }
}

int32_t np_decoder_state(np_decoder *decoder) { return decoder ? int32_t(decoder->mode) : -1; }

int64_t np_decoder_result(np_decoder *decoder, char *buffer, size_t size) {
    if (!decoder) return -1;
    const std::string text = decoder->result();
    if (buffer && size) {
        const size_t n = std::min(size - 1, text.size());
        std::memcpy(buffer, text.data(), n);
        buffer[n] = 0;
    }
    return int64_t(text.size());
}

}  // extern "C"
