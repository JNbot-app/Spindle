"""Copy guard: keep a value that is being copied from the request word for word.

Small models often start copying a phrase from the request and then reword it halfway
("Café Grano de Oro at 25 Calle del Prado" -> "Café Grano de Oro 25 Calle del Prado"). When the
string written so far appears in the request, and the model picks a token that breaks that copy
(and does not end the string), the token that continues the request is taken instead, as long as
the model finds it almost as likely (within `margin` in log probability). Ending the string is
never blocked, values that are not in the request (dates, codes) are left alone, and a token
that is only whitespace is never forced (the model then loses its way in the next word).
"""
import numpy as np

MIN_COPIED = 3  # bytes that must already match the request before the guard acts
TOP = 32        # candidate tokens looked at


def _continues(raw, piece, text):
    """True when raw + piece appears in `text`."""
    return (raw + piece) in text


def guard(sampler, logits, chosen, query, margin):
    """`chosen`, or a token that keeps copying the request. `query` is the request as UTF-8 bytes."""
    state = sampler.json
    if sampler.mode != 'json' or state.mode != 'string' or state.string_key or len(state.raw) < MIN_COPIED \
            or b'\\' in state.raw or state.raw not in query:
        return chosen
    piece = sampler.pieces[chosen]
    if piece.startswith(b'"') or _continues(state.raw, piece, query):
        return chosen
    values = np.asarray(logits)
    best = values[chosen]
    top = np.argpartition(-values, TOP)[:TOP]
    for token in top[np.argsort(-values[top], kind='stable')]:
        token = int(token)
        if values[token] < best - margin:
            break
        piece = sampler.pieces[token]
        if piece.strip() and b'"' not in piece and _continues(state.raw, piece, query) \
                and sampler.transition(token) is not None:
            return token
    return chosen
