"""Enum debiasing: correct the model's pull towards some labels of a closed list.

With many labels (a 14-way topic list, say) a small model tends to favour a few of them for almost
any input. This option learns that pull from the requests themselves, without labels:

* At the first token of an enum value (all labels still possible, each label starting with its own
  token), the model's probabilities over those first tokens are recorded.
* The prior of a field is the mean of those probabilities over earlier requests with the same
  field and labels. Once `MIN_REQUESTS` requests have been seen, the label is picked by
  p(label) / prior(label) instead of p(label).

This assumes the labels come up about equally often over time. When some labels are much more
common (an on/off switch that is nearly always "on", say), it pushes towards the rare ones, so it
is off by default (`Spindle(debias_enums=True)`, `spindle bench --debias-enums`).
"""
import numpy as np

from .grammar import variants

MIN_REQUESTS = 8


class EnumDebias:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.priors = {}   # (field, labels) -> [sum of probabilities, count]
        self._firsts = {}  # labels -> first token per label, or None when two labels share one

    def _first_tokens(self, labels):
        if labels not in self._firsts:
            firsts = [self.tokenizer.encode(label)[0] if label else None for label in labels]
            self._firsts[labels] = firsts if None not in firsts and len(set(firsts)) == len(firsts) else None
        return self._firsts[labels]

    @staticmethod
    def _enum_start(state):
        """(field, labels) when `state` sits right after the opening quote of an enum-only string value."""
        if state.mode != 'string' or state.string_key or state.raw or not state.stack:
            return None
        labels = []
        for option in variants(state.scalar_schema):
            values = option.get('enum', [option['const']] if 'const' in option else None)
            if values is None:
                return None
            labels += [v for v in values if isinstance(v, str)]
        labels = tuple(dict.fromkeys(labels))
        return (state.stack[-1].get('key'), labels) if len(labels) > 1 else None

    def select(self, sampler, logits):
        """The debiased token at the start of an enum value, or None to let the sampler choose."""
        start = self._enum_start(sampler.json) if sampler.mode == 'json' else None
        firsts = start and self._first_tokens(start[1])
        if not firsts or any(sampler.transition(token) is None for token in firsts):
            return None
        z = np.asarray(logits, np.float64)[firsts]
        p = np.exp(z - z.max())
        p /= p.sum()
        total = self.priors.setdefault(start, [np.zeros(len(firsts)), 0])
        choice = None
        if total[1] >= MIN_REQUESTS:
            prior = total[0] / total[1]
            choice = firsts[int(np.argmax(np.log(p) - np.log(np.maximum(prior, 1e-6))))]
        total[0] += p  # this request joins the prior only after its own choice
        total[1] += 1
        return choice
