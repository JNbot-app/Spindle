"""Loss over time, for the training log.

The loss of one logged window is noisy, so `LossTrend` keeps the whole history and reports:
  * `avg`: the mean loss over the last `window` updates,
  * `trend`: how much that mean changed against the `window` updates before it (negative = still improving),
  * every `chart_every` updates, a one-line text chart of the loss so far (one bar per bucket of updates).
`plot` draws the same history to a PNG (needs matplotlib; skipped when it is missing).
"""
import numpy as np

BARS = '▁▂▃▄▅▆▇█'


class LossTrend:
    def __init__(self, start=0, window=100, chart_every=250, width=60):
        self.start, self.window, self.chart_every, self.width = start, window, chart_every, width
        self.losses = []  # one per update, from update `start` + 1 on (a resumed run starts later)

    def add(self, loss):
        self.losses.append(loss)

    def summary(self):
        """{'avg', 'trend'} for the log line (trend only once two full windows exist)."""
        recent = self.losses[-self.window:]
        result = {'avg': round(float(np.mean(recent)), 4)}
        if len(self.losses) >= 2 * self.window:
            before = float(np.mean(self.losses[-2 * self.window:-self.window]))
            result['trend'] = f'{(result["avg"] - before) / before * 100:+.1f}%'
        return result

    def chart(self):
        """A text chart of the loss so far, or None when it is not time for one."""
        if len(self.losses) % self.chart_every:
            return None
        buckets = np.array_split(np.asarray(self.losses), min(self.width, len(self.losses)))
        means = np.array([b.mean() for b in buckets])
        low, high = means.min(), means.max()
        scaled = (means - low) / max(high - low, 1e-9) * (len(BARS) - 1)
        bars = ''.join(BARS[int(round(s))] for s in scaled)
        return f'Loss over {len(self.losses):,} updates: {means[0]:.3f} {bars} {means[-1]:.3f}  (min {low:.3f})'

    def plot(self, path, validation=()):
        """Write train loss (raw + smoothed) and validation points [(step, loss)] to a PNG."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        steps = self.start + np.arange(1, len(self.losses) + 1)
        smooth = np.convolve(self.losses, np.ones(self.window) / self.window, mode='valid')
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, self.losses, alpha=0.25, label='train (per update)')
        if len(smooth):
            ax.plot(steps[self.window - 1:], smooth, label=f'train (mean of {self.window})')
        if validation:
            ax.plot(*zip(*validation), 'o-', label='validation')
        ax.set_xlabel('update')
        ax.set_ylabel('loss')
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path
