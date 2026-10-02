"""NAV curves as inline SVG, rendered on the server.

Deliberately not a charting library. The only thing this page has to draw is
two polylines, and generating them here means the dashboard ships no
third-party JavaScript, loads nothing from a CDN, and never has to inject
markup into the DOM at runtime. Everything below is arithmetic on floats, so
there is no untrusted string anywhere in the output.
"""

from __future__ import annotations

from html import escape

WIDTH, HEIGHT = 960, 320
PAD_X, PAD_TOP, PAD_BOTTOM = 56, 16, 28


def nav_chart(dates: list[str], nav: list[float], benchmark: list[float], benchmark_name: str = "benchmark") -> str:
    """A two-series line chart sized to the viewBox, as an <svg> string."""
    if not dates or not nav:
        return '<p class="empty">No sessions to plot.</p>'

    series = [v for v in (*nav, *benchmark) if v is not None]
    low, high = min(series), max(series)
    if high == low:  # a flat curve still needs a band to sit in
        high = low + 1.0

    plot_w = WIDTH - 2 * PAD_X
    plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM
    span = max(len(nav) - 1, 1)

    def x(i: int) -> float:
        return PAD_X + plot_w * i / span

    def y(value: float) -> float:
        return PAD_TOP + plot_h * (1 - (value - low) / (high - low))

    def points(values: list[float]) -> str:
        return " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values) if v is not None)

    gridlines = []
    labels = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = low + (high - low) * frac
        gy = y(value)
        gridlines.append(f'<line x1="{PAD_X}" y1="{gy:.1f}" x2="{PAD_X + plot_w}" y2="{gy:.1f}" class="grid"/>')
        labels.append(f'<text x="{PAD_X - 8}" y="{gy + 4:.1f}" class="ylabel">{_money(value)}</text>')

    first, last = escape(dates[0]), escape(dates[-1])
    return f"""<svg viewBox="0 0 {WIDTH} {HEIGHT}" class="chart" role="img"
     aria-label="Fund net asset value against {escape(benchmark_name)} from {first} to {last}">
  {"".join(gridlines)}
  <polyline class="series benchmark" points="{points(benchmark)}"/>
  <polyline class="series fund" points="{points(nav)}"/>
  {"".join(labels)}
  <text x="{PAD_X}" y="{HEIGHT - 8}" class="xlabel">{first}</text>
  <text x="{PAD_X + plot_w}" y="{HEIGHT - 8}" class="xlabel end">{last}</text>
</svg>"""


def _money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}k"
    return f"${value:.0f}"
