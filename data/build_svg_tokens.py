"""Build data/svg_tokens.npz: the same icon sets as build_logos.py, as token sequences.

Every shape is turned into absolute path commands (svgelements converts circles,
rects, arcs and quadratic curves into these), coordinates are snapped to a 64x64
grid, and the drawing becomes a list of tokens:

    [label] M x y  C x1 y1 x2 y2 x y  L x y ...  Z  M ...  Z  [end]

Icons longer than MAX_LEN tokens are skipped (about 10%, mostly detailed brand marks).

    python data/build_svg_tokens.py
"""

import io
import sys
from pathlib import Path

import numpy as np
from svgelements import SVG, Close, CubicBezier, Line, Move, Path as SvgPath, Shape

sys.path.insert(0, str(Path(__file__).parent))
from build_logos import SOURCES, svgs_in  # noqa: E402

HERE = Path(__file__).parent
GRID = 64
MAX_LEN = 512

# Token ids. Coordinates 0..63 are ids COORD..COORD+63.
PAD, END, MOVE, LINE, CURVE, CLOSE = 0, 1, 2, 3, 4, 5
LABEL = 6          # 6 brand mark, 7 UI icon
COORD = 8
VOCAB = COORD + GRID


def snap(value: float, size: float) -> int:
    return int(round(min(max(value / size, 0.0), 1.0) * (GRID - 1)))


def tokenise(svg_bytes: bytes, label: int) -> list[int] | None:
    svg = SVG.parse(io.BytesIO(svg_bytes))
    size = svg.viewbox.width if svg.viewbox else 24.0
    tokens = [LABEL + label]

    def point(p):
        tokens.append(COORD + snap(p.x, size))
        tokens.append(COORD + snap(p.y, size))

    for element in svg.elements():
        if not isinstance(element, Shape):
            continue
        if element.fill is None or element.fill.value is None:
            continue            # fill="none": Tabler's invisible bounding box, outlines
        path = abs(SvgPath(element))
        # quadratic curves and arcs become cubic curves; everything is absolute
        path.approximate_arcs_with_cubics()
        for segment in path:
            if isinstance(segment, Move):
                tokens.append(MOVE)
                point(segment.end)
            elif isinstance(segment, Close):
                tokens.append(CLOSE)
            elif isinstance(segment, Line):
                tokens.append(LINE)
                point(segment.end)
            elif isinstance(segment, CubicBezier):
                tokens.append(CURVE)
                point(segment.control1)
                point(segment.control2)
                point(segment.end)
            else:       # quadratic: raise it to a cubic with the same shape
                q = segment
                c1 = q.start + (q.control - q.start) * (2 / 3)
                c2 = q.end + (q.control - q.end) * (2 / 3)
                tokens.append(CURVE)
                point(c1)
                point(c2)
                point(q.end)
    tokens.append(END)
    if len(tokens) > MAX_LEN or len(tokens) < 6:
        return None
    return tokens


def to_path(tokens: list[int]) -> str:
    """Tokens back to an SVG path `d` on a 0..63 grid. Shared with the trainer and checks."""
    parts = []
    for token in tokens:
        if token == MOVE:
            parts.append("M")
        elif token == LINE:
            parts.append("L")
        elif token == CURVE:
            parts.append("C")
        elif token == CLOSE:
            parts.append("Z")
        elif token >= COORD:
            parts.append(str(token - COORD))
    return " ".join(parts)


def main():
    sequences, sources = [], []
    for url, folder, label in SOURCES:
        svgs = svgs_in(url, folder)
        kept = 0
        for svg in svgs:
            try:
                tokens = tokenise(svg, label)
            except Exception:
                tokens = None
            if tokens is not None:
                sequences.append(tokens)
                sources.append(label)
                kept += 1
        print(f"{url.split('/-/')[1]}: {kept} of {len(svgs)}")
    padded = np.zeros((len(sequences), MAX_LEN), np.int16)
    for i, tokens in enumerate(sequences):
        padded[i, : len(tokens)] = tokens
    lengths = np.array([len(t) for t in sequences], np.int16)
    np.savez_compressed(HERE / "svg_tokens.npz", tokens=padded, lengths=lengths, source=np.array(sources, np.uint8))
    print(f"saved {len(sequences)} sequences, median length {int(np.median(lengths))}")


if __name__ == "__main__":
    main()
