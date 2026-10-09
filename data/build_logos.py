"""Build data/logos.npz: every solid icon from three open sets, as 48x48 ink masks.

  Simple Icons     brand marks        npm simple-icons          CC0-1.0
  Phosphor (fill)  solid UI icons     npm @phosphor-icons/core  MIT
  Tabler (filled)  solid UI icons     npm @tabler/icons         MIT

Each SVG is rendered by rsvg-convert at 40px and centred on a 48px canvas, so
every mark has a 4px margin. Pixels are ink amount: 0 is paper, 255 is ink.
`source` records which set each image came from (0 brand marks, 1 UI icons).

    python data/build_logos.py      (needs rsvg-convert: brew install librsvg)
"""

import io
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
SIZE = 48
MARK = 40

# (npm tarball, folder inside it with the solid icons, 0 brand / 1 UI icon)
SOURCES = [
    ("https://registry.npmjs.org/simple-icons/-/simple-icons-16.34.0.tgz", "package/icons/", 0),
    ("https://registry.npmjs.org/@phosphor-icons/core/-/core-2.1.1.tgz", "package/assets/fill/", 1),
    ("https://registry.npmjs.org/@tabler/icons/-/icons-3.49.0.tgz", "package/icons/filled/", 1),
]


def svgs_in(url: str, folder: str) -> list[bytes]:
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
    found = []
    for member in archive.getmembers():
        if member.name.startswith(folder) and member.name.endswith(".svg"):
            found.append(archive.extractfile(member).read())
    return found


def render(svg: bytes) -> np.ndarray | None:
    result = subprocess.run(
        ["rsvg-convert", "-w", str(MARK), "-h", str(MARK), "-a", "-f", "png"],
        input=svg, capture_output=True,
    )
    if result.returncode != 0:
        return None
    mark = Image.open(io.BytesIO(result.stdout)).convert("RGBA")
    # Ink is wherever the icon is opaque; colour is ignored on purpose.
    alpha = np.array(mark)[:, :, 3]
    inked_rows = np.nonzero(alpha.max(axis=1) > 127)[0]
    if len(inked_rows) == 0 or inked_rows[-1] - inked_rows[0] < MARK // 3:
        return None                     # blank, or a wide wordmark squashed into a strip
    canvas = np.zeros((SIZE, SIZE), np.uint8)
    top = (SIZE - alpha.shape[0]) // 2
    left = (SIZE - alpha.shape[1]) // 2
    canvas[top:top + alpha.shape[0], left:left + alpha.shape[1]] = alpha
    return canvas


def main():
    images, sources = [], []
    for url, folder, source in SOURCES:
        svgs = svgs_in(url, folder)
        before = len(images)
        for svg in svgs:
            image = render(svg)
            if image is not None:
                images.append(image)
                sources.append(source)
        print(f"{url.split('/-/')[1]}: {len(images) - before} of {len(svgs)}")
    stack = np.stack(images)
    np.savez_compressed(HERE / "logos.npz", images=stack, source=np.array(sources, np.uint8))
    print(f"saved {stack.shape} to data/logos.npz")


if __name__ == "__main__":
    main()
