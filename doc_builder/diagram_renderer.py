"""
RD.011 Agent — Graphviz DOT diagram renderer.

Converts a Graphviz DOT string to a PNG file using the graphviz Python
library (backed by the dot.EXE binary).  Falls back to a Pillow swimlane
renderer if the DOT source is invalid or the binary is unavailable.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from config import DIAGRAM_FALLBACK_HEIGHT, DIAGRAM_FALLBACK_WIDTH, DIAGRAMS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PATH patching — ensure the conda-env dot binary is discoverable
# ---------------------------------------------------------------------------

def _ensure_dot_on_path() -> None:
    """
    Add the conda-environment Library/bin directory to PATH at import time
    so that ``graphviz.Source.render()`` can find ``dot.exe`` on Windows.

    With a typical conda install the layout is:
        <env>/python.exe
        <env>/Library/bin/dot.exe
    """
    python_dir = Path(sys.executable).parent
    candidate = python_dir / "Library" / "bin"
    if candidate.is_dir():
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        candidate_str = str(candidate)
        if candidate_str not in path_entries:
            os.environ["PATH"] = candidate_str + os.pathsep + os.environ.get("PATH", "")
            logger.debug("Added %s to PATH for Graphviz dot binary", candidate_str)


_ensure_dot_on_path()


def _ensure_diagrams_dir() -> Path:
    """Create the diagrams output directory if it does not exist."""
    path = Path(DIAGRAMS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Swimlane rendering constants
# ---------------------------------------------------------------------------

LANE_H = 100          # Height per lane
HEADER_W = 140        # Width for actor labels
COL_W = 120           # Width per step column
CANVAS_W = 900        # Base canvas width
TITLE_H = 40          # Top margin / error-image header height

# Graphviz stderr strings that indicate structural corruption
GRAPHVIZ_FATAL_WARNINGS = [
    "already in a rankset",
    "removing empty cluster",
    "install_in_rank",
    "cyclic",
    "lost in space",
]

# Colors
COLOR_LANE_LINE = "#888888"
COLOR_LANE_HDR = "#E8EEF5"
COLOR_LANE_EVEN = "#F8FAFC"
COLOR_LANE_ALT = "#EBEBEB"
COLOR_NODE_START = "#1F3A5A"
COLOR_NODE_PROCESS = "#3C6E99"


def _font(size: int):
    """Return a PIL font at *size*, falling back to the built-in default."""
    from PIL import ImageFont
    for face in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(face, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def _extract_swimlane_structure(dot_string: str) -> tuple[List[str], List[Dict]]:
    """
    Extract actor/lane names and step information from DOT source.

    Returns: (lanes, steps)
    - lanes: List of actor names in order
    - steps: List of dicts with step info
    """
    lanes_set = set()
    lanes_order = []
    steps = []

    # Extract subgraph cluster names (actors/lanes)
    for match in re.finditer(r'subgraph\s+cluster_(\w+)\s*\{[^}]*label\s*=\s*"([^"]*)"', dot_string):
        cluster_id, label = match.groups()
        actor_name = label.replace("\\n", " / ").strip()
        if actor_name not in lanes_set:
            lanes_set.add(actor_name)
            lanes_order.append(actor_name)

    # Extract nodes (steps) and their actors
    # Pattern: nodeid [shape=..., label="...", ...]
    # We'll assign steps to lanes based on their position in clusters
    for match in re.finditer(r'(\w+)\s*\[([^\]]*)\]', dot_string):
        node_id, attrs_str = match.groups()
        if node_id.startswith("title") or node_id in ("s_start", "a_end", "title_bar"):
            continue  # Skip non-content nodes

        # Extract label
        label_match = re.search(r'label\s*=\s*"([^"]*)"', attrs_str)
        label = label_match.group(1) if label_match else node_id

        # Infer actor from node ID prefix (e.g., 's_' for Supplier, 'a_' for Admin)
        if not steps or len(steps) < 20:  # Limit steps shown
            steps.append({
                "id": node_id,
                "action": label.replace("\\n", " "),
                "business_actor": lanes_order[0] if lanes_order else "Actor"
            })

    # Fallback: if no lanes extracted, use generic ones
    if not lanes_order:
        lanes_order = ["Actor A", "Actor B"]

    return lanes_order, steps


def _render_swimlane_pillow(lanes: List[str], steps: List[Dict], process_id: str, output_file: str) -> None:
    """
    Render a swimlane diagram using Pillow (no Graphviz required).

    Draws horizontal lanes with steps positioned by column.
    No title bar — title comes from the Word document heading.
    """
    try:
        from PIL import Image, ImageDraw

        n_lanes = len(lanes)
        # Build all_steps with sentinel start/end entries for width calculation
        all_steps = [{"id": "__start__", "action": "Start"}] + steps + [{"id": "__end__", "action": "End"}]
        min_w = HEADER_W + (len(all_steps) + 2) * COL_W
        canvas_w = max(CANVAS_W, min_w)
        canvas_h = max(TITLE_H + n_lanes * LANE_H + 4, TITLE_H + 2 * LANE_H + 4)
        col_gap = max(COL_W, (canvas_w - HEADER_W - COL_W) // max(len(all_steps), 1))

        img = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(img)
        fn_lane = _font(11)
        fn_node = _font(9)

        # Draw lanes offset by TITLE_H to leave top margin
        lane_cy = {}
        for i, lane in enumerate(lanes):
            y0 = TITLE_H + i * LANE_H
            y1 = TITLE_H + (i + 1) * LANE_H
            bg_color = COLOR_LANE_EVEN if i % 2 == 0 else COLOR_LANE_ALT
            draw.rectangle([HEADER_W, y0, canvas_w, y1], fill=bg_color, outline=COLOR_LANE_LINE, width=1)
            draw.rectangle([0, y0, HEADER_W, y1], fill=COLOR_LANE_HDR, outline=COLOR_LANE_LINE, width=1)
            lane_cy[lane] = y0 + LANE_H // 2

            # Actor label — centered vertically in lane header
            lane_words = lane.split()
            lh = 13
            ty = lane_cy[lane] - (len(lane_words) * lh) // 2 + 2
            for word in lane_words:
                try:
                    bbox = draw.textbbox((0, 0), word, font=fn_lane)
                    tw = bbox[2] - bbox[0]
                except Exception:
                    tw = len(word) * 6
                draw.text((HEADER_W // 2 - tw // 2, ty), word, font=fn_lane, fill="#333333")
                ty += lh

        # Draw steps as boxes distributed across lanes using col_gap spacing
        if steps:
            for idx, step in enumerate(steps):
                cx = HEADER_W + col_gap // 2 + idx * col_gap
                cy_lane = lane_cy.get(
                    step.get("business_actor", lanes[0]),
                    TITLE_H + LANE_H // 2,
                )
                box_w, box_h = min(col_gap - 10, 80), 40
                draw.rectangle(
                    [cx - box_w // 2, cy_lane - box_h // 2,
                     cx + box_w // 2, cy_lane + box_h // 2],
                    fill=COLOR_NODE_PROCESS, outline="#1F3A5A", width=1,
                )
                label_short = step.get("action", "Step")[:15]
                try:
                    bbox = draw.textbbox((0, 0), label_short, font=fn_node)
                    tw = bbox[2] - bbox[0]
                except Exception:
                    tw = len(label_short) * 5
                draw.text((cx - tw // 2, cy_lane - 6), label_short, font=fn_node, fill="white")

        img.save(output_file, "PNG", dpi=(150, 150))
        logger.info("Pillow swimlane rendered: %s", output_file)

    except Exception as e:
        logger.error("Swimlane render failed for %s: %s", process_id, e)
        fallback_h = max(400, TITLE_H + 3 * LANE_H)
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (CANVAS_W, fallback_h), "white")
            draw = ImageDraw.Draw(img)
            draw.rectangle([2, 2, CANVAS_W - 2, fallback_h - 2],
                           outline="#AAAAAA", width=2)
            draw.rectangle([0, 0, CANVAS_W, TITLE_H],
                           fill="#EEEEEE", outline="#AAAAAA", width=1)
            draw.text((20, 14), f"Process Flow — {process_id}",
                      font=_font(13), fill="#1F3864")
            draw.text((20, TITLE_H + 20),
                      f"Diagram render error: {str(e)[:120]}",
                      font=_font(11), fill="#CC0000")
            img.save(output_file, "PNG", dpi=(150, 150))
        except Exception:
            # Last resort: bare-minimum valid PNG
            import struct, zlib
            def _png_chunk(tag: bytes, data: bytes) -> bytes:
                crc = zlib.crc32(tag + data) & 0xFFFFFFFF
                return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
            png = (
                b"\x89PNG\r\n\x1a\n"
                + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                + _png_chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
                + _png_chunk(b"IEND", b"")
            )
            with open(output_file, "wb") as f:
                f.write(png)

def render_dot_to_png(dot_string: str, process_id: str) -> str:
    """
    Render a Graphviz DOT string to a PNG file.

    Strategy 1 — subprocess dot binary (captures stderr for structural warnings):
        Uses the installed dot binary to produce a high-quality PNG.
        Falls back on Graphviz fatal warnings (rank conflicts, cycles, etc.).

    Strategy 2 — Pillow swimlane renderer:
        Parses the DOT source and draws swimlanes using Pillow.
        No external binary required.

    Parameters
    ----------
    dot_string
        The Graphviz DOT diagram source string (must start with ``digraph``).
    process_id
        Used to derive the output filename.
        May include a client name prefix (e.g. "Contoso.AP.01"); spaces and
        special characters are replaced with underscores for the filename.

    Returns
    -------
    str
        Absolute path to the generated PNG file.
    """
    diagrams_dir = _ensure_diagrams_dir()
    # Sanitize for filesystem: replace spaces and unsafe chars with underscores
    safe_name = re.sub(r"[^\w.\-]", "_", process_id)
    output_file = str(diagrams_dir / f"{safe_name}.png")
    dot_src_file = str(diagrams_dir / f"{safe_name}.dot")

    # Strategy 1: subprocess dot binary — captures stderr for structural warnings
    try:
        with open(dot_src_file, "w", encoding="utf-8") as f:
            f.write(dot_string)

        result = subprocess.run(
            ["dot", "-Tpng", "-o", output_file, dot_src_file],
            capture_output=True, text=True,
        )
        stderr_output = result.stderr or ""
        has_fatal_warning = any(w in stderr_output for w in GRAPHVIZ_FATAL_WARNINGS)

        if result.returncode != 0 or has_fatal_warning:
            if has_fatal_warning:
                logger.warning(
                    "Graphviz structural warning for %s — "
                    "falling back to Pillow renderer. stderr: %s",
                    process_id, stderr_output[:200],
                )
            raise RuntimeError(f"Graphviz failed: {stderr_output[:300]}")

        logger.info("Graphviz rendered: %s", output_file)
        return output_file

    except FileNotFoundError:
        logger.warning("Graphviz dot binary not found — falling back to Pillow swimlane renderer")
    except RuntimeError:
        pass  # Already logged; fall through to Pillow
    except Exception as exc:
        logger.warning(
            "Graphviz rendering failed for %s: %s — falling back to Pillow swimlane",
            process_id, exc,
        )
    finally:
        if os.path.exists(dot_src_file):
            try:
                os.remove(dot_src_file)
            except Exception:
                pass

    # Strategy 2: Pillow swimlane renderer
    lanes, steps = _extract_swimlane_structure(dot_string)
    _render_swimlane_pillow(lanes, steps, process_id, output_file)
    return output_file
