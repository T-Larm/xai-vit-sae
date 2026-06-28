import re

CATEGORIES = ['background', 'texture', 'color', 'object_part', 'scene', 'object']

# Keywords checked in priority order; first match wins.
# 'object' is the catch-all (no keywords) for ImageNet class names.
# NOTE: scene is placed before texture so that labels like 'grass texture'
# correctly resolve to 'scene' (grass keyword) rather than 'texture'.
# NOTE: word-boundary matching (\b) prevents substring collisions such as
# 'face' inside 'surface', or 'ear' inside 'wear'.
_RULES = [
    ('background',  ['background', 'backdrop']),
    ('object_part', ['eye', 'beak', 'nose', 'ear', 'paw', 'muzzle', 'wing', 'fin',
                     'tail', 'claw', 'wheel', 'window', 'roof', 'wall', 'face',
                     'slot', 'file', 'antenna', 'horn', 'tooth']),
    ('scene',       ['sky', 'water', 'ocean', 'mountain', 'forest', 'desert', 'snow',
                     'sand', 'rock', 'cloud', 'road', 'floor', 'grass', 'surface',
                     'field', 'beach', 'river', 'lake', 'soil', 'mud']),
    ('texture',     ['fur', 'feather', 'scale', 'grain', 'fabric', 'skin', 'wool',
                     'metal surface', 'wooden surface', 'stone surface',
                     'concrete surface', 'brick', 'tile', 'stripe', 'pattern',
                     'texture', 'edge', 'dot', 'grid']),
    ('color',       ['dark background', 'bright background', 'white', 'black',
                     'brown', 'dark', 'bright']),
]

# Pre-compile patterns for efficiency; strip surrounding spaces from color keywords.
_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    (cat, [re.compile(r'\b' + re.escape(kw.strip()) + r'\b') for kw in kws])
    for cat, kws in _RULES
]


def assign_category(label: str) -> str:
    """Map a CLIP label string to one of 6 semantic categories.

    Priority: background > object_part > scene > texture > color > object.
    'object' is the default catch-all for ImageNet class names.
    Matching uses word boundaries to avoid substring collisions
    (e.g. 'face' inside 'surface', 'ear' inside 'feather').
    """
    label_lower = label.lower()
    for category, patterns in _PATTERNS:
        if any(p.search(label_lower) for p in patterns):
            return category
    return 'object'


def get_category_distribution(labels: list) -> dict:
    """Count features per category for a list of label strings."""
    dist = {cat: 0 for cat in CATEGORIES}
    for label in labels:
        dist[assign_category(label)] += 1
    return dist
