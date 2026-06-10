from __future__ import annotations


DIVERSE_DOG_CLASSES = [
    "Afghan Hound",
    "Basset Hound",
    "Bull Terrier",
    "Chihuahua",
    "Chow Chow",
    "Dalmatian",
    "Great Dane",
    "Greyhound",
    "Pembroke Welsh Corgi",
    "Poodle",
]

DIVERSE_5_DOG_CLASSES = [
    "Afghan Hound",
    "Basset Hound",
    "Chihuahua",
    "Dalmatian",
    "Newfoundland",
]


def selected_classes_for_preset(preset: str) -> list[str] | None:
    if preset == "diverse5":
        return DIVERSE_5_DOG_CLASSES
    if preset == "diverse10":
        return DIVERSE_DOG_CLASSES
    return None
