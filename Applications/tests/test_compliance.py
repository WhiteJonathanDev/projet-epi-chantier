"""Tests unitaires de la regle de conformite EPI (Applications/compliance.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compliance import CLASS_NAMES_EPI, COMPLIANT_CLASSES_EPI, MISSING_HINT_EPI, evaluate_compliance

HELMET, HEAD, SAFETY_VEST = 0, 1, 2


def test_no_detection_is_conforme():
    conforme, missing = evaluate_compliance([])
    assert conforme is True
    assert missing == []


def test_helmet_detected_is_conforme():
    conforme, missing = evaluate_compliance([HELMET])
    assert conforme is True
    assert missing == []


def test_safety_vest_detected_is_conforme():
    conforme, missing = evaluate_compliance([SAFETY_VEST])
    assert conforme is True
    assert missing == []


def test_head_without_helmet_is_non_conforme():
    conforme, missing = evaluate_compliance([HEAD])
    assert conforme is False
    assert missing == ["helmet"]


def test_head_and_helmet_together_is_conforme():
    # Le casque detecte suffit a rendre l'ensemble conforme, meme si une tete
    # nue est aussi detectee dans la meme image (ex: deux travailleurs).
    conforme, missing = evaluate_compliance([HEAD, HELMET])
    assert conforme is True
    assert missing == []


def test_float_class_ids_are_handled():
    # Ultralytics renvoie des floats (tensor.tolist()), pas des int.
    conforme, missing = evaluate_compliance([1.0])
    assert conforme is False
    assert missing == ["helmet"]


def test_unrelated_class_without_hint_gives_empty_missing():
    conforme, missing = evaluate_compliance([SAFETY_VEST := 2], compliant_classes=[HELMET], missing_hint={})
    assert conforme is False
    assert missing == []


def test_custom_compliant_classes_override():
    conforme, _ = evaluate_compliance([HEAD], compliant_classes=[HEAD])
    assert conforme is True


def test_class_maps_are_consistent():
    # Chaque classe "manquante" citee dans MISSING_HINT_EPI doit correspondre
    # a un nom de classe connu, et ne doit pas etre elle-meme une classe "compliant".
    for cls_id, hint_name in MISSING_HINT_EPI.items():
        assert hint_name in CLASS_NAMES_EPI.values()
        assert cls_id not in COMPLIANT_CLASSES_EPI
