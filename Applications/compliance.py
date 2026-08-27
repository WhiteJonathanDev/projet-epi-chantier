"""Logique de conformite EPI, isolee de Streamlit pour etre testable unitairement.

Regle : un travailleur est non conforme si une classe "compliant" (ex. helmet,
safety-vest) est absente alors qu'une classe associee est detectee sans elle
(ex. head detecte -> signifie une tete sans casque).
"""

CLASS_NAMES_EPI = {0: "helmet", 1: "head", 2: "safety-vest"}
COMPLIANT_CLASSES_EPI = [0, 2]  # helmet, safety-vest presents = conforme
MISSING_HINT_EPI = {1: "helmet"}  # "head" detecte sans casque -> casque manquant


def evaluate_compliance(classes, compliant_classes=None, missing_hint=None):
    """Evalue la conformite EPI a partir des classes detectees (ids YOLO, floats ou int).

    Retourne (conforme: bool, missing: list[str]).
    - Aucune detection -> conforme=True, missing=[] (rien a signaler).
    - Au moins une classe "compliant" detectee -> conforme=True.
    - Sinon -> conforme=False, avec la liste des EPI manquants deduits de missing_hint.
    """
    compliant_classes = COMPLIANT_CLASSES_EPI if compliant_classes is None else compliant_classes
    missing_hint = MISSING_HINT_EPI if missing_hint is None else missing_hint

    if not classes:
        return True, []

    int_classes = [int(c) for c in classes]
    conforme = any(c in compliant_classes for c in int_classes)

    if conforme:
        return True, []

    missing = [missing_hint[c] for c in int_classes if c in missing_hint]
    return False, missing
