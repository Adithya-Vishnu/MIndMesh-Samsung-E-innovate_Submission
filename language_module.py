from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
ACTIONS = [
    "move",
    "pick and place",
    "push"
]
COLOURS = [
    "red",
    "green",
    "blue",
    "yellow",
    "cyan"
]
RELATIONS = {
    "right": (0.08, 0.00),
    "left": (-0.08, 0.00),
    "front": (0.00, -0.10),
    "behind": (0.00, 0.10),
    "next to": (0.08, 0.00),
}
def parse_command(command):

    result = {
        "raw": command,
        "action": None,
        "object": None,
        "target": None,
        "relation": None,
        "offset": None,
        "error": None
    }

    cmd = command.lower()

    # -------------------------
    # ACTION
    # -------------------------
    ai_result = classifier(cmd, ACTIONS)

    result["action"] = ai_result["labels"][0]

    # -------------------------
    # RELATION FIRST
    # IMPORTANT FIX
    # -------------------------
    relation_found = False

    for rel, offset in RELATIONS.items():

        if rel in cmd:

            result["relation"] = rel
            result["offset"] = offset

            relation_found = True
            break

    if not relation_found:
        result["error"] = "No spatial relation found"
        return result

    # -------------------------
    # FIND COLOURS
    # -------------------------
    found = []

    words = cmd.split()

    for word in words:

        if word in COLOURS:
            found.append(word)

    if len(found) < 2:
        result["error"] = "Need source and target objects"
        return result

    result["object"] = found[0]
    result["target"] = found[1]

    return result