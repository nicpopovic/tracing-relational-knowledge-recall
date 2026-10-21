"""Dataset helpers."""

# Wikidata property ID -> human-readable name for the 16 FewRel validation relations.
PROPERTY_NAMES = {
    "P25": "mother",
    "P26": "spouse",
    "P40": "child",
    "P59": "constellation",
    "P155": "follows",
    "P177": "crosses",
    "P206": "located in or next to body of water",
    "P361": "part of",
    "P364": "original language of film or TV show",
    "P410": "military, police or special rank",
    "P412": "voice type",
    "P413": "position played on team / speciality",
    "P463": "member of",
    "P641": "sport",
    "P921": "main subject",
    "P2094": "competition class",
}


def mask_text(item):
    """Replace every annotated entity mention in ``item["text"]`` with ``[_]``."""
    spans = sorted(
        [(start, end, eid, sf) for (start, end), eid, sf
         in zip(item['entities_pos'], item['entities_ids'], item['entities_sf'])],
        key=lambda x: x[0]
    )

    masked_text = ""
    last = 0

    for start, end, eid, sf in spans:
        # text before the entity
        if last < start:
            masked_text += item['text'][last:start]

        masked_text += "[_]"
        last = end

    # append remaining text
    if last < len(item['text']):
        masked_text += item['text'][last:]

    return masked_text
