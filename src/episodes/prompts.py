"""
Fill-in-the-blanks prompts covering both entity orders (Sec. 4.2).

Order "ht": every entity mention is blanked. Order "th": only the object t is
blanked, so the subject h stays visible. The probing position is always the
token right before the first mention of t in the assistant's answer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


def mask_text_orderfix(item: Dict[str, Any], mask_only_t: bool = False) -> str:
    spans = sorted(
        [
            (start, end, eid, sf)
            for (start, end), eid, sf in zip(
                item["entities_pos"], item["entities_ids"], item["entities_sf"]
            )
        ],
        key=lambda x: x[0],
    )

    masked_text = ""
    last = 0
    for start, end, eid, _ in spans:
        if last < start:
            masked_text += item["text"][last:start]
        if mask_only_t and eid == "h":
            masked_text += item["text"][start:end]
        else:
            masked_text += "[_]"
        last = end

    if last < len(item["text"]):
        masked_text += item["text"][last:]
    return masked_text


def build_orderfix_prompts_subset(ds, indices: Sequence[int], tokenizer) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []

    for idx in indices:
        item = ds[int(idx)]
        text = item["text"]

        if item["entity_order"] == "ht":
            item["entities_pos"] = item["h"] + item["t"]
            item["entities_ids"] = ["h"] * len(item["h"]) + ["t"] * len(item["t"])
            mask_only_t = False
            target_pos = item["entities_pos"][len(item["h"])]
        else:
            item["entities_pos"] = item["t"] + item["h"]
            item["entities_ids"] = ["t"] * len(item["t"]) + ["h"] * len(item["h"])
            mask_only_t = True
            target_pos = item["entities_pos"][0]

        item["entities_sf"] = [text[x:y] for (x, y) in item["entities_pos"]]
        masked_text = mask_text_orderfix(item, mask_only_t=mask_only_t)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "Fill in the blanks in the following text:\n\n"
                    f"{masked_text}\n\n"
                    "Provide only the completed text as your answer."
                ),
            },
            {"role": "assistant", "content": text},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        match = re.search(re.escape(text), prompt)
        if not match:
            continue
        start_in_prompt = match.start()

        rel_start = int(target_pos[0]) + start_in_prompt

        tokenized = tokenizer(
            prompt,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        token_id = None
        for i, (os, oe) in enumerate(tokenized["offset_mapping"][0]):
            if os <= rel_start < oe:
                token_id = i - 1
                break

        if token_id is None or token_id < 0:
            continue

        prompts.append(
            {
                "dataset_idx": int(idx),
                "prompt": prompt,
                "tokenized": tokenized,
                "token_id": int(token_id),
                "label": int(str(item["label"])[1:].split(" -")[0]),
                "entity_order": item["entity_order"],
            }
        )

    return prompts
