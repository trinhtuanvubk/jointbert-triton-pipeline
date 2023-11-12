
import os


def get_intent_labels(path):
    return [label.strip() for label in open(os.path.join(path), 'r', encoding='utf-8')]

def get_slot_labels(path):
    return [label.strip() for label in open(os.path.join(path), 'r', encoding='utf-8')]

def post_processing(input_ids, slots, tokenizer):
    list_raw_slots = []
    for index in range(input_ids.shape[0]):
        list_tokens = tokenizer.convert_ids_to_tokens(input_ids[index])
        list_index_g = [index for index, i in enumerate(list_tokens) if 'Ġ' in i]
        list_index_g = [1] + list_index_g
        raw_slots = []
        for j, token in enumerate(list_tokens):
            if j in list_index_g:
                raw_slots.append(slots[index][j])
        list_raw_slots.append(raw_slots)
    return list_raw_slots