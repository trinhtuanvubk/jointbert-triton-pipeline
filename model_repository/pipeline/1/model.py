import json
import numpy as np
import torch
from torch import autocast
from pathlib import Path
import random

import triton_python_backend_utils as pb_utils

from torch.utils.dlpack import to_dlpack, from_dlpack

from transformers import RobertaTokenizer
from loguru import logger
from utils import crf_decode, util

ROBERTA_CONFIG = "config/roberta_config"
INTENT_LABEL = "config/roberta_labels/intent_label.txt"
SLOT_LABEL = "config/roberta_labels/slot_label.txt"


cur_folder = Path(__file__).parent
roberta_tokenizer_path = str(cur_folder/ROBERTA_CONFIG)
intent_label_path = str(cur_folder/INTENT_LABEL)
slot_label_path = str(cur_folder/SLOT_LABEL)



class TritonPythonModel:
    def initialize(self, args):
        # parse model_config
        self.model_config = model_config = json.loads(args["model_config"])
        # get last configuration
        # last_output_config = pb_utils.get_output_config_by_name(model_config, "paraphrase_answer")
        #convert triton type to numpy type/
        # self.last_output_dtype = pb_utils.triton_string_to_numpy(last_output_config["data_type"])

        # load roberta tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_tokenizer_path, local_files_only=True)

        # load dict intent and dict tag
        intent_label_lst = util.get_intent_labels(intent_label_path)
        self.dict_intents = {i: intent for i, intent in enumerate(intent_label_lst)}

        slot_label_lst = util.get_slot_labels(slot_label_path)
        self.dict_tags = {i: tag for i, tag in enumerate(slot_label_lst)}
        
        # string dtype
        self._dtypes = [np.bytes_, np.object_]

    def execute(self, requests):
        responses = []
        for request in requests:
            
            input_text = pb_utils.get_input_tensor_by_name(request, "input_text")
            input_text = input_text.as_numpy().astype(np.bytes_)[0]
            input_text = [i.decode("utf-8").lower() for i in input_text]

            logger.debug("input text:{}".format(input_text))
            
            # ==== JointBert ====
            # string = "customer service"
            joinbert_inputs = self.tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
            _input_ids = joinbert_inputs.input_ids
            _attention_mask = joinbert_inputs.attention_mask

            input_ids = pb_utils.Tensor(
                "input_ids",
                _input_ids.numpy().astype(np.int64)
            )
            
            attention_mask = pb_utils.Tensor(
                "attention_mask",
                _attention_mask.numpy().astype(np.int64)
            )

            jointbert_request = pb_utils.InferenceRequest(
                model_name="jointbert",
                requested_output_names=["intent_logits", "slot_logits", "transitions", "start_transition", "end_transition"],
                inputs=[input_ids, attention_mask],
            )
            response = jointbert_request.exec()


            intent_logits = pb_utils.get_output_tensor_by_name(response, "intent_logits")
            intent_logits = from_dlpack(intent_logits.to_dlpack()).clone()

            slot_logits = pb_utils.get_output_tensor_by_name(response, "slot_logits")
            slot_logits = from_dlpack(slot_logits.to_dlpack()).clone()

            transitions = pb_utils.get_output_tensor_by_name(response, "transitions")
            transitions = from_dlpack(transitions.to_dlpack()).clone()

            start_transition = pb_utils.get_output_tensor_by_name(response, "start_transition")
            start_transitions = from_dlpack(start_transition.to_dlpack()).clone()

            end_transition = pb_utils.get_output_tensor_by_name(response, "end_transition")
            end_transitions = from_dlpack(end_transition.to_dlpack()).clone()

            # decode crf 
            best_tags_list = crf_decode.decode(slot_logits, transitions, start_transitions, end_transitions)

            intent_preds = intent_logits.argmax(dim=-1).numpy().tolist()
            logger.debug(f"{intent_preds}-{best_tags_list}")

            intentions =  [self.dict_intents[i] for i in intent_preds]
            slots = util.post_processing(_input_ids, best_tags_list, self.tokenizer)
            slots = [[self.dict_tags[i] for i in tag] for tag in slots]

            logger.debug(f"{intentions}-{slots}")
            final_answers = [f"{intentions}-{slots}"]
            
            # ==== Sending Response ====
            last_output_tensor = pb_utils.Tensor(
                "intent_slot", np.array([i.encode('utf-8') for i in final_answers], dtype=self._dtypes[0]))
    
            inference_response = pb_utils.InferenceResponse([last_output_tensor])
        
            responses.append(inference_response)

            return responses



