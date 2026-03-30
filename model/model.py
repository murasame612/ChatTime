import re
from statistics import mode

import numpy as np
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer


class ChatTime:
    def __init__(self, model_path, hist_len=None, pred_len=None,
                 max_pred_len=16, num_samples=8, top_k=100, top_p=1.0, temperature=1.0):
        self.model_path = model_path
        self.hist_len = hist_len
        self.pred_len = pred_len

        self.max_pred_len = max_pred_len
        self.num_samples = num_samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        self.discretizer = Discretizer()
        self.serializer = Serializer()

        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            # device_map="auto",
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda:0")

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id
        self.generation_pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )

    def _estimate_prediction_token_budget(self, pred_len):
        # Estimate generation length from the actual serialized target format
        # rather than assuming a fixed token cost per forecast point.
        placeholder = np.zeros(pred_len, dtype=float)
        serialized_placeholder = self.serializer.serialize(placeholder)
        response_prefix = "### Response:\n"
        tokenized = self.tokenizer(response_prefix + serialized_placeholder, add_special_tokens=False)

        return len(tokenized["input_ids"]) + 8

    def _generate_prediction_samples(self, prompt, pred_len):
        token_budget = self._estimate_prediction_token_budget(pred_len)
        return self.generation_pipe(
            prompt,
            max_new_tokens=token_budget,
            do_sample=True,
            num_return_sequences=self.num_samples,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            eos_token_id=self.eos_token_id,
        )

    def _aggregate_predictions(self, pred_list, fallback_value):
        pred_array = np.asarray(pred_list, dtype=float)
        valid_mask = ~np.isnan(pred_array)
        valid_columns = np.any(valid_mask, axis=0)

        prediction = np.full(pred_array.shape[1], fallback_value, dtype=float)
        if valid_columns.any():
            prediction[valid_columns] = np.nanmedian(pred_array[:, valid_columns], axis=0)

        return prediction

    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        series = hist_data
        prediction_list = []
        remaining = self.pred_len

        while remaining > 0:
            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)
            serialized_series = getPrompt(flag="prediction", context=context, input=serialized_series)
            current_pred_len = min(remaining, self.max_pred_len)

            samples = self._generate_prediction_samples(serialized_series, current_pred_len)

            pred_list = []
            for sample in samples:
                generated_text = sample["generated_text"]
                if "### Response:\n" not in generated_text:
                    pred = np.full(current_pred_len, np.nan)
                    pred_list.append(pred)
                    continue

                serialized_prediction = generated_text.split("### Response:\n", 1)[1]
                dispersed_prediction = self.serializer.inverse_serialize(serialized_prediction)
                if dispersed_prediction.size == 0:
                    pred = np.full(current_pred_len, np.nan)
                    pred_list.append(pred)
                    continue

                pred = self.discretizer.inverse_discretize(dispersed_prediction)

                if len(pred) < current_pred_len:
                    pred = np.concatenate([pred, np.full(current_pred_len - len(pred), np.nan)])

                pred_list.append(pred[:current_pred_len])

            prediction = self._aggregate_predictions(pred_list, fallback_value=series[-1])
            prediction_list.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            series = np.concatenate([series, prediction], axis=-1)

        prediction = np.concatenate(prediction_list, axis=-1)

        return prediction

    def analyze(self, question, series):
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)
        serialized_series = getPrompt(flag="analysis", instruction=question, input=serialized_series)

        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )
        samples = pipe(
            serialized_series,
            max_new_tokens=self.max_pred_len,
            do_sample=True,
            num_return_sequences=self.num_samples,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            eos_token_id=self.eos_token_id,
        )

        response_list = []
        for sample in samples:
            response = sample["generated_text"].split("### Response:\n")[1].split('.')[0] + "."
            response = re.findall(r"\([abc]\)", response)[0]
            response_list.append(response)

        response = mode(response_list)

        return response
