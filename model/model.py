import re
import os
from statistics import mode
from pathlib import Path

import numpy as np
import torch
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
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

        self.model, tokenizer_source = self._load_model_and_tokenizer_source(self.model_path)
        if torch.cuda.is_available():
            self.model = self.model.to(self._resolve_inference_device())

        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id
        self.device = next(self.model.parameters()).device
        self.last_generation_stats = {}
        self.last_prediction_stats = {}
        self.last_batch_prediction_stats = []

    def _resolve_inference_device(self):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return f"cuda:{local_rank}"

    def _load_model_and_tokenizer_source(self, model_path):
        model_dir = Path(model_path)
        adapter_config = model_dir / "adapter_config.json"

        if adapter_config.exists():
            try:
                from peft import AutoPeftModelForCausalLM, PeftConfig
            except Exception as exc:
                raise RuntimeError(
                    "This model path looks like a PEFT adapter checkpoint, but peft could not be imported."
                ) from exc

            peft_config = PeftConfig.from_pretrained(model_path)
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )

            tokenizer_source = model_path
            if not (model_dir / "tokenizer_config.json").exists():
                tokenizer_source = peft_config.base_model_name_or_path

            return model, tokenizer_source

        model = LlamaForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            # device_map="auto",
        )
        return model, model_path

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
        return self._generate_text(prompt, max_new_tokens=token_budget)

    def _generate_text(self, prompt, max_new_tokens):
        single_input = isinstance(prompt, str)
        prompts = [prompt] if single_input else list(prompt)
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        prompt_tokens_per_input = [
            int(length) for length in inputs["attention_mask"].sum(dim=-1).tolist()
        ]
        generation_config = GenerationConfig.from_model_config(self.model.config)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = True
        generation_config.num_return_sequences = self.num_samples
        generation_config.top_k = self.top_k
        generation_config.top_p = self.top_p
        generation_config.temperature = self.temperature
        generation_config.eos_token_id = self.eos_token_id
        generation_config.pad_token_id = self.eos_token_id
        generation_config.max_length = None

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )

        generated_token_counts = []
        generated_token_counts_per_input = []
        grouped_texts = []
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for prompt_idx in range(len(prompts)):
            sample_group = []
            token_count_group = []
            for sample_idx in range(self.num_samples):
                flat_idx = prompt_idx * self.num_samples + sample_idx
                token_count = int(outputs[flat_idx].shape[-1] - prompt_tokens_per_input[prompt_idx])
                token_count_group.append(token_count)
                generated_token_counts.append(token_count)
                sample_group.append({"generated_text": generated_texts[flat_idx]})
            generated_token_counts_per_input.append(token_count_group)
            grouped_texts.append(sample_group)

        self.last_generation_stats = {
            "prompt_tokens": prompt_tokens_per_input[0] if len(prompt_tokens_per_input) == 1 else None,
            "prompt_tokens_per_input": prompt_tokens_per_input,
            "max_new_tokens": int(max_new_tokens),
            "generated_token_counts": generated_token_counts,
            "generated_token_counts_per_input": generated_token_counts_per_input,
        }

        return grouped_texts[0] if single_input else grouped_texts

    def _aggregate_predictions(self, pred_list, fallback_value):
        pred_array = np.asarray(pred_list, dtype=float)
        valid_mask = ~np.isnan(pred_array)
        valid_columns = np.any(valid_mask, axis=0)

        prediction = np.full(pred_array.shape[1], fallback_value, dtype=float)
        if valid_columns.any():
            prediction[valid_columns] = np.nanmedian(pred_array[:, valid_columns], axis=0)

        return prediction

    def _deserialize_prediction_group(self, sample_group, current_pred_len):
        pred_list = []
        invalid_sample_outputs = 0

        for sample in sample_group:
            generated_text = sample["generated_text"]
            if "### Response:\n" not in generated_text:
                invalid_sample_outputs += 1
                pred = np.full(current_pred_len, np.nan)
                pred_list.append(pred)
                continue

            serialized_prediction = generated_text.split("### Response:\n", 1)[1]
            dispersed_prediction = self.serializer.inverse_serialize(serialized_prediction)
            if dispersed_prediction.size == 0:
                invalid_sample_outputs += 1
                pred = np.full(current_pred_len, np.nan)
                pred_list.append(pred)
                continue

            pred = self.discretizer.inverse_discretize(dispersed_prediction)

            if len(pred) < current_pred_len:
                pred = np.concatenate([pred, np.full(current_pred_len - len(pred), np.nan)])

            pred_list.append(pred[:current_pred_len])

        return pred_list, invalid_sample_outputs

    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        series = hist_data
        prediction_list = []
        remaining = self.pred_len
        total_prompt_tokens = 0
        total_generated_tokens = 0
        generation_calls = 0
        per_call_generated_token_counts = []
        invalid_sample_outputs = 0

        while remaining > 0:
            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)
            serialized_series = getPrompt(flag="prediction", context=context, input=serialized_series)
            current_pred_len = min(remaining, self.max_pred_len)

            samples = self._generate_prediction_samples(serialized_series, current_pred_len)
            generation_stats = self.last_generation_stats
            generation_calls += 1
            total_prompt_tokens += generation_stats.get("prompt_tokens", 0)
            token_counts = generation_stats.get("generated_token_counts", [])
            total_generated_tokens += sum(token_counts)
            per_call_generated_token_counts.extend(token_counts)

            pred_list = []
            for sample in samples:
                generated_text = sample["generated_text"]
                if "### Response:\n" not in generated_text:
                    invalid_sample_outputs += 1
                    pred = np.full(current_pred_len, np.nan)
                    pred_list.append(pred)
                    continue

                serialized_prediction = generated_text.split("### Response:\n", 1)[1]
                dispersed_prediction = self.serializer.inverse_serialize(serialized_prediction)
                if dispersed_prediction.size == 0:
                    invalid_sample_outputs += 1
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
        self.last_prediction_stats = {
            "generation_calls": generation_calls,
            "total_prompt_tokens": int(total_prompt_tokens),
            "total_generated_tokens": int(total_generated_tokens),
            "max_generated_tokens_per_sequence": int(max(per_call_generated_token_counts, default=0)),
            "avg_generated_tokens_per_sequence": (
                float(np.mean(per_call_generated_token_counts)) if per_call_generated_token_counts else 0.0
            ),
            "invalid_sample_outputs": int(invalid_sample_outputs),
        }

        return prediction

    def predict_batch(self, hist_data_batch, context_batch=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")
        if not hist_data_batch:
            return []

        batch_size = len(hist_data_batch)
        if context_batch is None:
            context_batch = [None] * batch_size
        if len(context_batch) != batch_size:
            raise ValueError("context_batch must match hist_data_batch length")

        series_batch = [np.array(hist_data, dtype=float, copy=True) for hist_data in hist_data_batch]
        prediction_chunks = [[] for _ in range(batch_size)]
        stats_batch = [
            {
                "generation_calls": 0,
                "total_prompt_tokens": 0,
                "total_generated_tokens": 0,
                "generated_token_counts": [],
                "invalid_sample_outputs": 0,
            }
            for _ in range(batch_size)
        ]

        remaining = self.pred_len
        while remaining > 0:
            current_pred_len = min(remaining, self.max_pred_len)
            prompt_batch = []
            for series, context in zip(series_batch, context_batch):
                dispersed_series = self.discretizer.discretize(series)
                serialized_series = self.serializer.serialize(dispersed_series)
                prompt_batch.append(getPrompt(flag="prediction", context=context, input=serialized_series))

            sample_groups = self._generate_prediction_samples(prompt_batch, current_pred_len)
            generation_stats = self.last_generation_stats
            prompt_tokens_per_input = generation_stats.get("prompt_tokens_per_input", [0] * batch_size)
            generated_token_counts_per_input = generation_stats.get(
                "generated_token_counts_per_input",
                [[] for _ in range(batch_size)],
            )

            predictions = []
            for idx, sample_group in enumerate(sample_groups):
                pred_list, invalid_sample_outputs = self._deserialize_prediction_group(sample_group, current_pred_len)
                prediction = self._aggregate_predictions(pred_list, fallback_value=series_batch[idx][-1])
                predictions.append(prediction)
                prediction_chunks[idx].append(prediction)

                stats_batch[idx]["generation_calls"] += 1
                stats_batch[idx]["total_prompt_tokens"] += prompt_tokens_per_input[idx]
                stats_batch[idx]["total_generated_tokens"] += sum(generated_token_counts_per_input[idx])
                stats_batch[idx]["generated_token_counts"].extend(generated_token_counts_per_input[idx])
                stats_batch[idx]["invalid_sample_outputs"] += invalid_sample_outputs

            remaining -= current_pred_len
            if remaining <= 0:
                break

            for idx, prediction in enumerate(predictions):
                series_batch[idx] = np.concatenate([series_batch[idx], prediction], axis=-1)

        batch_predictions = []
        finalized_stats = []
        for idx in range(batch_size):
            prediction = np.concatenate(prediction_chunks[idx], axis=-1)
            batch_predictions.append(prediction)
            generated_counts = stats_batch[idx]["generated_token_counts"]
            finalized_stats.append(
                {
                    "generation_calls": int(stats_batch[idx]["generation_calls"]),
                    "total_prompt_tokens": int(stats_batch[idx]["total_prompt_tokens"]),
                    "total_generated_tokens": int(stats_batch[idx]["total_generated_tokens"]),
                    "max_generated_tokens_per_sequence": int(max(generated_counts, default=0)),
                    "avg_generated_tokens_per_sequence": (
                        float(np.mean(generated_counts)) if generated_counts else 0.0
                    ),
                    "invalid_sample_outputs": int(stats_batch[idx]["invalid_sample_outputs"]),
                }
            )

        self.last_batch_prediction_stats = finalized_stats
        self.last_prediction_stats = finalized_stats[0]
        return batch_predictions

    def analyze(self, question, series):
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)
        serialized_series = getPrompt(flag="analysis", instruction=question, input=serialized_series)
        samples = self._generate_text(serialized_series, max_new_tokens=self.max_pred_len)

        response_list = []
        for sample in samples:
            response = sample["generated_text"].split("### Response:\n")[1].split('.')[0] + "."
            response = re.findall(r"\([abc]\)", response)[0]
            response_list.append(response)

        response = mode(response_list)

        return response
