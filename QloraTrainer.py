import logging
import math

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
from transformers.integrations import rewrite_logs
import numpy as np

import mlfoundry

from data_processor.RawTextDataProcessor import RawTextDataProcessor
from data_processor.VicunaDataProcessor import VicunaDataProcessor


class Callback(TrainerCallback):
    def __init__(
        self,
        run: mlfoundry.MlFoundryRun,
    ):
        self._run = run

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not state.is_world_process_zero:
            return

        for loss_key, perplexity_key in [
            ("loss", "train_perplexity"),
            ("eval_loss", "eval_perplexity"),
        ]:
            if loss_key in logs:
                try:
                    perplexity = math.exp(logs[loss_key])
                except OverflowError:
                    perplexity = float("inf")
                    logging.warning(
                        "Encountered inf in eval perplexity, cannot log it as a metric"
                    )
                logs[perplexity_key] = perplexity

        metrics = {}
        for k, v in logs.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and math.isfinite(
                v
            ):
                metrics[k] = v
            else:
                logging.warning(
                    f'Trainer is attempting to log a value of "{v}" of'
                    f' type {type(v)} for key "{k}" as a metric.'
                    " Mlfoundry's log_metric() only accepts finite float and"
                    " int types so we dropped this attribute."
                )
        try:
            self._run.log_metrics(rewrite_logs(metrics), step=state.global_step)
        except Exception:
            logging.info("Error raised while publishing logs to mlfoundry")


class QloraTrainer:
    def __init__(self, config: dict, run: mlfoundry.MlFoundryRun):
        self.config = config
        self.tokenizer = None
        self.base_model = None
        self.adapter_model = None
        self.merged_model = None
        self.data_processor = None
        self._run = run

    def load_base_model(self):
        model_id = self.config["base_model"]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        if "model_family" in self.config and self.config["model_family"] == "llama":
            tokenizer = LlamaTokenizer.from_pretrained(model_id)
            model = LlamaForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, device_map={"": 0}
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, device_map={"": 0}
            )

        if not tokenizer.pad_token:
            # Add padding token if missing, e.g. for llama tokenizer
            # tokenizer.pad_token = tokenizer.eos_token  # https://github.com/huggingface/transformers/issues/22794
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        self.tokenizer = tokenizer
        self.base_model = model

    def load_adapter_model(self, adapter_path: str):
        """Load pre-trained lora adapter"""
        self.adapter_model = PeftModel.from_pretrained(self.base_model, adapter_path)

    def train(self):
        # Set up lora config or load pre-trained adapter
        if self.adapter_model is None:
            config_dict = self.config["lora"]
            config = LoraConfig(
                r=config_dict["r"],
                lora_alpha=config_dict["lora_alpha"],
                target_modules=config_dict["target_modules"],
                lora_dropout=config_dict["lora_dropout"],
                bias=config_dict["bias"],
                task_type=config_dict["task_type"],
            )
            model = get_peft_model(self.base_model, config)
        else:
            model = self.adapter_model
        self._print_trainable_parameters(model)

        print("Start data preprocessing")
        self._setup_data_processor()
        data = self.data_processor.get_data()

        print("Start training")
        config_dict = self.config["trainer"]
        train_data = (
            data["train"]
            .shuffle(seed=42)
            .select(range(int(math.ceil(len(data["train"]) * 0.7))))
        )
        print(f"Training data points: {len(train_data)}")

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=config_dict["batch_size"],
                gradient_accumulation_steps=config_dict["gradient_accumulation_steps"],
                warmup_steps=config_dict["warmup_steps"],
                num_train_epochs=config_dict["num_train_epochs"],
                learning_rate=config_dict["learning_rate"],
                fp16=True,
                logging_steps=config_dict["logging_steps"],
                output_dir=self.config["trainer_output_dir"],
                report_to="tensorboard",
                # optim="adamw"
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
            callbacks=[Callback(run=self._run)],
        )
        model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )
        trainer.train()

        model_save_path = (
            f"{self.config['model_output_dir']}/{self.config['model_name']}_adapter"
        )
        trainer.save_model(model_save_path)
        self.adapter_model = model
        print(f"Training complete, adapter model saved in {model_save_path}")

    def merge_and_save(self):
        """Merge base model and adapter, save to disk"""
        # Cannot merge when base model loaded in 8-bit/4-bit mode, so load separately
        model_id = self.config["base_model"]
        if "model_family" in self.config and self.config["model_family"] == "llama":
            base_model = LlamaForCausalLM.from_pretrained(model_id, device_map="cpu")
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="cpu"
            )

        adapter_save_path = (
            f"{self.config['model_output_dir']}/{self.config['model_name']}_adapter"
        )
        model = PeftModel.from_pretrained(base_model, adapter_save_path)

        self.merged_model = (
            model.merge_and_unload()
        )  # note it's on CPU, don't run inference on it

        model_save_path = (
            f"{self.config['model_output_dir']}/{self.config['model_name']}"
        )
        self.merged_model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)

        pl = pipeline(
            "text-generation",
            model=self.merged_model,
            tokenizer=self.tokenizer,
        )
        self._run.log_model(name="llama-2", model=pl, framework="transformers")

    def push_to_hub(self):
        """Push merged model to HuggingFace Hub"""
        raise NotImplementedError("push_to_hub not implemented yet")

    def _print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _setup_data_processor(self):
        if self.config["data"]["type"] == "vicuna":
            self.data_processor = VicunaDataProcessor(self.config, self.tokenizer)
        elif self.config["data"]["type"] == "raw_text":
            self.data_processor = RawTextDataProcessor(self.config, self.tokenizer)
        else:
            raise ValueError("Dataset type not specified in config.data.type")
