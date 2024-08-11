import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2Config, PreTrainedTokenizerFast, AutoModelForCausalLM, AutoConfig, AdamW
from datasets import load_dataset
import datasets
from tqdm import tqdm
import random
import os
import re
import yaml
import json
import logging
import time
import gc
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file
from itertools import islice
from typing import Optional
from torch.optim import AdamW
import os
from torch.optim.lr_scheduler import LambdaLR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the cache directory for datasets
datasets.config.HF_DATASETS_CACHE = "./datasets"

def log_memory_usage(message: str, device: Optional[torch.device] = None):
    if device is None:
        device = torch.cuda.current_device()
    logging.info(f"{message}")
    logging.info(f"Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f}GB")
    logging.info(f"Cached: {torch.cuda.memory_reserved(device) / 1e9:.2f}GB")
    logging.info(f"Max allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f}GB")
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

def load_tokenizer(config):
    logger.info("Starting to load tokenizer")
    start_time = time.time()
    
    # Load tokenizer configuration
    try:
        with open(config.tokenizer_config_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
    except UnicodeDecodeError:
        with open(config.tokenizer_config_path, 'r', encoding='utf-8-sig') as f:
            tokenizer_config = json.load(f)

    # Extract added_tokens_decoder
    added_tokens_decoder = tokenizer_config.pop('added_tokens_decoder', {})

    # Initialize tokenizer
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.tokenizer_path, **tokenizer_config)
    except TypeError as e:
        logger.error(f"Error initializing tokenizer: {e}")
        logger.error("Tokenizer configuration:")
        for key, value in tokenizer_config.items():
            logger.error(f"{key}: {type(value)} - {value}")
        raise

    # Manually add the tokens from added_tokens_decoder
    for token_id, token_data in added_tokens_decoder.items():
        tokenizer.add_special_tokens({'additional_special_tokens': [token_data['content']]})

    # Set chat template if available
    if 'chat_template' in tokenizer_config:
        tokenizer.chat_template = tokenizer_config['chat_template']

    # Ensure all necessary tokens are set
    special_tokens_to_add = {}
    if tokenizer.unk_token is None:
        logger.info("UNK token was not set. It has been explicitly set to '<unk>'.")
        special_tokens_to_add['unk_token'] = '<unk>'

    if tokenizer.pad_token is None:
        logger.warning("PAD token is not set. Setting it to tokenizer.eos_token.")
        special_tokens_to_add['pad_token'] = tokenizer.eos_token

    if special_tokens_to_add:
        num_added = tokenizer.add_special_tokens(special_tokens_to_add)
        logger.info(f"Added {num_added} special tokens to the tokenizer.")

    # Log tokenizer information
    logger.info(f"Tokenizer loaded: vocabulary size = {len(tokenizer)}, max length = {tokenizer.model_max_length}")
    for token_type in ['bos', 'eos', 'pad', 'unk']:
        token_value = getattr(tokenizer, f'{token_type}_token', None)
        logger.info(f"{token_type.upper()} token: {token_value}")
    
    # Check if BOS and EOS tokens are set
    logger.info(f"BOS token is set: {tokenizer.bos_token is not None}")
    logger.info(f"EOS token is set: {tokenizer.eos_token is not None}")

    logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
    return tokenizer, num_added

class EnhancedMathReasoningDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if 'question' in item:  # GSM8K format
            question = item['question']
            answer = item['answer']
        elif 'problem' in item:  # NuminaMath format
            question = item['problem']
            answer = item['solution']
        else:
            question = item['text'].split('\n')[0]
            answer = '\n'.join(item['text'].split('\n')[1:])

        encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'question': question,
            'answer': answer
        }

class DynamicBatchSampler:
    def __init__(self, dataset, max_tokens, max_batch_size):
        logger.info(f"Initializing DynamicBatchSampler with max_tokens={max_tokens}, max_batch_size={max_batch_size}")
        start_time = time.time()
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.lengths = [len(dataset[i]['input_ids']) for i in tqdm(range(len(dataset)), desc="Calculating lengths")]
        logger.info(f"DynamicBatchSampler initialized in {time.time() - start_time:.2f} seconds")

    def __iter__(self):
        logger.info("Starting new iteration of DynamicBatchSampler")
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch = []
        batch_size = 0
        for idx in indices:
            batch.append(idx)
            batch_size += self.lengths[idx]
            if batch_size >= self.max_tokens or len(batch) >= self.max_batch_size:
                logger.debug(f"Yielding batch of size {len(batch)} with {batch_size} tokens")
                yield batch
                batch = []
                batch_size = 0
        if batch:
            logger.debug(f"Yielding final batch of size {len(batch)} with {batch_size} tokens")
            yield batch

    def __len__(self):
        return (sum(self.lengths) + self.max_tokens - 1) // self.max_tokens

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class EnhancedPEERLayer(nn.Module):
    def __init__(self, d_model: int, n_experts: int, n_heads: int, top_k: int, d_key: int):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_heads = n_heads
        self.top_k = top_k
        self.d_key = d_key

        self.query_net = nn.Linear(d_model, n_heads * d_key)
        self.key_net = nn.Linear(d_model, n_heads * d_key)
        n_sub_keys = int(math.sqrt(n_experts))
        self.sub_keys = nn.Parameter(torch.randn(2, n_sub_keys, d_key // 2))
        self.expert_weights = nn.Parameter(torch.randn(n_experts, d_key))
        self.layer_norm = RMSNorm(d_model)
        self.query_bn = nn.BatchNorm1d(n_heads * d_key)
        self.output_proj = nn.Linear(n_heads * d_key, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads)

    def forward(self, x):
        b, t, _ = x.shape
        queries = self.query_net(x)
        keys = self.key_net(x)
        queries = self.query_bn(queries.view(b * t, -1)).view(b, t, self.n_heads, self.d_key)
        
        indices = self.get_indices(queries)
        expert_weights = self.expert_weights[indices]
        
        similarities = torch.sum(queries.unsqueeze(-2) * expert_weights, dim=-1)
        router_weights = F.softmax(similarities, dim=-1)
        
        outputs = torch.sum(router_weights.unsqueeze(-1) * expert_weights, dim=-2)
        outputs = self.output_proj(outputs.view(b, t, -1))
        
        attn_output, attn_weights = self.attention(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))
        return self.layer_norm(x + outputs + attn_output.transpose(0, 1)), attn_weights

    def get_indices(self, queries):
        b, t, h, d = queries.shape
        queries = queries.view(b * t * h, 1, 2, -1)
        keys1, keys2 = self.sub_keys[0], self.sub_keys[1]
        
        scores1 = torch.matmul(queries[:, :, 0], keys1.t())
        scores2 = torch.matmul(queries[:, :, 1], keys2.t())
        
        scores = (scores1.unsqueeze(-1) + scores2.unsqueeze(-2)).view(b * t * h, -1)
        
        _, indices = torch.topk(scores, k=self.top_k, dim=-1)
        return indices.view(b, t, h, self.top_k)

class EnhancedMambaBlock(nn.Module):
    def __init__(self, d_model: int, n_experts: int, n_heads: int, top_k: int, d_key: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.peer = EnhancedPEERLayer(d_model, n_experts, n_heads, top_k, d_key)
        self.gate = nn.Linear(d_model, d_model)
        self.layer_norm = RMSNorm(d_model)
        self.step_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        peer_out, attn_weights = self.peer(conv_out)
        gate = torch.sigmoid(self.gate(x))
        main_out = self.layer_norm(x + gate * peer_out)
        
        step_out = torch.cat([self.step_token.expand(x.size(0), -1, -1), main_out[:, :-1, :]], dim=1)
        return main_out + 0.1 * step_out, attn_weights

class EnhancedPEERLanguageModel(nn.Module):
    def __init__(self, config, n_experts: int, n_heads: int, top_k: int, d_key: int, share_params: bool = False):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Embedding(config.n_positions, config.n_embd)
        
        block = EnhancedMambaBlock(config.n_embd, n_experts, n_heads, top_k, d_key)
        self.layers = nn.ModuleList([block for _ in range(config.n_layer)] if share_params else 
                                    [EnhancedMambaBlock(config.n_embd, n_experts, n_heads, top_k, d_key) for _ in range(config.n_layer)])
        
        self.hidden_proj = nn.Linear(config.n_embd, 4096)
        
        self.output = nn.Linear(4096, config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(b, -1)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        
        attention_maps = []
        layer_outputs = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attention_maps.append(attn_weights)
            layer_outputs.append(x)
        
        x = self.hidden_proj(x)
        return self.output(x), attention_maps, layer_outputs

    def generate(self, input_ids, attention_mask=None, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, do_sample=True, top_k=50, top_p=0.95, temperature=0.7):
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            outputs = self.forward(generated, attention_mask=attention_mask)
            next_token_logits = outputs[0][:, -1, :]
            
            if do_sample:
                # Temperature
                next_token_logits = next_token_logits / temperature
                # Top-k
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                # Top-p
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
            
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)
            
            # No repeat ngram
            if no_repeat_ngram_size > 0:
                for i in range(batch_size):
                    gen_tokens = generated[i].tolist()
                    for ngram in zip(*[gen_tokens[j:] for j in range(no_repeat_ngram_size)]):
                        if gen_tokens[-no_repeat_ngram_size:] == list(ngram):
                            generated[i, -1] = torch.randint(self.config.vocab_size, (1,)).to(generated.device)
            
            # Early stopping if all sequences have reached the end token
            if (generated[:, -1] == self.config.eos_token_id).all():
                break
        
        return generated

    def resize_token_embeddings(self, new_num_tokens):
        if new_num_tokens == self.config.vocab_size:
            return

        self.embedding = nn.Embedding(new_num_tokens, self.config.n_embd)
        self.output = nn.Linear(4096, new_num_tokens)
        self.config.vocab_size = new_num_tokens

class TeacherModel(nn.Module):
    def __init__(self, model_files):
        super().__init__()
        log_memory_usage("Start of TeacherModel initialization")
        if len(model_files) != 3:
            raise ValueError(f"Expected 3 model files, but found {len(model_files)}.")
        
        # Initialize with DeepSeek Math 7B configuration
        config = AutoConfig.from_pretrained("deepseek-ai/deepseek-math-7b-base")
        self.model = AutoModelForCausalLM.from_config(config)
        
        logger.info("Loading 3-part sharded model")
        state_dict = self.load_3part_sharded_model(model_files)
        
        # Load the state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Model loaded. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
        self.model.eval()
        log_memory_usage("End of TeacherModel initialization")

    def load_3part_sharded_model(self, model_files):
        state_dict = {}
        for file in model_files:
            logger.info(f"Loading weights from {file}")
            state_dict.update(load_file(file))
        return state_dict

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_length=100):
        log_memory_usage("Start of TeacherModel generate")
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        log_memory_usage("End of TeacherModel generate")
        return output

    def forward(self, input_ids, attention_mask=None):
        log_memory_usage("Start of TeacherModel forward")
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        log_memory_usage("End of TeacherModel forward")
        return outputs.logits, outputs.hidden_states

class DistillationTrainer:
    def __init__(self, student_model, teacher_model, train_loader, val_loader, tokenizer, config):
        self.student_model = student_model.to(config.device)
        self.teacher_model = teacher_model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config

        self.optimizer = AdamW(
            self.student_model.parameters(),
            lr=float(config.learning_rate),
            betas=tuple(config.optimizer['betas']),
            eps=float(config.optimizer['eps']),
            weight_decay=config.weight_decay
        )

        self.lr_scheduler = self.get_lr_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)
        
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.layer_wise_loss = nn.SmoothL1Loss()
        self.step = 0
        
        self.writer = SummaryWriter(log_dir=config.log_dir)

    def get_lr_scheduler(self):
        def lr_lambda(current_step: int):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            return max(
                0.0, float(self.config.max_steps - current_step) / float(max(1, self.config.max_steps - self.config.warmup_steps))
            )
        return LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch):
        self.student_model.train()
        self.teacher_model.eval()
        
        input_ids = batch['input_ids'].to(self.config.device)
        attention_mask = batch['attention_mask'].to(self.config.device)

        with torch.no_grad():
            teacher_logits, teacher_hidden_states = self.teacher_model(input_ids, attention_mask)

        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            student_logits, _, student_hidden_states = self.student_model(input_ids, attention_mask)
            student_hidden_states = [self.student_model.hidden_proj(h) for h in student_hidden_states]

            distillation_loss = self.kl_div_loss(
                F.log_softmax(student_logits / self.config.temperature, dim=-1),
                F.softmax(teacher_logits / self.config.temperature, dim=-1)
            ) * (self.config.temperature ** 2)
        
            layer_wise_loss = sum(self.layer_wise_loss(s, t) for s, t in zip(student_hidden_states, teacher_hidden_states))
        
            loss = distillation_loss + self.config.layer_wise_loss_weight * layer_wise_loss

        self.scaler.scale(loss).backward()
        
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()

        self.step += 1
        return loss.item()

    def train(self, num_epochs):
        best_val_accuracy = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                loss = self.train_step(batch)
                epoch_loss += loss
                
                self.writer.add_scalar('Loss/train', loss, self.step)
                self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], self.step)
                
                if self.step >= self.config.max_steps:
                    break
            
            val_loss, val_accuracy = self.validate()
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(self.train_loader):.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_checkpoint(epoch, epoch_loss / len(self.train_loader))
            
            if self.step >= self.config.max_steps:
                break

        self.writer.close()

        @torch.no_grad()
        def validate(self):
            self.student_model.eval()
            self.teacher_model.eval()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                
                # Generate student's answer
                student_output = self.student_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=200,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7
                )
                
                student_answers = [self.tokenizer.decode(output, skip_special_tokens=True) for output in student_output] 
                
                # Prepare prompts for teacher evaluation
                evaluation_prompts = []
                for question, answer in zip(batch['question'], student_answers):
                    prompt = f"Question: {question}\nStudent's answer: {answer}\nIs this answer correct? Respond with 'Correct' or 'Incorrect' and explain why."
                    evaluation_prompts.append(prompt)
                
                # Get teacher's evaluation
                teacher_input = self.tokenizer(evaluation_prompts, return_tensors="pt", padding=True, truncation=True).to(self.config.device)
                teacher_output = self.teacher_model.generate(
                    input_ids=teacher_input.input_ids,
                    attention_mask=teacher_input.attention_mask,
                    max_length=100,
                    num_return_sequences=1
                )
                
                teacher_evaluations = [self.tokenizer.decode(output, skip_special_tokens=True) for output in teacher_output]
                
                # Count correct predictions
                for evaluation in teacher_evaluations:
                    if "Correct" in evaluation:
                        correct_predictions += 1
                    total_predictions += 1
                
                # Calculate loss (optional, depending on your needs)
                student_logits, _, _ = self.student_model(input_ids, attention_mask)
                loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), input_ids.view(-1))
                total_loss += loss.item()
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            return total_loss / len(self.val_loader), accuracy

    def save_checkpoint(self, epoch, loss):
        checkpoint_path = f'{self.config.checkpoint_dir}/checkpoint_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss': loss,
            'step': self.step,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.step = checkpoint['step']
        return checkpoint['epoch']

    def adaptive_layer_freezing(self):
        num_layers = len(self.student_model.layers)
        layers_to_freeze = max(0, self.config.initial_frozen_layers - (self.step // self.config.unfreeze_interval))
        
        for i, layer in enumerate(self.student_model.layers):
            for param in layer.parameters():
                param.requires_grad = (i >= layers_to_freeze)

    def increase_sequence_length(self):
        if self.current_seq_length < self.max_seq_length:
            self.current_seq_length = min(
                self.current_seq_length + self.config.sequence_length_increase,
                self.max_seq_length
            )
            logger.info(f"Increased sequence length to {self.current_seq_length}")

def setup_data_and_model(config):
    tokenizer, num_added_tokens = load_tokenizer(config)
    config.vocab_size = len(tokenizer)

    datasets = {
        'numina_cot': load_dataset("AI-MO/NuminaMath-CoT"),
        'gsm8k': load_dataset("gsm8k", "main")
    }

    processed_datasets = {}
    for name, dataset in datasets.items():
        if config.subset_size > 0:
            subset_size = min(config.subset_size, len(dataset['train']))
            subset_data = list(islice(dataset['train'], subset_size))
            processed_datasets[name] = EnhancedMathReasoningDataset(subset_data, tokenizer, max_length=config.n_positions)
        else:
            processed_datasets[name] = EnhancedMathReasoningDataset(dataset['train'], tokenizer, max_length=config.n_positions)

    train_loaders, val_loaders = {}, {}
    for name, dataset in processed_datasets.items():
        train_size = int(0.9 * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        
        train_sampler = DynamicBatchSampler(train_dataset, max_tokens=config.max_tokens_per_batch, max_batch_size=config.max_batch_size)
        val_sampler = DynamicBatchSampler(val_dataset, max_tokens=config.max_tokens_per_batch, max_batch_size=config.max_batch_size)
        
        train_loaders[name] = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=config.dataloader['num_workers'], pin_memory=config.dataloader['pin_memory'])
        val_loaders[name] = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=config.dataloader['num_workers'], pin_memory=config.dataloader['pin_memory'])

    student_model = EnhancedPEERLanguageModel(config, config.n_experts, config.n_heads, config.top_k, config.d_key, share_params=config.share_params)
    
    if num_added_tokens > 0:
        student_model.resize_token_embeddings(config.vocab_size)

    teacher_model = TeacherModel(config.teacher_model['files'])
    teacher_model.model.resize_token_embeddings(config.vocab_size)

    return student_model, teacher_model, tokenizer, train_loaders, val_loaders

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config = type('Config', (), config)()
    
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    if config.logging['log_to_file']:
        file_handler = logging.FileHandler(config.logging['log_file'])
        file_handler.setLevel(getattr(logging, config.logging['level']))
        logger.addHandler(file_handler)
    
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    device = torch.device(config.device)
    config.device = device

    student_model, teacher_model, tokenizer, train_loaders, val_loaders = setup_data_and_model(config)
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    for dataset in config.datasets:
        logger.info(f"Starting training on {dataset} dataset")
        
        if dataset not in train_loaders or dataset not in val_loaders:
            logger.error(f"Dataset '{dataset}' not found in train_loaders or val_loaders. Skipping.")
            continue
        
        train_loader = train_loaders[dataset]
        val_loader = val_loaders[dataset]

        trainer = DistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            config=config
        )

        checkpoint_path = f'{config.checkpoint_dir}/checkpoint_{dataset}_latest.pth'
        if os.path.exists(checkpoint_path):
            start_epoch = trainer.load_checkpoint(checkpoint_path)
        else:
            start_epoch = 0

        logger.info(f"Training for {config.num_epochs[dataset] - start_epoch} epochs")
        try:
            trainer.train(config.num_epochs[dataset] - start_epoch)
        except Exception as e:
            logger.error(f"Error during training on {dataset}: {e}")
            logger.exception("Detailed traceback:")
            continue

        val_loss, val_accuracy = trainer.validate()
        logger.info(f"Validation after {dataset} - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        logger.info(f"Saving model after training on {dataset}")
        torch.save(student_model.state_dict(), f'{config.checkpoint_dir}/model_after_{dataset}.pth')

        gc.collect()
        torch.cuda.empty_cache()

    final_loss, final_accuracy = trainer.validate()
    logger.info(f"Final Validation - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")

    logger.info("Saving final model")
    torch.save(student_model.state_dict(), f'{config.checkpoint_dir}/final_math_model.pth')

if __name__ == "__main__":
    main()
