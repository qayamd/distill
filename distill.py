import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2Config, PreTrainedTokenizerFast, AutoModelForCausalLM, AutoConfig
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
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub
from safetensors.torch import load_file
from modifiedadam import Adam_mini
import torch.nn.utils.prune as prune
from transformers import PreTrainedTokenizerFast
from transformers import AutoModelForCausalLM, AutoConfig
from itertools import islice

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the cache directory for datasets
datasets.config.HF_DATASETS_CACHE = "./datasets"

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
    if tokenizer.unk_token is None:
        logger.warning("UNK token is not set. Setting it to '<unk>'.")
        tokenizer.add_special_tokens({'unk_token': '<unk>'})

    if tokenizer.pad_token is None:
        logger.warning("PAD token is not set. Setting it to tokenizer.eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Log tokenizer information
    logger.info(f"Tokenizer loaded: vocabulary size = {len(tokenizer)}, max length = {tokenizer.model_max_length}")
    for token_type in ['bos', 'eos', 'pad', 'unk']:
        token_value = getattr(tokenizer, f'{token_type}_token', None)
        logger.info(f"{token_type.upper()} token: {token_value}")
    
    # Check if BOS and EOS tokens are set
    logger.info(f"BOS token is set: {tokenizer.bos_token is not None}")
    logger.info(f"EOS token is set: {tokenizer.eos_token is not None}")

    logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
    return tokenizer

class EnhancedMathReasoningDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        logger.info(f"Initializing EnhancedMathReasoningDataset with {len(dataset)} items")
        start_time = time.time()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"EnhancedMathReasoningDataset initialized in {time.time() - start_time:.2f} seconds")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        start_time = time.time()
        item = self.dataset[idx]
        if 'question' in item:  # GSM8K format
            dialog = f"Question: {item['question']}\nAnswer: {item['answer']}"
        elif 'problem' in item:  # NuminaMath format
            dialog = f"Problem: {item['problem']}\nSolution: {item['solution']}"
        else:
            dialog = item['text']
        
        encoding = self.tokenizer(
            dialog,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        target = self.extract_number(item.get('answer') or item.get('solution') or item['text'])
        
        logger.debug(f"Processed item {idx} in {time.time() - start_time:.4f} seconds")
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': torch.tensor(target, dtype=torch.float) if target is not None else torch.tensor(float('nan'))
        }

    @staticmethod
    def extract_number(text):
        match = re.search(r'-?\d+\.?\d*', text.split('\n')[-1])
        return float(match.group()) if match else None

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
        
        self.output = nn.Linear(config.n_embd, config.vocab_size)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(b, -1)
        x = self.quant(self.embedding(input_ids) + self.pos_embedding(pos))
        
        attention_maps = []
        layer_outputs = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attention_maps.append(attn_weights)
            layer_outputs.append(x)
        
        x = self.dequant(x)
        return self.output(x), attention_maps, layer_outputs

    def resize_token_embeddings(self, new_num_tokens):
        if new_num_tokens == self.config.vocab_size:
            return

        self.embedding = nn.Embedding(new_num_tokens, self.config.n_embd)
        self.output = nn.Linear(self.config.n_embd, new_num_tokens)
        self.config.vocab_size = new_num_tokens

class TeacherModel(nn.Module):
    def __init__(self, model_files):
        super().__init__()
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

    def load_3part_sharded_model(self, model_files):
        state_dict = {}
        for file in model_files:
            logger.info(f"Loading weights from {file}")
            state_dict.update(load_file(file))
        return state_dict

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_length=100):
        return self.model.generate(
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

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return outputs.logits, outputs.hidden_states

class DistillationTrainer:
    def __init__(self, student_model, teacher_model, train_loader, val_loader, tokenizer, config):
        self.student_model = student_model.to(config.device)
        self.teacher_model = teacher_model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config

        self.optimizer = Adam_mini(
            model=self.student_model,
            lr=float(config.learning_rate),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            model_sharding=config.model_sharding,
            n_feature=config.n_embd,
            n_head=config.n_heads,
            n_kv_head=config.n_kv_heads 
        )

        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=config.T_0, T_mult=config.T_mult)
        self.scaler = GradScaler()
        
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        self.step = 0
        
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        self.pruning_amount = 0.3  # Prune 30% of connections
        self.current_seq_length = config.initial_sequence_length
        self.max_seq_length = config.n_positions

    def train_step(self, batch):
        self.student_model.train()
        self.teacher_model.eval()
        
        input_ids = batch['input_ids'][:, :self.current_seq_length].to(self.config.device)
        attention_mask = batch['attention_mask'][:, :self.current_seq_length].to(self.config.device)

        # Adaptive Layer Freezing
        self.adaptive_layer_freezing()

        with torch.no_grad():
            teacher_logits, teacher_hidden_states = self.teacher_model(input_ids, attention_mask)

        with autocast():
            student_logits, _, student_hidden_states = self.student_model(input_ids, attention_mask)

            # KL Divergence Loss
            distillation_loss = self.kl_div_loss(
                F.log_softmax(student_logits / self.config.temperature, dim=-1),
                F.softmax(teacher_logits / self.config.temperature, dim=-1)
            ) * (self.config.temperature ** 2)

            # Layer-wise Distillation Loss
            layer_wise_loss = sum(self.mse_loss(s, t) for s, t in zip(student_hidden_states, teacher_hidden_states))
            
            loss = distillation_loss + self.config.layer_wise_loss_weight * layer_wise_loss
            loss = loss / self.config.gradient_accumulation_steps

        self.scaler.scale(loss).backward()
        
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Model Pruning
            if self.step % self.config.pruning_interval == 0:
                self.prune_model()

        self.step += 1
        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def validate(self):
        self.student_model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.config.device)
            attention_mask = batch['attention_mask'].to(self.config.device)
            targets = batch['target'].to(self.config.device)
            
            with autocast():
                student_logits, _, _ = self.student_model(input_ids, attention_mask)
                loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            
            pred_tokens = student_logits.argmax(dim=-1)
            preds = [self.extract_number(self.tokenizer.decode(tokens)) for tokens in pred_tokens]
            
            all_preds.extend([p for p in preds if p is not None])
            all_targets.extend([t.item() for t in targets if not torch.isnan(t)])
        
        mse = mean_squared_error(all_targets, all_preds) if all_preds else float('inf')
        accuracy = accuracy_score([round(t) for t in all_targets], [round(p) for p in all_preds]) if all_preds else 0.0
        
        return total_loss / len(self.val_loader), mse, accuracy

    @staticmethod
    def extract_number(text):
        match = re.search(r'-?\d+\.?\d*', text.split('\n')[-1])
        return float(match.group()) if match else None

    def save_checkpoint(self, epoch, loss):
        checkpoint_path = f'{self.config.checkpoint_dir}/checkpoint_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'step': self.step,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        return checkpoint['epoch']

    def train(self, num_epochs):
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                loss = self.train_step(batch)
                epoch_loss += loss
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
                
                self.writer.add_scalar('Loss/train', loss, self.step)
                self.writer.add_scalar('LearningRate', self.scheduler.get_last_lr()[0], self.step)
                
                # Progressive Resizing
                if self.step % self.config.resize_interval == 0:
                    self.increase_sequence_length()
                
                if self.step >= self.config.max_steps:
                    break

            epoch_loss /= len(self.train_loader)
            val_loss, val_mse, val_accuracy = self.validate()
            
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('MSE/val', val_mse, epoch)
            self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, epoch_loss)
            
            if self.step >= self.config.max_steps:
                break

        self.writer.close()

    def adaptive_layer_freezing(self):
        # Implement a strategy to freeze and unfreeze layers
        num_layers = len(self.student_model.layers)
        layers_to_freeze = int(num_layers * (1 - self.step / self.config.max_steps))
        
        for i, layer in enumerate(self.student_model.layers):
            for param in layer.parameters():
                param.requires_grad = (i >= layers_to_freeze)

    def prune_model(self):
        for name, module in self.student_model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.pruning_amount)
                prune.remove(module, 'weight')

    def increase_sequence_length(self):
        if self.current_seq_length < self.max_seq_length:
            self.current_seq_length = min(
                self.current_seq_length + self.config.sequence_length_increase,
                self.max_seq_length
            )
            logger.info(f"Increased sequence length to {self.current_seq_length}")


def setup_data_and_model(config):
    logger.info("Starting setup_data_and_model")
    start_time = time.time()

    tokenizer = load_tokenizer(config)
    
    # Set the vocab_size based on the tokenizer
    config.vocab_size = len(tokenizer)
    logger.info(f"Setting vocab_size to {config.vocab_size} based on tokenizer")

    logger.info("Loading datasets")
    datasets = {
        'numina_cot': load_dataset("AI-MO/NuminaMath-CoT")
    }

    processed_datasets = {}
    for name, dataset in datasets.items():
        logger.info(f"Processing {name} dataset")
        try:
            start_process_time = time.time()
            
            # Use a subset of the data if specified in config
            if hasattr(config, 'subset_size') and config.subset_size > 0:
                subset_size = min(config.subset_size, len(dataset['train']))
                logger.info(f"Using a subset of {subset_size} examples from {name} dataset")
                subset_data = list(islice(dataset['train'], subset_size))
                processed_datasets[name] = EnhancedMathReasoningDataset(subset_data, tokenizer, max_length=config.n_positions)
            else:
                processed_datasets[name] = EnhancedMathReasoningDataset(dataset['train'], tokenizer, max_length=config.n_positions)
            
            logger.info(f"{name} dataset processed in {time.time() - start_process_time:.2f} seconds. Size: {len(processed_datasets[name])}")
        except Exception as e:
            logger.error(f"Error processing {name} dataset: {e}")
            continue

    train_loaders, val_loaders = {}, {}
    for name, dataset in processed_datasets.items():
        logger.info(f"Creating data loaders for {name}")
        start_loader_time = time.time()
        train_size = int(0.9 * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        
        logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        
        try:
            logger.info("Initializing train sampler")
            train_sampler = DynamicBatchSampler(train_dataset, max_tokens=config.max_tokens_per_batch, max_batch_size=config.max_batch_size)
            logger.info("Initializing validation sampler")
            val_sampler = DynamicBatchSampler(val_dataset, max_tokens=config.max_tokens_per_batch, max_batch_size=config.max_batch_size)
            
            logger.info("Creating DataLoader for train dataset")
            train_loaders[name] = DataLoader(train_dataset, batch_sampler=train_sampler)
            logger.info("Creating DataLoader for validation dataset")
            val_loaders[name] = DataLoader(val_dataset, batch_sampler=val_sampler)
            
            logger.info(f"Data loaders created for {name} in {time.time() - start_loader_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error creating data loaders for {name}: {e}")
            continue

    logger.info(f"Initializing student model with vocab size: {config.vocab_size}")
    model = EnhancedPEERLanguageModel(config, config.n_experts, config.n_heads, config.top_k, config.d_key, share_params=config.share_params)
    
    logger.info("Loading teacher model")
    safetensor_files = [f for f in os.listdir() if f.endswith('.safetensors')]
    safetensor_files.sort()
    if len(safetensor_files) != 3:
        raise ValueError(f"Found {len(safetensor_files)} SafeTensors files. Expected exactly 3.")
    logger.info(f"Found SafeTensors files: {', '.join(safetensor_files)}")
    teacher_model = TeacherModel(safetensor_files)

    # Ensure the teacher model uses the same vocabulary size
    teacher_model.model.resize_token_embeddings(config.vocab_size)
    logger.info(f"Adjusted teacher model vocab size to {config.vocab_size}")

    logger.info(f"setup_data_and_model completed in {time.time() - start_time:.2f} seconds")
    return model, teacher_model, tokenizer, train_loaders, val_loaders

def main():
    logger.info("Starting main function")
    start_time = time.time()

    # Load configuration from YAML file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert config to an object for easier access
    config = type('Config', (), config)()
    
    # Set up logging
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    if torch.cuda.is_available():
        total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
        #config.max_tokens_per_batch = int(total_gpu_memory * 0.4 // (config.n_positions * 4))
        config.max_tokens_per_batch = int(1024**2)
    else:
        config.max_tokens_per_batch = 1024  # Default value for CPU

    logger.info("Setting up data and model")
    model, teacher_model, tokenizer, train_loaders, val_loaders = setup_data_and_model(config)
    model = model.to(device)
    teacher_model = teacher_model.to(device)

    for dataset in config.datasets:
        logger.info(f"Starting training on {dataset} dataset")
        train_loader = train_loaders[dataset]
        val_loader = val_loaders[dataset]

        trainer = DistillationTrainer(
            student_model=model,
            teacher_model=teacher_model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            config=config
        )

        checkpoint_path = f'{config.checkpoint_dir}/checkpoint_{dataset}_latest.pth'
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            start_epoch = trainer.load_checkpoint(checkpoint_path)
        else:
            logger.info("No checkpoint found, starting from epoch 0")
            start_epoch = 0

        logger.info(f"Training for {config.num_epochs[dataset] - start_epoch} epochs")
        trainer.train(config.num_epochs[dataset] - start_epoch)

        val_loss, val_mse, val_accuracy = trainer.validate()
        logger.info(f"Validation after {dataset} - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, Accuracy: {val_accuracy:.4f}")

        logger.info(f"Saving model after training on {dataset}")
        torch.save(model.state_dict(), f'{config.checkpoint_dir}/model_after_{dataset}.pth')

    final_loss, final_mse, final_accuracy = trainer.validate()
    logger.info(f"Final Validation - Loss: {final_loss:.4f}, MSE: {final_mse:.4f}, Accuracy: {final_accuracy:.4f}")

    logger.info("Saving final model")
    torch.save(model.state_dict(), f'{config.checkpoint_dir}/final_math_model.pth')

    logger.info(f"Main function completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
