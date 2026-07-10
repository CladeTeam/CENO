# coding=utf-8
# Copyright (c) 2025, Arc Institute. All rights reserved.
# Copyright (c) 2026, CENO Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Byte-level (character-level) tokenizer for CENO."""

import json
import os
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import torch
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class CENOCharLevelTokenizer(PreTrainedTokenizer):
    """
    HuggingFace-style byte-level (character-level) tokenizer for CENO.
    
    This tokenizer converts text directly to byte values using numpy's fromstring,
    which is perfect for DNA sequences and other character-level tasks.
    
    Args:
        vocab_size (int): Size of the vocabulary (default: 512)
        eos_token (str): End of sequence token
        pad_token (str): Padding token  
        unk_token (str): Unknown token
        **kwargs: Additional arguments passed to PreTrainedTokenizer
    """
    
    vocab_files_names = VOCAB_FILES_NAMES
    
    def __init__(
        self,
        vocab_size: int = 512,
        eos_token: str = "<eos>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        **kwargs
    ):
        self._vocab_size = vocab_size
        self.eod_id = 0
        self.eos_id = 0
        self.pad_id = 1
        self.unk_id = 2
        
        # Build vocabulary - builds the CENO character mapping
        self._vocab = self._build_vocab()
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        
        super().__init__(
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs
        )
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary mapping characters to IDs"""
        vocab = {}
        
        # Add special tokens
        vocab["<unk>"] = 2
        vocab["<pad>"] = 1
        vocab["<eos>"] = 0
        
        # Add printable ASCII characters (32-126)
        for i in range(32, min(127, self._vocab_size)):
            vocab[chr(i)] = i
        
        # Add extended byte values as special tokens
        for i in range(127, self._vocab_size):
            vocab[f"<byte_{i}>"] = i
            
        return vocab
    
    def clamp(self, n: int) -> int:
        """Clamp token ID to valid range, matching the CENO tokenizer implementation"""
        return max(0, min(n, self._vocab_size - 1))
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size"""
        return self._vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary dictionary"""
        return self._vocab.copy()
    
    def _tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using numpy's fromstring (byte-level tokenization).
        Byte-level tokenization: text is converted directly to its ASCII byte IDs.
        """
        # Convert text to byte array using numpy (matches the CENO implementation)
        token_ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).tolist()
        return token_ids
    
    def _convert_token_to_id(self, token: Union[str, int]) -> int:
        """Convert token to ID"""
        if isinstance(token, int):
            return self.clamp(token)
        
        # Handle string tokens
        if token in self._vocab:
            return self._vocab[token]
        
        # Handle single characters
        if len(token) == 1:
            return self.clamp(ord(token))
        
        # Handle byte tokens
        if token.startswith("<byte_") and token.endswith(">"):
            try:
                byte_val = int(token[6:-1])
                return self.clamp(byte_val)
            except ValueError:
                pass
        
        # Return unknown token ID
        return self._vocab.get(self.unk_token, 0)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token, CENO decode-token behavior"""
        clamped_index = self.clamp(index)
        
        # Handle special cases before interpreting byte values.
        if clamped_index == self.eos_id:
            return self.eos_token
        if clamped_index == self.pad_id:
            return self.pad_token
        if clamped_index == self.unk_id:
            return self.unk_token
        
        # Convert to character if in printable range
        if 32 <= clamped_index <= 126:
            return chr(clamped_index)
        
        # Return byte token for extended range
        return f"<byte_{clamped_index}>"
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens back to string"""
        result = []
        for token in tokens:
            if token in [self.pad_token, self.eos_token, self.unk_token]:
                continue
            elif token.startswith("<byte_") and token.endswith(">"):
                try:
                    byte_val = int(token[6:-1])
                    result.append(chr(self.clamp(byte_val)))
                except (ValueError, OverflowError):
                    continue
            else:
                result.append(token)
        return "".join(result)
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text and return string tokens.
        This wraps the numeric tokenization for HuggingFace compatibility.
        """
        # Get numeric tokens
        numeric_tokens = self._tokenize(text)
        
        # Convert to string tokens
        string_tokens = [self._convert_id_to_token(token_id) for token_id in numeric_tokens]
        
        return string_tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs.
        Core tokenization functionality of the CENO byte-level tokenizer.
        """
        # Tokenize to get numeric IDs directly
        token_ids = self._tokenize(text)
        
        # Handle truncation
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        
        # Handle padding
        if padding and max_length is not None:
            if len(token_ids) < max_length:
                token_ids.extend([self.pad_id] * (max_length - len(token_ids)))
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            return torch.tensor([token_ids], dtype=torch.long)
        elif return_tensors == "np":
            return np.array([token_ids], dtype=np.int64)
        
        return token_ids
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor, np.ndarray],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        """
        Decode token IDs back to text.
        CENO detokenization.
        """
        # Convert to list if tensor or numpy array
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Convert IDs to tokens
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
        
        # Filter special tokens if requested
        if skip_special_tokens:
            tokens = [
                token for token in tokens 
                if token not in [self.pad_token, self.eos_token, self.unk_token]
            ]
        
        # Convert tokens to string
        return self.convert_tokens_to_string(tokens)
    
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[List[str], List[Tuple[str, str]]],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> BatchEncoding:
        """Batch encode multiple texts"""
        batch_outputs = []
        
        for text in batch_text_or_text_pairs:
            if isinstance(text, tuple):
                # Handle text pairs (not typically used for DNA sequences)
                text = text[0]  # Just use first text for now
            
            encoded = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                padding=False,  # We'll handle padding after
                truncation=truncation,
                max_length=max_length,
                return_tensors=None,
            )
            batch_outputs.append(encoded)
        
        # Handle batch padding
        if padding and max_length is not None:
            max_len = max_length
        elif padding:
            max_len = max(len(output) for output in batch_outputs)
        else:
            max_len = None
        
        if max_len is not None:
            for i, output in enumerate(batch_outputs):
                if len(output) < max_len:
                    batch_outputs[i] = output + [self.pad_id] * (max_len - len(output))
                elif len(output) > max_len:
                    batch_outputs[i] = output[:max_len]
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            batch_outputs = torch.tensor(batch_outputs, dtype=torch.long)
        elif return_tensors == "np":
            batch_outputs = np.array(batch_outputs)
        
        return BatchEncoding({"input_ids": batch_outputs})
    
    def batch_decode(
        self,
        sequences: Union[List[List[int]], torch.Tensor, np.ndarray],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> List[str]:
        """Batch decode multiple sequences"""
        # Convert to list format
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        elif isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()
        
        return [
            self.decode(
                sequence,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs
            )
            for sequence in sequences
        ]
    
    def save_pretrained(
        self, 
        save_directory: str, 
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs
    ) -> Tuple[str]:
        """
        Save the tokenizer to a directory.
        
        Args:
            save_directory (str): Directory to save the tokenizer
            legacy_format (bool, optional): Whether to save in legacy format
            filename_prefix (str, optional): Prefix for filenames
            push_to_hub (bool): Whether to push to HuggingFace Hub
            **kwargs: Additional arguments
        
        Returns:
            Tuple[str]: Tuple of saved file paths
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        # Save vocabulary
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        
        # Save tokenizer configuration
        config_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "tokenizer_config.json"
        )
        
        tokenizer_config = {
            "tokenizer_class": "CENOCharLevelTokenizer",
            "vocab_size": self._vocab_size,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "eod_id": self.eod_id,
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "model_max_length": getattr(self, 'model_max_length', 1000000),
            "clean_up_tokenization_spaces": True,
            "tokenize_chinese_chars": False,
            "strip_accents": None,
            "do_lower_case": False,
            "do_basic_tokenize": False,
            "never_split": None,
            "tokenizer_type": "CharLevelTokenizer",
            "name_or_path": save_directory,
        }
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        
        # Save special tokens map
        special_tokens_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "special_tokens_map.json"
        )
        
        special_tokens_map = {
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }
        
        with open(special_tokens_file, "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Tokenizer saved to {save_directory}")
        
        return (vocab_file, config_file, special_tokens_file)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str] = None,
        revision: str = "main",
        **kwargs
    ):
        """
        Load a tokenizer from a pretrained model.
        
        Args:
            pretrained_model_name_or_path (str): Path to directory containing tokenizer files
                or name of a model on HuggingFace Hub
            cache_dir (str, optional): Directory to cache downloaded files
            force_download (bool): Whether to force download even if cached
            local_files_only (bool): Whether to only use local files
            token (str, optional): HuggingFace access token
            revision (str): Model revision to use
            **kwargs: Additional arguments
        
        Returns:
            CENOCharLevelTokenizer: Loaded tokenizer instance
        """
        # Handle local directory
        if os.path.isdir(pretrained_model_name_or_path):
            model_path = pretrained_model_name_or_path
        else:
            # Try to download from HuggingFace Hub
            try:
                from huggingface_hub import snapshot_download
                
                model_path = snapshot_download(
                    repo_id=pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                )
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to download models from the Hub. "
                    "Install it with: pip install huggingface_hub"
                )
            except Exception as e:
                logger.warning(f"Failed to download from HuggingFace Hub: {e}")
                logger.warning("Falling back to local initialization...")
                return cls(**kwargs)
        
        # Load tokenizer configuration
        config_file = os.path.join(model_path, "tokenizer_config.json")
        config = {}
        
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"Loaded tokenizer config from {config_file}")
        
        # Load special tokens map
        special_tokens_file = os.path.join(model_path, "special_tokens_map.json")
        special_tokens = {}
        
        if os.path.exists(special_tokens_file):
            with open(special_tokens_file, "r", encoding="utf-8") as f:
                special_tokens = json.load(f)
            logger.info(f"Loaded special tokens from {special_tokens_file}")
        
        # Load vocabulary
        vocab_file = os.path.join(model_path, VOCAB_FILES_NAMES["vocab_file"])
        vocab = None
        
        if os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            logger.info(f"Loaded vocabulary from {vocab_file}")
        
        # Merge configurations (kwargs override file config)
        init_kwargs = {
            "vocab_size": config.get("vocab_size", 512),
            "eos_token": special_tokens.get("eos_token", config.get("eos_token", "<eos>")),
            "pad_token": special_tokens.get("pad_token", config.get("pad_token", "<pad>")),
            "unk_token": special_tokens.get("unk_token", config.get("unk_token", "<unk>")),
        }
        
        # Override with any provided kwargs
        init_kwargs.update(kwargs)
        
        # Create tokenizer instance
        tokenizer = cls(**init_kwargs)
        
        # Load custom vocabulary if available
        if vocab is not None:
            tokenizer._vocab = vocab
            tokenizer._id_to_token = {v: k for k, v in vocab.items()}
            logger.info("Loaded custom vocabulary")
        
        # Set additional attributes from config
        if config:
            tokenizer.eod_id = config.get("eod_id", 0)
            tokenizer.eos_id = config.get("eos_id", 0)
            tokenizer.pad_id = config.get("pad_id", 1)
            if hasattr(tokenizer, 'model_max_length'):
                tokenizer.model_max_length = config.get("model_max_length", 1000000)
        
        tokenizer.name_or_path = pretrained_model_name_or_path
        logger.info(f"Successfully loaded tokenizer from {pretrained_model_name_or_path}")
        
        return tokenizer
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save vocabulary to file (legacy method)"""
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        vocab_file = os.path.join(
            save_directory, 
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        
        return (vocab_file,)
    
    @property
    def unique_identifiers(self) -> Dict[str,Any]:
        """
        Megatron will call .unique_identifiers when it encounters
        this object during its JSON‐dump of the dataset config.
        Must be JSON-serializable.
        """
        return {
            "tokenizer_class": self.__class__.__name__,
            "name_or_path": getattr(self, "name_or_path", None),
            "vocab_size": self.vocab_size,
        }
    # Compatibility methods for the CENO tokenizer interface
    def tokenize_batch(self, text_batch: Union[List[str], str]) -> Union[List[List[int]], List[int]]:
        """Batch tokenization matching the CENO tokenizer interface"""
        if isinstance(text_batch, str):
            return self._tokenize(text_batch)
        return [self._tokenize(text) for text in text_batch]
    
    def detokenize(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """Alias for decode method matching the CENO tokenizer interface"""
        return self.decode(token_ids, skip_special_tokens=True)
    
    def detokenize_batch(self, token_ids_batch: Union[List[List[int]], torch.Tensor]) -> List[str]:
        """Batch detokenization matching the CENO tokenizer interface"""
        return self.batch_decode(token_ids_batch, skip_special_tokens=True)
    
    @property
    def eod(self) -> int:
        """End of document token ID"""
        return self.eod_id
    
    @property
    def eos(self) -> int:
        """End of sequence token ID"""
        return self.eos_id 
