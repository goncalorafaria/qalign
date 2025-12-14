from transformers import AutoTokenizer
import anthropic
import asyncio
import os
from qalign.utils.list import unflatten_list

## get env variable DEBUG
DEBUG = os.getenv("DEBUG", "False").lower() == "true"


class RemoteAnthropic:
    def __init__(
        self,
        api_key: str = None,
        model_path: str = "claude-3-5-haiku-latest",
        max_new_tokens: int = 1024,
        max_prompt_length: int = 1024*3,
        stop_tokens: list = None,
        temperature: float = 1.0,
        timeout: float = 300,
        max_retries: int = 15,
        max_concurrent_requests: int = 256,
        tokenizer_path: str = None,  # Optional tokenizer for tokenization
    ):
        """
        Initialize RemoteAnthropic client.
        
        Args:
            api_key: Anthropic API key. If None, will try to get from ANTHROPIC_API_KEY env var.
            model_path: Anthropic model name (e.g., "claude-3-5-haiku-latest")
            max_new_tokens: Maximum tokens to generate
            max_prompt_length: Maximum prompt length in tokens
            stop_tokens: List of stop tokens/sequences
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            max_concurrent_requests: Maximum concurrent requests
            tokenizer_path: Path to tokenizer for tokenization. If None, will try to infer from model_path.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided or set as ANTHROPIC_API_KEY environment variable.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_prompt_length = max_prompt_length - 1
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests
        
        
        # Initialize tokenizer - use Qwen3 tokenizer as default
        # Anthropic models use claude tokenizer, but we use Qwen for tokenization
        if tokenizer_path is None:
            tokenizer_path = "Qwen/Qwen3-30B-A3B-Thinking-2507"
        self.model_path = tokenizer_path
        self.anthropic_model_path = model_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="left",
        )
        
        if stop_tokens is None:
            self.stop_tokens = [self.tokenizer.eos_token] if self.tokenizer.eos_token else []
        else:
            self.stop_tokens = stop_tokens + ([self.tokenizer.eos_token] if self.tokenizer.eos_token else [])
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else 0
            self.tokenizer.pad_token = self.tokenizer.bos_token if self.tokenizer.bos_token else "<pad>"
        
        # Check health by making a simple test request
        self._check_health()

    def encode(self, prompt_txt):
        tokens = self.tokenize(prompt_txt)
        return tokens

    def tokenize(self, prompt, **tokenizer_kwargs):
        return [
            self.tokenizer.encode(
                p,
                max_length=self.max_prompt_length+self.max_new_tokens,
                truncation=True,
                add_special_tokens=False,
                **tokenizer_kwargs
            )
            for p in prompt
        ]
    
    def decode_tokenize(self, ids): 
        return self.tokenizer.batch_decode(ids, skip_special_tokens=False, spaces_between_special_tokens=False)

    def _truncate_tokens(self, tokenized_input):
        """
        Truncate tokenized input if it exceeds max_prompt_length.
        Keeps the end of the sequence (most recent tokens).
        
        Args:
            tokenized_input: List of token IDs
            
        Returns:
            Truncated list of token IDs
        """
        if len(tokenized_input) <= self.max_prompt_length:
            return tokenized_input
        
        # Truncate from the beginning, keeping the end
        return tokenized_input[-self.max_prompt_length:]

    def _check_health(self):
        """Check if Anthropic API is accessible by making a simple test request."""
        try:
            # Make a minimal test request
            response = self.client.messages.create(
                model=self.anthropic_model_path,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            if DEBUG:
                print(f"Anthropic API[{self.anthropic_model_path}] is healthy and ready for requests.")
            return True
        except Exception as e:
            raise ConnectionError(f"Anthropic API health check failed: {str(e)}")

    def _text_to_messages(self, text):
        """
        Convert text prompt to Anthropic messages format.
        Assumes the text is a user message.
        
        Args:
            text: Text prompt string
            
        Returns:
            List of message dicts in Anthropic format
        """
        return [{"role": "user", "content": text}]

    async def _create_message_async(self, messages, prefix=None, use_tqdm=False):
        """
        Create a message using Anthropic API with retry logic.
        
        Args:
            messages: List of messages in Anthropic format
            prefix: Optional prefix text to include as assistant message (for prefill)
            use_tqdm: Whether to show progress (not used for single requests)
            
        Returns:
            Response text
        """
        # Add prefix as assistant message if provided
        if prefix:
            messages = messages + [{"role": "assistant", "content": prefix.rstrip()}]
        
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.anthropic_model_path,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    messages=messages,
                    stop_sequences=self.stop_tokens if self.stop_tokens else None,
                )
                return response.content[0].text.rstrip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Request failed after {self.max_retries} attempts: {e}")
                await asyncio.sleep(1)
        
        raise RuntimeError("Unexpected error in _create_message_async")

    async def _create_messages_async(self, messages_list, prefixes=None, use_tqdm=False):
        """
        Create multiple messages concurrently with rate limiting.
        
        Args:
            messages_list: List of message lists (one per request)
            prefixes: Optional list of prefix strings (one per request)
            use_tqdm: Whether to show progress bar
            
        Returns:
            List of response texts
        """
        if prefixes is None:
            prefixes = [None] * len(messages_list)
        
        # Limit concurrent requests using a semaphore
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def limited_request(messages, prefix):
            async with semaphore:
                return await self._create_message_async(messages, prefix, use_tqdm=False)
        
        tasks = [limited_request(msgs, prefix) for msgs, prefix in zip(messages_list, prefixes)]
        
        if use_tqdm:
            from tqdm import tqdm
            
            # Create a wrapper to update progress bar
            completed_count = [0]
            pbar = tqdm(total=len(tasks), desc="Processing")
            
            async def tracked_task(task):
                result = await task
                completed_count[0] += 1
                pbar.update(1)
                return result
            
            results = await asyncio.gather(*[tracked_task(task) for task in tasks])
            pbar.close()
        else:
            results = await asyncio.gather(*tasks)
        
        return results

    def continuation(self, prompt, prefix=None, use_tqdm=False):
        """
        Generate continuations for tokenized prompts.
        
        Args:
            prompt: List of tokenized prompts (list of token IDs)
            prefix: Optional list of tokenized prefixes (list of token IDs)
            use_tqdm: Whether to show progress bar
            
        Returns:
            List of tokenized completions (list of token IDs)
        """
        # Truncate prompts to max_prompt_length
        prompt = [self._truncate_tokens(p) for p in prompt]
        
        # Decode tokenized prompts to text
        prompt_text = self.decode_tokenize(prompt)
        
        # Extract prefix text if provided
        if prefix is not None:
            # Truncate prefixes
            prefix = [self._truncate_tokens(p) for p in prefix]
            prefix_text_list = self.decode_tokenize(prefix)
        else:
            prefix_text_list = [None] * len(prompt_text)
        
        # Convert text prompts to messages format
        messages_list = [self._text_to_messages(p) for p in prompt_text]
        
        if DEBUG:
            print("model_packet:", {"messages": messages_list[0], "prefix": prefix_text_list[0]})
        
        # Make async requests
        results = asyncio.run(self._create_messages_async(messages_list, prefix_text_list, use_tqdm=use_tqdm))
        
        # Tokenize the completions
        completion_ids = [xi for xi in self.tokenize(results)]
        
        return completion_ids

    def ancestral(
        self,
        input_data,
        n: int = 1,
        use_tqdm=False,
    ):
        """
        Generate multiple ancestral samples for each input.
        
        Args:
            input_data: List of text prompts
            n: Number of samples per prompt
            use_tqdm: Whether to show progress bar
            
        Returns:
            Nested list of text completions (unflattened)
        """
        # Truncate input_data (text) by tokenizing, truncating, then decoding
        tokenized_data = []
        for prompt in input_data:
            tokens = self.tokenizer.encode(prompt, max_length=self.max_prompt_length, truncation=True)
            truncated_prompt = self.tokenizer.decode(tokens, skip_special_tokens=False)
            tokenized_data.append(truncated_prompt)
        
        prompts = []
        for prompt in tokenized_data: 
            prompts.extend([prompt] * n)
        
        # Convert to messages format
        messages_list = [self._text_to_messages(p) for p in prompts]
        
        # Make async requests
        results = asyncio.run(self._create_messages_async(messages_list, use_tqdm=use_tqdm))
        
        return unflatten_list(results, [n] * len(input_data))

    def __str__(self):
        return f"RemoteAnthropic(model_path={self.anthropic_model_path}, api_key={'***' if self.api_key else None})"

