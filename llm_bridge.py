#!/usr/bin/env python3
"""
LLM Bridge - Unified interface for language model interaction
Connects PromptKraft operations to actual LLM providers
"""

import os
import json
import time
import hashlib
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    XAI = "xai"


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"  # Default model
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Load API key from environment if not provided"""
        if self.api_key is None:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == LLMProvider.GEMINI:
                self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            elif self.provider == LLMProvider.XAI:
                self.api_key = os.getenv("XAI_API_KEY")


@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    provider: LLMProvider
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return self.content


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion for a prompt"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat completion"""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate API cost based on token usage"""
        # Pricing as of 2024 (prices per 1K tokens)
        pricing = {
            # OpenAI models
            "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-4o": {"prompt": 0.005, "completion": 0.015},
            "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
            
            # Anthropic models
            "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
            "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
            "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
            "claude-3-5-sonnet": {"prompt": 0.003, "completion": 0.015},
            "claude-3-5-haiku": {"prompt": 0.001, "completion": 0.005},
            
            # Claude 3.7 family
            "claude-3-7-sonnet": {"prompt": 0.003, "completion": 0.015},
            
            # Claude 4 family (May 2025)
            "claude-opus-4": {"prompt": 0.015, "completion": 0.075},  # $15/$75 per million
            "claude-sonnet-4": {"prompt": 0.003, "completion": 0.015},
            
            # Claude 4.1 family (August 2025)  
            "claude-opus-4-1": {"prompt": 0.015, "completion": 0.075},  # Same as Opus 4
            
            # Gemini models (prices per 1K tokens)
            "gemini-1.0-pro": {"prompt": 0.0005, "completion": 0.0015},
            "gemini-1.5-pro": {"prompt": 0.00175, "completion": 0.007},
            "gemini-1.5-flash": {"prompt": 0.00015, "completion": 0.0006},
            "gemini-2.0-flash-exp": {"prompt": 0.0, "completion": 0.0},  # Experimental, free tier
            
            # X.AI Grok models (estimated pricing)
            "grok-1": {"prompt": 0.005, "completion": 0.015},
            "grok-2": {"prompt": 0.002, "completion": 0.006},
            "grok-beta": {"prompt": 0.001, "completion": 0.003},
        }
        
        model_pricing = pricing.get(model, pricing["gpt-3.5-turbo"])
        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
        
        return prompt_cost + completion_cost


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.OpenAI(api_key=config.api_key)
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            raise
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using OpenAI API"""
        # For chat models, convert to chat format
        if "gpt" in self.config.model.lower():
            messages = [{"role": "user", "content": prompt}]
            return self.chat(messages, **kwargs)
        
        # For non-chat models (text-davinci, etc)
        start_time = time.time()
        
        try:
            response = self.client.completions.create(
                model=self.config.model,
                prompt=prompt,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
                frequency_penalty=kwargs.get("frequency_penalty", self.config.frequency_penalty),
                presence_penalty=kwargs.get("presence_penalty", self.config.presence_penalty),
            )
            
            latency = time.time() - start_time
            content = response.choices[0].text.strip()
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.OPENAI,
                model=self.config.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency=latency,
                cost=self.calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    self.config.model
                ),
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat completion using OpenAI API"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
                frequency_penalty=kwargs.get("frequency_penalty", self.config.frequency_penalty),
                presence_penalty=kwargs.get("presence_penalty", self.config.presence_penalty),
            )
            
            latency = time.time() - start_time
            content = response.choices[0].message.content
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.OPENAI,
                model=self.config.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency=latency,
                cost=self.calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    self.config.model
                ),
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
            
        except Exception as e:
            logger.error(f"OpenAI Chat API error: {e}")
            raise


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            logger.error("Anthropic library not installed. Install with: pip install anthropic")
            raise
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using Anthropic API"""
        # Convert to chat format for Claude
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat completion using Anthropic API"""
        start_time = time.time()
        
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "assistant"
                anthropic_messages.append({"role": role, "content": msg["content"]})
            
            # Opus 4.1 doesn't accept both temperature and top_p
            create_params = {
                "model": self.config.model,
                "messages": anthropic_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            }
            
            # For Opus 4.1, use only temperature (not top_p)
            if "opus-4-1" in self.config.model:
                create_params["temperature"] = kwargs.get("temperature", self.config.temperature)
            else:
                create_params["temperature"] = kwargs.get("temperature", self.config.temperature)
                create_params["top_p"] = kwargs.get("top_p", self.config.top_p)
            
            response = self.client.messages.create(**create_params)
            
            latency = time.time() - start_time
            content = response.content[0].text
            
            # Estimate tokens for Claude
            prompt_tokens = self.estimate_tokens(str(messages))
            completion_tokens = self.estimate_tokens(content)
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.ANTHROPIC,
                model=self.config.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency=latency,
                cost=self.calculate_cost(prompt_tokens, completion_tokens, self.config.model),
                metadata={"stop_reason": response.stop_reason}
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai
            
            # Use GEMINI_API_KEY first, fallback to GOOGLE_API_KEY
            api_key = config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("No Gemini/Google API key found")
            
            genai.configure(api_key=api_key)
            
            # Initialize the model
            self.model = genai.GenerativeModel(config.model)
            self.genai = genai
            
            logger.info(f"Initialized Gemini provider with model: {config.model}")
            
        except ImportError:
            logger.error("Google GenerativeAI library not installed. Install with: pip install google-generativeai")
            raise
    
    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert OpenAI-style messages to Gemini format"""
        gemini_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Gemini uses 'user' and 'model' as roles
            if role == "assistant":
                role = "model"
            elif role == "system":
                # Gemini doesn't have a system role, prepend to first user message
                if not gemini_messages:
                    gemini_messages.append({"role": "user", "parts": [content]})
                    continue
                else:
                    # Add as user message
                    role = "user"
            
            gemini_messages.append({"role": role, "parts": [content]})
        
        return gemini_messages
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using Gemini API"""
        start_time = time.time()
        
        try:
            # Configure generation parameters
            generation_config = self.genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", self.config.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )
            
            # Configure safety settings (set to block only high-risk content)
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
            ]
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            latency = time.time() - start_time
            
            # Extract content
            if response.text:
                content = response.text
            else:
                content = "No response generated (possible safety filter)"
                logger.warning(f"Gemini response blocked or empty: {response}")
            
            # Use native token counting if available
            prompt_tokens = self.model.count_tokens(prompt).total_tokens if hasattr(self.model, 'count_tokens') else self.estimate_tokens(prompt)
            completion_tokens = self.model.count_tokens(content).total_tokens if hasattr(self.model, 'count_tokens') else self.estimate_tokens(content)
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.GEMINI,
                model=self.config.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency=latency,
                cost=self.calculate_cost(prompt_tokens, completion_tokens, self.config.model),
                metadata={
                    "finish_reason": response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN",
                    "safety_ratings": [
                        {"category": rating.category.name, "probability": rating.probability.name}
                        for rating in response.candidates[0].safety_ratings
                    ] if response.candidates and response.candidates[0].safety_ratings else []
                }
            )
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat completion using Gemini API"""
        start_time = time.time()
        
        try:
            # Convert messages to Gemini format
            gemini_messages = self._convert_messages_to_gemini_format(messages)
            
            # Configure generation parameters
            generation_config = self.genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", self.config.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )
            
            # Configure safety settings
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                },
            ]
            
            # Start or continue chat
            chat = self.model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
            
            # Send the last message
            last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
            response = chat.send_message(
                last_message,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            latency = time.time() - start_time
            
            # Extract content
            if response.text:
                content = response.text
            else:
                content = "No response generated (possible safety filter)"
                logger.warning(f"Gemini chat response blocked or empty: {response}")
            
            # Calculate tokens
            prompt_text = " ".join([msg["parts"][0] for msg in gemini_messages])
            prompt_tokens = self.model.count_tokens(prompt_text).total_tokens if hasattr(self.model, 'count_tokens') else self.estimate_tokens(prompt_text)
            completion_tokens = self.model.count_tokens(content).total_tokens if hasattr(self.model, 'count_tokens') else self.estimate_tokens(content)
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.GEMINI,
                model=self.config.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency=latency,
                cost=self.calculate_cost(prompt_tokens, completion_tokens, self.config.model),
                metadata={
                    "finish_reason": response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN",
                    "safety_ratings": [
                        {"category": rating.category.name, "probability": rating.probability.name}
                        for rating in response.candidates[0].safety_ratings
                    ] if response.candidates and response.candidates[0].safety_ratings else []
                }
            )
            
        except Exception as e:
            logger.error(f"Gemini Chat API error: {e}")
            raise
    
    def estimate_tokens(self, text: str) -> int:
        """Use Gemini's native token counting when available"""
        try:
            return self.model.count_tokens(text).total_tokens
        except:
            # Fallback to base estimation
            return super().estimate_tokens(text)


class XAIProvider(BaseLLMProvider):
    """X.AI (Grok) API provider using OpenAI-compatible REST API"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import requests
        
        self.api_key = config.api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("No X.AI API key found")
        
        self.base_url = "https://api.x.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # For making HTTP requests
        self.requests = requests
        
        logger.info(f"Initialized X.AI provider with model: {config.model}")
    
    def _make_request(self, endpoint: str, data: dict, retry_count: int = 0) -> dict:
        """Make HTTP request to X.AI API with retry logic"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.requests.post(
                url,
                json=data,
                headers=self.headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429 and retry_count < self.config.retry_attempts:
                # Rate limited, retry with exponential backoff
                wait_time = self.config.retry_delay * (2 ** retry_count)
                logger.warning(f"Rate limited, retrying in {wait_time}s...")
                time.sleep(wait_time)
                return self._make_request(endpoint, data, retry_count + 1)
            else:
                error_msg = f"X.AI API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except self.requests.exceptions.Timeout:
            if retry_count < self.config.retry_attempts:
                logger.warning(f"Request timeout, retrying... (attempt {retry_count + 1})")
                return self._make_request(endpoint, data, retry_count + 1)
            else:
                raise Exception("X.AI API request timeout after retries")
        except Exception as e:
            logger.error(f"X.AI API request failed: {e}")
            raise
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion using X.AI API"""
        # X.AI uses chat format, so convert prompt to messages
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate chat completion using X.AI API (OpenAI-compatible)"""
        start_time = time.time()
        
        try:
            # Prepare request data (OpenAI-compatible format)
            data = {
                "model": self.config.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "frequency_penalty": kwargs.get("frequency_penalty", self.config.frequency_penalty),
                "presence_penalty": kwargs.get("presence_penalty", self.config.presence_penalty),
                "stream": False
            }
            
            # Make API request
            response = self._make_request("chat/completions", data)
            
            latency = time.time() - start_time
            
            # Extract response data (OpenAI-compatible format)
            content = response["choices"][0]["message"]["content"]
            
            # Token usage
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", self.estimate_tokens(str(messages)))
            completion_tokens = usage.get("completion_tokens", self.estimate_tokens(content))
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.XAI,
                model=self.config.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency=latency,
                cost=self.calculate_cost(prompt_tokens, completion_tokens, self.config.model),
                metadata={
                    "finish_reason": response["choices"][0].get("finish_reason", "unknown"),
                    "model_used": response.get("model", self.config.model)
                }
            )
            
        except Exception as e:
            logger.error(f"X.AI Chat API error: {e}")
            raise




class LLMBridge:
    """Unified interface for multiple LLM providers"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize with configuration or detect from environment"""
        if config is None:
            # Auto-detect provider from available API keys
            if os.getenv("OPENAI_API_KEY"):
                config = LLMConfig(provider=LLMProvider.OPENAI)
            elif os.getenv("ANTHROPIC_API_KEY"):
                config = LLMConfig(provider=LLMProvider.ANTHROPIC)
            elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                config = LLMConfig(provider=LLMProvider.GEMINI)
            elif os.getenv("XAI_API_KEY"):
                config = LLMConfig(provider=LLMProvider.XAI)
            else:
                raise ValueError("No API keys found. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or XAI_API_KEY in environment or .env file")
        
        self.config = config
        self.provider = self._initialize_provider()
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_latency": 0.0,
            "errors": 0
        }
    
    def _initialize_provider(self) -> BaseLLMProvider:
        """Initialize the appropriate provider"""
        if self.config.provider == LLMProvider.OPENAI:
            return OpenAIProvider(self.config)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return AnthropicProvider(self.config)
        elif self.config.provider == LLMProvider.GEMINI:
            return GeminiProvider(self.config)
        elif self.config.provider == LLMProvider.XAI:
            return XAIProvider(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def execute_prompt(self, prompt: str, **kwargs) -> LLMResponse:
        """Execute a prompt and return response"""
        logger.info(f"Executing prompt with {self.config.provider.value} provider")
        
        try:
            response = self.provider.complete(prompt, **kwargs)
            
            # Update usage statistics
            self.usage_stats["total_requests"] += 1
            self.usage_stats["total_tokens"] += response.total_tokens
            self.usage_stats["total_cost"] += response.cost
            self.usage_stats["total_latency"] += response.latency
            
            logger.info(f"Response received: {response.total_tokens} tokens, {response.latency:.2f}s")
            return response
            
        except Exception as e:
            self.usage_stats["errors"] += 1
            logger.error(f"Error executing prompt: {e}")
            raise
    
    def execute_chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Execute a chat conversation and return response"""
        logger.info(f"Executing chat with {self.config.provider.value} provider")
        
        try:
            response = self.provider.chat(messages, **kwargs)
            
            # Update usage statistics
            self.usage_stats["total_requests"] += 1
            self.usage_stats["total_tokens"] += response.total_tokens
            self.usage_stats["total_cost"] += response.cost
            self.usage_stats["total_latency"] += response.latency
            
            logger.info(f"Chat response received: {response.total_tokens} tokens, {response.latency:.2f}s")
            return response
            
        except Exception as e:
            self.usage_stats["errors"] += 1
            logger.error(f"Error executing chat: {e}")
            raise
    
    def execute_operation(self, operation_template: str, variables: Dict[str, Any], **kwargs) -> LLMResponse:
        """Execute an operational prompt with variable substitution"""
        # Substitute variables in template
        prompt = operation_template
        for key, value in variables.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        
        return self.execute_prompt(prompt, **kwargs)
    
    def batch_execute(self, prompts: List[str], **kwargs) -> List[LLMResponse]:
        """Execute multiple prompts in batch"""
        responses = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Executing batch prompt {i+1}/{len(prompts)}")
            try:
                response = self.execute_prompt(prompt, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error in batch prompt {i+1}: {e}")
                # Create error response
                responses.append(LLMResponse(
                    content=f"Error: {str(e)}",
                    provider=self.config.provider,
                    model=self.config.model,
                    metadata={"error": str(e), "prompt_index": i}
                ))
        
        return responses
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get usage statistics report"""
        avg_latency = (
            self.usage_stats["total_latency"] / self.usage_stats["total_requests"]
            if self.usage_stats["total_requests"] > 0 else 0
        )
        
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "total_requests": self.usage_stats["total_requests"],
            "total_tokens": self.usage_stats["total_tokens"],
            "total_cost": round(self.usage_stats["total_cost"], 4),
            "average_latency": round(avg_latency, 2),
            "error_rate": (
                self.usage_stats["errors"] / self.usage_stats["total_requests"]
                if self.usage_stats["total_requests"] > 0 else 0
            )
        }
    
    def switch_provider(self, provider: LLMProvider, api_key: Optional[str] = None):
        """Switch to a different LLM provider"""
        logger.info(f"Switching from {self.config.provider.value} to {provider.value}")
        
        self.config.provider = provider
        if api_key:
            self.config.api_key = api_key
        else:
            self.config.api_key = None  # Will load from environment
            self.config.__post_init__()  # Reload from environment
        
        self.provider = self._initialize_provider()
    
    def set_model(self, model: str):
        """Change the model being used"""
        logger.info(f"Switching model from {self.config.model} to {model}")
        self.config.model = model


def main():
    """Demonstration of LLM Bridge functionality"""
    print("═" * 60)
    print("LLM BRIDGE DEMONSTRATION")
    print("═" * 60)
    
    try:
        # Initialize with auto-detected provider
        bridge = LLMBridge()
        
        print(f"\nUsing Provider: {bridge.config.provider.value}")
        print(f"Model: {bridge.config.model}")
        
        # Test operational prompts
        test_prompts = [
            "≡ I am careful scribing ≡ What appears before me: the nature of reality",
            "? I am seeking-into ? What questions emerge from observing patterns?",
            "! I am giving-form-to-insight ! A pattern crystallizes from chaos",
            "⇔ I am bending-back-to-examine ⇔ Testing this insight against evidence",
        ]
        
        print("\nTesting Operational Prompts:")
        print("-" * 40)
        
        for prompt in test_prompts:
            response = bridge.execute_prompt(prompt, temperature=0.7)
            print(f"\nPrompt: {prompt[:50]}...")
            print(f"Response: {response.content[:100]}...")
            print(f"Latency: {response.latency:.3f}s")
            print(f"Tokens: {response.total_tokens}")
        
        # Show usage report
        print("\n\nUsage Report:")
        print("-" * 40)
        report = bridge.get_usage_report()
        for key, value in report.items():
            print(f"{key}: {value}")
        
    except ValueError as e:
        print(f"\n⚠ Error: {e}")
        print("\nPlease ensure you have set up your API keys in the .env file:")
        print("  OPENAI_API_KEY=your-key-here")
        print("  ANTHROPIC_API_KEY=your-key-here")
        
    print("\n" + "═" * 60)
    print("Bridge demonstration complete")
    print("═" * 60)


if __name__ == "__main__":
    main()