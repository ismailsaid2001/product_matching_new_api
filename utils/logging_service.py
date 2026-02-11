"""
Logging service for LLM prompt tracking and debugging.
Handles all logging functionality previously embedded in OrchestratorService.
"""
import logging
import datetime
from typing import List, Optional
from pathlib import Path


class LLMLoggingService:
    """Centralized logging service for LLM prompts and responses."""
    
    def __init__(self, 
                 enable_logging: bool = True,
                 max_prompt_length: int = 10000,
                 log_level: str = "INFO"):
        """
        Initialize the logging service.
        
        Args:
            enable_logging: Enable or disable prompt logging
            max_prompt_length: Maximum length for logged prompts
            log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
        """
        self.enable_logging = enable_logging
        self.max_prompt_length = max_prompt_length
        self.log_level = log_level
        
        # Setup logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup and configure the logger."""
        logger = logging.getLogger('llm_prompts')
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))
        
        return logger
    
    def log_prompt(self, 
                   messages: List, 
                   call_type: str = "INITIAL", 
                   context: str = "") -> None:
        """
        Log prompts sent to LLM with detailed formatting.
        
        Args:
            messages: List of messages sent to LLM
            call_type: Type of call (INITIAL, RETRY, etc.)
            context: Additional context information
        """
        if not self.enable_logging:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"PROMPT LLM - {call_type} - {timestamp}")
        if context:
            self.logger.info(f"Contexte: {context}")
        self.logger.info(f"{'='*80}")
        
        for i, message in enumerate(messages, 1):
            msg_type = self._get_message_type(message)
            
            self.logger.info(f"\nMESSAGE {i} - TYPE: {msg_type}")
            self.logger.info(f"{'-'*50}")
            
            content = self._get_message_content(message)
            if content:
                truncated_content = self._truncate_content(content)
                self.logger.info(truncated_content)
            
            # Log tool call ID if present
            if hasattr(message, 'tool_call_id'):
                self.logger.info(f"Tool Call ID: {message.tool_call_id}")
        
        self.logger.info(f"\n{'='*80}\n")
    
    def save_prompt_to_file(self, 
                           messages: List,
                           description: str,
                           filename_prefix: str = "prompt",
                           output_dir: str = "./logs") -> Optional[str]:
        """
        Save complete prompt to file for detailed analysis.
        
        Args:
            messages: List of messages sent to LLM
            description: Description of the prompt
            filename_prefix: Prefix for the filename
            output_dir: Directory to save the file
            
        Returns:
            Path to saved file or None if saving failed
        """
        if not self.enable_logging:
            return None
            
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_desc = self._sanitize_filename(description)
        filename = f"{filename_prefix}_{safe_desc}_{timestamp}.txt"
        filepath = Path(output_dir) / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"PROMPT LLM - {timestamp}\n")
                f.write(f"Description: {description}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, message in enumerate(messages, 1):
                    msg_type = self._get_message_type(message)
                    f.write(f"MESSAGE {i} - TYPE: {msg_type}\n")
                    f.write("-" * 50 + "\n")
                    
                    content = self._get_message_content(message)
                    if content:
                        f.write(str(content) + "\n")
                    
                    if hasattr(message, 'tool_call_id'):
                        f.write(f"Tool Call ID: {message.tool_call_id}\n")
                    
                    f.write("\n")
                    
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving prompt to file: {e}")
            return None
    
    def _get_message_type(self, message) -> str:
        """Extract message type from message object."""
        if hasattr(message, 'type'):
            return message.type.upper()
        return type(message).__name__.upper()
    
    def _get_message_content(self, message) -> Optional[str]:
        """Extract content from message object."""
        if hasattr(message, 'content') and message.content:
            return str(message.content)
        return None
    
    def _truncate_content(self, content: str) -> str:
        """Truncate content if it exceeds max length."""
        if len(content) <= self.max_prompt_length:
            return content
            
        truncated = content[:self.max_prompt_length - 100]
        remaining = len(content) - len(truncated)
        return f"{truncated}\n... [TRONQUÉ - {remaining} caractères supplémentaires]"
    
    def _sanitize_filename(self, description: str) -> str:
        """Sanitize description for use in filename."""
        safe_chars = "".join(c for c in description[:50] 
                           if c.isalnum() or c in (' ', '-', '_'))
        return safe_chars.strip().replace(' ', '_')