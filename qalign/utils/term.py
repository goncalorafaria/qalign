from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.markup import escape
import time
import difflib

import asyncio
from rich.live import Live
from rich.text import Text
import shutil
console = Console()

def find_changes(old_text, new_text, change_color="bold red", base_color=" cyan"):
    """
    Find the differences between two texts and return styled Text object
    with changes highlighted in bold red
    """
    if not old_text:
        # First response - no changes to highlight
        return Text(new_text, style=base_color)
    
    # Use difflib to find differences
    old_words = old_text.split()
    new_words = new_text.split()
    
    # Get the diff operations
    opcodes = difflib.SequenceMatcher(None, old_words, new_words).get_opcodes()
    
    styled_text = Text()
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Unchanged text - normal formatting
            unchanged = ' '.join(new_words[j1:j2])
            if unchanged:
                styled_text.append(unchanged + ' ')
        elif tag == 'replace' or tag == 'insert':
            # Changed or new text - bold red
            changed = ' '.join(new_words[j1:j2])
            if changed:
                styled_text.append(changed + ' ', style=change_color)
        # Skip 'delete' operations as they're not in the new text
    
    return styled_text

def animate_llm_responses(responses, prompt, delay=1.5,base_color="cyan",change_color="bold red", prompt_color="white"):
    """
    Animate a sequence of LLM responses with changes highlighted
    """
    previous_text = ""
    
    with Live(console=console, refresh_per_second=10) as live:
        for i, response in enumerate(responses):
            # Create header
            header = Text(f"Prompt:{prompt} \n(Step {i+1}/{len(responses)})\n\n", style=prompt_color)
            
            # Find and highlight changes 
            styled_response = find_changes(previous_text, response, change_color=change_color, base_color=base_color)
            
            # Combine header and response
            display_text = header + styled_response
            
            live.update(display_text)
            previous_text = response
            
            if i < len(responses) - 1:  # Don't sleep after last response
                time.sleep(delay)


class AsyncAnimateLLMResponsesCallback:
    """
    Async callback class for animating LLM responses.
    Completely non-blocking and async-friendly.
    """
    
    def __init__(self, prompt, delay=0, total_steps=None, base_color="cyan", change_color="bold red", prompt_color="cyan"):
        self.prompt = prompt
        self.delay = delay
        self.base_color = base_color
        self.change_color = change_color
        self.prompt_color = prompt_color
        self.previous_text = ""
        self.step_count = -1
        self.total_steps = total_steps
        self.live = None
        self.console = console
        self._task = None
        self.extra_text = None
        self._update_separators()
    
    def __enter__(self):
        """Start the live display"""
        self.live = Live(console=self.console, refresh_per_second=10)
        self.live.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the live display"""

        if self.live:
            self.live.stop()
            #self.console.clear()
    
    def _update_separators(self):
        """Update separator lengths based on current terminal size"""
        try:
            terminal_width = shutil.get_terminal_size().columns
        except:
            terminal_width = 80
        self.separator = "=" * terminal_width
        self.dash_separator = "-" * terminal_width
        
    def __call__(self, state):
        """Add a new response with optional async delay"""
        self.add_response(state.text[0])
    
    def add_extra_text(self, extra_text, style="bold blue"):
        """Add extra text to the response"""
        self.extra_text = Text(extra_text, style=style)
        
        
        
        
    def add_response(self, response_text):
        """Add a new response with optional async delay"""
        self.step_count += 1
        # Create header
        
        if self.total_steps is not None:
            
            progress= f"(Step {self.step_count}/{self.total_steps})"
            
            # Calculate progress bar
            progress_ratio = self.step_count / self.total_steps
            filled_length = int(progress_ratio * len(self.dash_separator))
            unfilled_length = len(self.dash_separator) - filled_length
            
            # Create progress bar: filled portion with "=", unfilled with "-"
            
            if filled_length > 0:
                progress_bar = Text("=" * (filled_length-1) + "|",style=self.change_color) + Text("-" * unfilled_length+"\n", style=self.prompt_color)
            else:
                progress_bar = Text(self.dash_separator+"\n", style=self.prompt_color)
            
        else:
            progress= f"(Step {self.step_count})"
            progress_bar = self.dash_separator
        
        header = Text(f"{self.separator}\nPrompt: {self.prompt}\n{progress}\n", style=self.prompt_color)+progress_bar
            
        if response_text != self.previous_text:
            
            # Find and highlight changes
            styled_response = find_changes(self.previous_text, response_text, 
                                        change_color=self.change_color, base_color=self.base_color)
            
            self.cached_text = styled_response + Text(f"\n{self.separator}\n", style=self.prompt_color)
            # Combine header and response
            display_text = header + self.cached_text
            if self.extra_text:
                display_text += self.extra_text
            
            # Update the display immediately
            if self.live:
                self.live.update(display_text)
            
            # Store current text for next comparison
            self.previous_text = response_text
            
        else:
            display_text = header + self.cached_text
            if self.extra_text:
                display_text += self.extra_text
            
            # Update the display immediately
            if self.live:
                self.live.update(display_text)
            
        
    
    def update_prompt(self, new_prompt):
        """Update the prompt text"""
        self.prompt = new_prompt
    
    def reset(self):
        """Reset the callback state"""
        self.previous_text = ""
        self.step_count = 0
        if self._task:
            self._task.cancel()
            
# Alternative approach: Manual markup for specific changes
def create_manual_response(base_text, changes_dict, highlight_style="bold red"):
    """
    Create a response with manually specified changes highlighted
    
    Args:
        base_text: The text with placeholders like {variable}
        changes_dict: Dict of variable names to their values
        highlight_style: Rich style for highlighted text
    """
    styled_text = Text()
    
    # Simple template replacement with highlighting
    current_text = base_text
    
    for var, value in changes_dict.items():
        placeholder = f"{{{var}}}"
        if placeholder in current_text:
            parts = current_text.split(placeholder)
            current_text = parts[0]
            
            # Add the part before the variable (normal style)
            styled_text.append(parts[0])
            
            # Add the variable value (highlighted)
            styled_text.append(str(value), style=highlight_style)
            
            # Continue with the rest
            current_text = placeholder.join(parts[1:])
    
    # Add any remaining text
    styled_text.append(current_text)
    
    return styled_text


if __name__ == "__main__":
    # Demo 1: Automatic change detection
    print("Demo 1: Automatic change detection")
    
    sample_responses = ['To find the total number of apples, we need to multiply the initial number of apples by 2 for each 10-second interval, adding the new total each time. \n\nStarting with 10 apples at 0 seconds:\n- After 10 seconds: 10 x 2 = 20 apples\n- After 20 seconds: 20 x 2 = 40 apples\n- After 30 seconds: 40 x 2 = 80 apples\n- After 40 seconds: 80 x 2 = 160 apples\n- After 50 seconds: 160 x 2 = 320 apples\n\nSo, after 50 seconds, Joana will have 320 apples.<|eot_id|>', 'To find the total number of apples, we need to multiply the initial number of apples by 2 for each 10-second interval, adding the new total each time. \n\nStarting with 10 apples at 0 seconds:\n- After 10 seconds: 10 x 2 = 20 apples\n- After 20 seconds: 20 x 2 = 40 apples\n- After 30 seconds: 40 x 2 = 80 apples\n- After 40 seconds: 80 x 2 = 160 apples\n- After 50 seconds: 160 x 2 = 320 apples\n\nSo, after 50 seconds, Joana has 320 apples.<|eot_id|>', 'To find the total number of apples, we need to multiply the initial number of apples by 2 for each 10-second interval, adding the new total each time. \n\nStarting with 10 apples at 0 seconds:\n- After 10 seconds: 10 x 2 = 20 apples\n- After 20 seconds: 20 x 2 = 40 apples\n- After 30 seconds: 40 x 2 = 80 apples\n- After 40 seconds: 80 x 2 = 160 apples\n- After 50 seconds: 160 x 2 = 320 apples\n\nSo, Joana will have 320 apples after 50 seconds.<|eot_id|>', 'To find the total number of apples, we need to multiply the initial number of apples by 2 for each 10-second interval, adding the new total each time. \n\nStarting with 10 apples at 0 seconds:\n- After 10 seconds: 10 x 2 = 20 apples\n- After 20 seconds: 20 x 2 = 40 apples\n- After 30 seconds: 40 x 2 = 80 apples\n- After 40 seconds: 80 x 2 = 160 apples\n- After 50 seconds: 160 x 2 = 320 apples\n\nSo, after 50 seconds, Joana has 320 apples.<|eot_id|>', 'To find the total number of apples, we need to multiply the initial number of apples by 2 for each 10-second interval, adding the new total each time. \n\n- After 0 seconds (the starting point): Joana has 10 apples.\n- After 10 seconds: 10 * 2 = 20 apples.\n- After 20 seconds: 20 * 2 = 40 apples.\n- After 30 seconds: 40 * 2 = 80 apples.\n- After 40 seconds: 80 * 2 = 160 apples.\n- After 50 seconds: 160 * 2 = 320 apples.\n\nSo, after 50 seconds, Joana has 320 apples.<|eot_id|>', 'To find the total number of apples, we need to multiply the initial number of apples by 2 for each 10-second interval, adding the new total each time. \n\nStarting with 10 apples at 0 seconds:\n- After 10 seconds: 10 x 2 = 20 apples\n- After 20 seconds: 20 x 2 = 40 apples\n- After 30 seconds: 40 x 2 = 80 apples\n- After 40 seconds: 80 x 2 = 160 apples\n- After 50 seconds: 160 x 2 = 320 apples\n\nSo, after 50 seconds, Joana will have 320 apples.<|eot_id|>', 'To find the total number of apples, we need to multiply the initial number of apples by 2 for each 10-second interval, adding the new total each time. \n\nStarting with 10 apples at 0 seconds:\n- After 10 seconds: 10 x 2 = 20 apples\n- After 20 seconds: 20 x 2 = 40 apples\n- After 30 seconds: 40 x 2 = 80 apples\n- After 40 seconds: 80 x 2 = 160 apples\n- After 50 seconds: 160 x 2 = 320 apples\n\nJoana will have 320 apples after 50 seconds.<|eot_id|>', 'To find the total number of apples, we need to multiply the initial number of apples by 2 for each 10-second interval, adding the new total each time. \n\nStarting with 10 apples at 0 seconds:\n- After 10 seconds: 10 x 2 = 20 apples\n- After 20 seconds: 20 x 2 = 40 apples\n- After 30 seconds: 40 x 2 = 80 apples\n- After 40 seconds: 80 x 2 = 160 apples\n- After 50 seconds: 160 x 2 = 320 apples\n\nSo, after 50 seconds, Joana will have 320 apples.<|eot_id|>', 'To find the total number of apples, we need to multiply the initial number of apples by 2 for each 10-second interval, adding the new total each time. \n\nStarting with 10 apples at 0 seconds:\n- After 10 seconds: 10 x 2 = 20 apples\n- After 20 seconds: 20 x 2 = 40 apples\n- After 30 seconds: 40 x 2 = 80 apples\n- After 40 seconds: 80 x 2 = 160 apples\n- After 50 seconds: 160 x 2 = 320 apples\n\nTherefore, Joana has 320 apples after 50 seconds.<|eot_id|>']
    
    animate_llm_responses(sample_responses)
    
    time.sleep(2)
    #console.clear()
    
   # Demo 2: Template-based with manual highlighting
    #print("Demo 2: Template-based highlighting")
    #demo_with_placeholders()
    
    #console.print("\n\n✨ Demo complete!", style="bold green")