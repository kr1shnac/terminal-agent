from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.containers import Vertical

MODELS = [
    # Free models on the OpenRouter catalogue (all support tool calling).
    # `:free` variants bill $0.00, so a demo run on them costs nothing.
    ("Nemotron-3.5 Lightning (Free)", "nvidia/nemotron-3.5-lightning:free"),
    ("Nemotron-3 Ultra 550B (Free)", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    ("Nemotron-3 Super 120B (Free)", "nvidia/nemotron-3-super-120b-a12b:free"),
    ("Nemotron-3 Nano (Free)", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
    ("Poolside Laguna S-2.1 (Free)", "poolside/laguna-s-2.1:free"),
    ("Poolside Laguna XS-2.1 (Free)", "poolside/laguna-xs-2.1:free"),
    ("Gemma-4 31B IT (Free)", "google/gemma-4-31b-it:free"),
    ("Gemma-4 26B A4B IT (Free)", "google/gemma-4-26b-a4b-it:free"),
    ("Dots-3 Note Preview (Free)", "dots-studio/dots-3-note-preview:free"),
    ("Ling-3.0 Flash Sante (Free)", "inclusionai/ling-3.0-flash-sante:free"),
    ("LFM-2.5 2.6B (Free)", "liquid/lfm-2.5-2.6b:free"),
    ("Apex-1.1 Mini (Free)", "apodex/apodex-1.1-mini:free"),
    ("North Mini Code (Free)", "cohere/north-mini-code:free"),
    ("Inkling (Free)", "thinkingmachines/inkling:free"),
    ("Inkling Small (Free)", "thinkingmachines/inkling-small:free"),
    # Paid models (billed at real rates).
    ("Claude 3.5 Sonnet", "anthropic/claude-3.5-sonnet"),
    ("Claude 3 Opus", "anthropic/claude-3-opus"),
    ("GPT-4o", "openai/gpt-4o"),
    ("Gemini 1.5 Pro", "google/gemini-pro-1.5"),
    ("Llama 3 70B", "meta-llama/llama-3-70b-instruct"),
]

class ModelPicker(ModalScreen[str]):
    BINDINGS = [("escape", "dismiss_modal", "Cancel")]
    
    CSS = """
    ModelPicker {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    
    #picker-dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: #0f172a;
        border: round #00b8ff;
        color: #e2e8f0;
    }
    
    #picker-title {
        text-style: bold;
        color: #00ff9f;
        margin-bottom: 1;
        content-align: center middle;
    }
    
    OptionList {
        background: transparent;
        border: none;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Vertical(id="picker-dialog"):
            yield Static("Select an LLM Model", id="picker-title")
            options = [name for name, _ in MODELS]
            yield OptionList(*options, id="model-list")
            
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        index = event.option_index
        model_id = MODELS[index][1]
        self.dismiss(model_id)

    def action_dismiss_modal(self) -> None:
        self.dismiss("")
