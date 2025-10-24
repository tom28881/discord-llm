import os
import sys
import types
from typing import Any
from dotenv import load_dotenv

try:  # pragma: no cover - exercised indirectly by tests
    import google.generativeai as genai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback path covered in tests
    genai = None  # Will be lazily imported when needed if the package becomes available

    def _missing_dependency(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            "google.generativeai is not installed. Install 'google-generativeai' to enable LLM features."
        )

    class _MissingGenerativeModel:  # pragma: no cover - simple sentinel used in tests
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_dependency()

        def __getattr__(self, _name: str) -> Any:
            return _missing_dependency

    _stub = types.ModuleType("google.generativeai")
    _stub.configure = _missing_dependency  # type: ignore[attr-defined]
    _stub.GenerativeModel = _MissingGenerativeModel  # type: ignore[attr-defined]
    _stub.__all__ = ["configure", "GenerativeModel"]
    sys.modules.setdefault("google.generativeai", _stub)

load_dotenv()

def _ensure_genai_module() -> Any:
    global genai

    if genai is not None:
        return genai

    try:
        import google.generativeai as genai_module  # type: ignore
    except ModuleNotFoundError:
        return None

    genai = genai_module
    return genai


def get_completion(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """
    Get completion from Google Gemini API
    
    Args:
        prompt: The prompt to send to the model
        model: The model name (defaults to gemini-2.5-flash)
    
    Returns:
        The model's response text
    """
    
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables")
        return ""
    
    genai_module = _ensure_genai_module()

    if genai_module is None:
        print("Error: google.generativeai package is not installed")
        return ""

    try:
        # Configure the API key
        genai_module.configure(api_key=api_key)

        # Initialize the model
        gemini_model = genai_module.GenerativeModel(model_name=model)

        # Generate response
        response = gemini_model.generate_content(prompt)

        # Return the text response
        return getattr(response, "text", "") or ""

    except ModuleNotFoundError as exc:
        print(f"Error: {exc}")
        return ""
    except Exception as e:
        print(f"An error occurred with Gemini API: {e}")
        return ""