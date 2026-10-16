"""
Sign2Text - Text Buffer Module

Modular text layer that manages sign-to-text conversion with support for
words/signs, spaces, backspace/delete, and clear operations.

The text buffer maintains a sequence of words/signs that form a sentence,
with operations to add signs, insert spaces, delete characters/words,
and clear the entire buffer.

Design
------
- Words/signs are added to the buffer as units
- Space adds a word separator
- Backspace removes the last character or word depending on context
- Clear resets the entire buffer
- Supports undo/redo for extensibility
"""

from __future__ import annotations
from typing import Deque, List, Optional, Tuple
from collections import deque


class TextBuffer:
    """Manages signed-based text input with sentence building capabilities.

    The text buffer maintains a sequence of signed words/phrases that
    form a complete sentence. Supports adding signs, spaces, backspace/delete,
    and clear operations.

    Character-by-letter entry design:
    - A-Z signs: append corresponding character to current word
    - Space sign: finalize current word and add space separator
    - Del sign: delete last character from current word
    - Nothing sign: no-op (ignored)

    Attributes:
        words: Deque of finalized words/phrases in the current sentence.
        current_word: The word currently being typed (characters added here
            won't be finalized until space is pressed).
        history: Stack of previous buffer states for undo support.
    """

    # Mapping from CNN class names to character operations
    _SIGN_TO_CHAR = {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
        "E": "E",
        "F": "F",
        "G": "G",
        "H": "H",
        "I": "I",
        "J": "J",
        "K": "K",
        "L": "L",
        "M": "M",
        "N": "N",
        "O": "O",
        "P": "P",
        "Q": "Q",
        "R": "R",
        "S": "S",
        "T": "T",
        "U": "U",
        "V": "V",
        "W": "W",
        "X": "X",
        "Y": "Y",
        "Z": "Z",
    }

    def __init__(self) -> None:
        """Initialize an empty text buffer."""
        self._words: Deque[str] = deque()  # Finalized words/phrases
        self._current_word: str = ""  # Word being actively typed
        self._history: List[Tuple[Deque[str], str]] = []  # For undo
        self._history_index: int = -1  # Current position in history

    def _save_history(self) -> None:
        """Save the current buffer state to the history stack.

        Saves a snapshot of the current buffer state (words and current_word)
        before making changes, for undo/redo support.
        """
        # Remove forward history if we're appending after undo
        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]
            self._history_index = len(self._history) - 1

        # Save current state
        self._history.append((list(self._words), self._current_word))
        self._history_index += 1

    def add_sign(self, sign: str) -> None:
        """Add a sign/label to the current word being typed.

        Supports character-by-letter entry:
        - A-Z: append corresponding character to current word
        - Space: handled by add_space()
        - Del: delete last character from current word
        - Nothing: no-op (ignored)

        Args:
            sign: The sign label to add (e.g. "A", "HELLO", "SPACE", "DEL", "NOTHING").
        """
        if not sign or not sign.strip():
            return

        sign = sign.strip().upper()

        # Handle special signs
        if sign == "DEL":
            # Delete last character from current word
            if self._current_word:
                self._current_word = self._current_word[:-1]
            # If current word becomes empty, nothing else to do
            return
        if sign == "NOTHING":
            # No-op - ignore
            return
        if sign == "SPACE":
            # Delegate to add_space() for proper space handling
            self.add_space()
            return

        # For A-Z signs, append the corresponding character
        char = self._SIGN_TO_CHAR.get(sign)
        if char is not None:
            # Add character to current word
            if self._current_word:
                self._current_word += char
            else:
                self._current_word = char
            return

        # For unknown signs (e.g. "HELLO", "YES"), use original behavior
        # Add sign to current word
        if self._current_word:
            self._current_word += f" {sign}"
        else:
            self._current_word = sign

    def add_space(self) -> None:
        """Finalize the current word and add a space separator.

        Moves the current word to the words deque and starts a new
        current word. If the current word is empty, just adds a space
        separator between words.
        """
        # Save history for undo before modifying
        if self._history_index < len(self._history) - 1:
            # Remove forward history if we're appending after undo
            self._history = self._history[: self._history_index + 1]
            self._history_index = len(self._history) - 1

        # Save current state before change
        self._save_history()

        # Move current word to words deque if not empty
        if self._current_word.strip():
            self._words.append(self._current_word.strip())
            self._current_word = ""

        # If current word is empty but we have words, add a space separator
        # (this handles pressing space twice or space at start)
        elif self._words and not self._words[-1].endswith(" "):
            # Keep repeated separator key presses idempotent.
            self._words[-1] += " "

    def backspace(self) -> Optional[str]:
        """Remove the last character or word.

        Removes the last character from the current word being typed.
        If the current word is empty, removes the last word from the
        words deque.

        Returns:
            The removed word if a word was removed, None if only a
            character was removed from current word.
        """
        # Save history for undo
        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]
            self._history_index = len(self._history) - 1

        # Save current state before change
        self._save_history()

        removed_word: Optional[str] = None

        if self._current_word:
            # Remove last character from current word
            self._current_word = self._current_word[:-1]
            if not self._current_word.strip():
                self._current_word = ""
        elif self._words:
            # A word separator does not change the meaning of backspace: it
            # deletes one character, including when the current word is empty.
            previous = self._words.pop()
            if previous.endswith(" "):
                previous = previous.rstrip()
            if previous:
                self._current_word = previous[:-1]
                removed_word = previous if not self._current_word else None

        return removed_word

    def clear(self) -> None:
        """Clear the entire text buffer, resetting all state."""
        # Save history
        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]
            self._history_index = len(self._history) - 1

        self._save_history()

        # Reset all state
        self._words.clear()
        self._current_word = ""
        self._history_index = -1

    def undo(self) -> bool:
        """Undo the last operation.

        Restores the buffer to its previous state from the history stack.

        Returns:
            True if undo was successful, False if no history to undo.
        """
        if self._history_index < 0:
            return False

        # Move back one step in history
        self._history_index -= 1

        # Restore state from history
        if self._history_index >= 0:
            previous_words, previous_current_word = self._history[self._history_index]
            self._words = deque(previous_words)
            self._current_word = previous_current_word

        return True

    def redo(self) -> bool:
        """Redo the undone operation.

        Restores the buffer to the state after the last undone operation.

        Returns:
            True if redo was successful, False if no redo available.
        """
        if self._history_index >= len(self._history) - 1:
            return False

        # Move forward one step in history
        self._history_index += 1

        # Restore state from history
        if self._history_index < len(self._history):
            previous_words, previous_current_word = self._history[self._history_index]
            self._words = deque(previous_words)
            self._current_word = previous_current_word

        return True

    @property
    def sentence(self) -> str:
        """Get the full sentence as a string.

        Returns:
            The complete sentence formed by all words/phrases,
            with spaces between words.
        """
        if self._words and self._current_word:
            # Words + current word being typed
            return " ".join(list(self._words) + [self._current_word])
        elif self._words:
            return " ".join(list(self._words))
        else:
            return self._current_word

    @property
    def current_word(self) -> str:
        """Get the word currently being typed."""
        return self._current_word

    @property
    def words(self) -> List[str]:
        """Get the list of finalized words/phrases."""
        return list(self._words)

    @property
    def char_count(self) -> int:
        """Get the total number of characters in the sentence (excluding spaces)."""
        return sum(len(word.replace(" ", "")) for word in self.words) + len(
            self._current_word.replace(" ", "")
        )

    @property
    def word_count(self) -> int:
        """Get the total number of words/phrases in the sentence."""
        return len(self._words) + (1 if self._current_word.strip() else 0)

    def to_dict(self) -> dict:
        """Serialize the buffer state to a dictionary.

        Returns:
            Dictionary representation of the text buffer.
        """
        return {
            "words": list(self._words),
            "current_word": self._current_word,
            "sentence": self.sentence,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TextBuffer":
        """Create a TextBuffer from a dictionary (for persistence).

        Args:
            data: Dictionary with buffer state.

        Returns:
            New TextBuffer instance initialized from the data.
        """
        buffer = cls()
        buffer._words = deque(data.get("words", []))
        buffer._current_word = data.get("current_word", "")
        buffer._history = []
        buffer._history_index = -1
        return buffer

    def __str__(self) -> str:
        """String representation of the buffer."""
        return self.sentence

    def __len__(self) -> int:
        """Number of words in the buffer."""
        return self.word_count
