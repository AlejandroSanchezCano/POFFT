"""
===============================================================================
Title:      Hook
Outline:    Hook class that allows to register hooks on PyTorch modules, 
            capturing inputs and outputs.
Date:       13/10/2026
===============================================================================
"""

# Built-in modules
from contextlib import contextmanager
from typing import Callable, Optional, Any, Literal

# Third-party modules
import torch
from torch import nn

HookType = Literal[
    "forward",
    "forward_pre",
    "backward",
    "backward_pre",
]

class Hook:

    def __init__(
        self,
        module: nn.Module,
        hook_type: HookType = "forward",
        callback: Optional[Callable] = None,
        *,
        enabled: bool = True,
        auto_register: bool = True,
        capture: bool = False,
        detach: bool = True,
        clone: bool = False,
        cpu: bool = False,
        prepend: bool = False,
        with_kwargs: bool = False,
    ):
        self.module = module
        self.hook_type = hook_type
        self.callback = callback

        self.enabled = enabled
        self.capture = capture
        self.detach = detach
        self.clone = clone
        self.cpu = cpu
        self.prepend = prepend
        self.with_kwargs = with_kwargs

        self.handle = None
        self.history = []
        self.last_input = None
        self.last_output = None
        self.call_count = 0

        if auto_register:
            self.register()

    def _snapshot(self, value):
        """Optionally detach and clone captured tensors."""
        if isinstance(value, torch.Tensor):
            if self.detach:
                value = value.detach()
            if self.clone:
                value = value.clone()
            if self.cpu:
                value = value.cpu()
            return value

        if isinstance(value, tuple):
            return tuple(self._snapshot(v) for v in value)
        if isinstance(value, list):
            return [self._snapshot(v) for v in value]
        if isinstance(value, dict):
            return {k: self._snapshot(v) for k, v in value.items()}

        return value

    def _hook_fn(self, *args):
        if not self.enabled:
            return None

        self.call_count += 1

        if self.with_kwargs and self.hook_type in (
            "forward", "forward_pre"
        ):
            module, inputs, kwargs, *rest = args
            output = rest[0] if rest else None
        else:
            module, inputs, *rest = args
            kwargs = None
            output = rest[0] if rest else None

        if self.capture:
            self.last_input = self._snapshot(inputs)
            self.last_output = self._snapshot(output)

            self.history.append({
                "input": self.last_input,
                "output": self.last_output,
            })

        if self.callback is not None:
            return self.callback(*args)

        return None

    def register(self):
        """Register the hook, avoiding duplicate registration."""
        if self.handle is not None:
            return self

        if self.hook_type == "forward":
            self.handle = self.module.register_forward_hook(
                self._hook_fn,
                prepend=self.prepend,
                with_kwargs=self.with_kwargs,
            )

        elif self.hook_type == "forward_pre":
            self.handle = self.module.register_forward_pre_hook(
                self._hook_fn,
                prepend=self.prepend,
                with_kwargs=self.with_kwargs,
            )

        elif self.hook_type == "backward":
            if self.with_kwargs:
                raise ValueError(
                    "with_kwargs is not supported for backward hooks"
                )
            self.handle = self.module.register_full_backward_hook(
                self._hook_fn,
                prepend=self.prepend,
            )

        elif self.hook_type == "backward_pre":
            if self.with_kwargs:
                raise ValueError(
                    "with_kwargs is not supported for backward hooks"
                )
            self.handle = self.module.register_full_backward_pre_hook(
                self._hook_fn,
                prepend=self.prepend,
            )

        else:
            raise ValueError(
                f"Unsupported hook type: {self.hook_type}"
            )

        return self

    def remove(self):
        """Permanently unregister until register() is called."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return self

    def enable(self):
        self.enabled = True
        return self

    def disable(self):
        self.enabled = False
        return self

    def set_callback(self, callback: Optional[Callable]):
        self.callback = callback
        return self

    def clear(self):
        """Clear captured data and statistics."""
        self.history.clear()
        self.last_input = None
        self.last_output = None
        self.call_count = 0
        return self

    @property
    def registered(self) -> bool:
        return self.handle is not None

    @contextmanager
    def paused(self):
        """Temporarily disable the callback."""
        previous = self.enabled
        self.disable()
        try:
            yield self
        finally:
            self.enabled = previous

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()
        return False

    def __repr__(self):
        return (
            f"Hook(type={self.hook_type!r}, "
            f"registered={self.registered}, "
            f"enabled={self.enabled}, "
            f"calls={self.call_count})"
        )

if __name__ == "__main__":

    # Example usage
    model = nn.Linear(10, 5)
    hook = Hook(model, hook_type="forward", capture=True)

    x = torch.randn(2, 10)
    output = model(x)

    print("Hooked inputs:", hook.last_input)
    print("Hooked outputs:", hook.last_output)