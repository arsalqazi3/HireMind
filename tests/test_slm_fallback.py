from types import SimpleNamespace

import pytest

from hiremind import slm


def test_mlx_applies_adapter_only_to_matching_roles_selected_for_it(monkeypatch):
    class TorchPath(Exception):
        pass

    def load(**kwargs):
        raise TorchPath(kwargs.get("adapter_dir"))

    monkeypatch.setenv("HIREMIND_BACKEND", "mlx")
    monkeypatch.setenv("HIREMIND_DEVICE", "auto")
    monkeypatch.setenv("HIREMIND_ADAPTER_DIR", "example-adapter")
    monkeypatch.setenv("HIREMIND_ADAPTER_ROLES", "parser")
    monkeypatch.setattr(slm, "load", load)
    monkeypatch.setattr(slm, "_generate_mlx", lambda *args: args)

    assert slm.generate("critic", "example", 80) == ("critic", "example", 80, None, None)
    assert slm.generate("parser", "example") == ("parser", "example", 512, None, "example-adapter")
    with pytest.raises(TorchPath):
        slm.generate("writer", "example")

    monkeypatch.setenv("HIREMIND_DEVICE", "cpu")
    with pytest.raises(TorchPath, match="example-adapter"):
        slm.generate("parser", "example")


def test_async_gpu_error_during_token_read_falls_back_to_cpu(monkeypatch):
    class Inputs(dict):
        input_ids = SimpleNamespace(shape=(1, 4))

        def to(self, device):
            return self

    class Output:
        def __init__(self, device):
            self.device = device

        def __getitem__(self, index):
            return self

        def tolist(self):
            if self.device == "mps":
                raise RuntimeError("async device failure")
            return [42]

    class Model:
        device = SimpleNamespace(type="mps")

        def generate(self, **kwargs):
            return Output(self.device.type)

        def to(self, device):
            self.device = SimpleNamespace(type=device)
            return self

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return Inputs()

        def decode(self, tokens, **kwargs):
            assert tokens == [42]
            return "result"

    model = Model()
    monkeypatch.setenv("HIREMIND_BACKEND", "pytorch")
    monkeypatch.setenv("HIREMIND_DEVICE", "auto")
    monkeypatch.delenv("HIREMIND_ADAPTER_DIR", raising=False)
    monkeypatch.setattr(slm, "load", lambda: (Tokenizer(), model))
    monkeypatch.setattr(slm, "format_prompt", lambda *_: "prompt")
    with pytest.warns(UserWarning, match="using CPU"):
        assert slm.generate("critic", "prompt") == "result"
    assert model.device.type == "cpu"
