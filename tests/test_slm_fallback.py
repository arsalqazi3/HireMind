from types import SimpleNamespace

import pytest

from hiremind import slm


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
    monkeypatch.setenv("HIREMIND_DEVICE", "auto")
    monkeypatch.delenv("HIREMIND_ADAPTER_DIR", raising=False)
    monkeypatch.setattr(slm, "load", lambda: (Tokenizer(), model))
    monkeypatch.setattr(slm, "format_prompt", lambda *_: "prompt")
    with pytest.warns(UserWarning, match="using CPU"):
        assert slm.generate("critic", "prompt") == "result"
    assert model.device.type == "cpu"
