"""Training-only PARSeqTiny AR auxiliary reader; discarded for CTC export.
Existing pinned PARSeq vendor/license applies; PP graph from Apache2 official model.
"""
import math, torch
from .models import download_pretrained
from strhub.models.utils import create_model


class AuxiliaryReader(torch.nn.Module):
    def __init__(self, base, pretrained_dir):
        super().__init__()
        self.base = base
        nodes = list(base.graph.nodes)
        softout = nodes[-1]
        logits = softout.args[0]
        features = logits.args[0].args[0]
        assert features.op in ["call_module", "call_function"]
        dim = None
        # The final classifier is MatMul(features, learned class columns) then Add.
        classifier = logits.args[0]
        assert "mat_mul" in classifier.name
        weight = getattr(base.initializers, classifier.args[1].target.split(".")[-1])
        dim = weight.shape[0]
        softout.args = ((logits, features),)
        base.recompile()
        old = create_model("parseq-tiny", False)
        old.model.load_state_dict(
            torch.load(
                download_pretrained("parseq-tiny", pretrained_dir),
                weights_only=True,
                map_location="cpu",
            )
        )
        chars = "".join(old.tokenizer._itos[1:-2]) + " "
        new = create_model(
            "parseq-tiny", False, charset_train=chars, charset_test=chars
        )
        src = old.model.state_dict()
        dst = new.model.state_dict()
        for key, v in src.items():
            if dst[key].shape == v.shape:
                dst[key] = v
            elif key in ["head.weight", "head.bias", "text_embed.embedding.weight"]:
                for i, token in enumerate(old.tokenizer._itos):
                    if (
                        i < v.shape[0]
                        and new.tokenizer._stoi[token] < dst[key].shape[0]
                    ):
                        dst[key][new.tokenizer._stoi[token]] = v[i]
            else:
                raise RuntimeError(key)
        new.model.load_state_dict(dst)
        self.tokenizer = new.tokenizer
        self.reader = new.model
        self.reader.encoder = torch.nn.Identity()
        self.reader.max_label_length = 12
        self.reader.decode_ar = True
        self.reader.refine_iters = 1
        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(dim, 192), torch.nn.LayerNorm(192)
        )
        position = torch.arange(40, dtype=torch.float32)[:, None]
        div = torch.exp(
            torch.arange(0, 192, 2, dtype=torch.float32) * (-math.log(10000) / 192)
        )
        pe = torch.zeros(40, 192)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("position", pe[None] * 0.1)
        self.memory = None

    def forward(self, x):
        logits, features = self.base(x)
        self.memory = features if self.training else None
        return logits

    def auxiliary_loss(self, labels):
        target = self.tokenizer.encode(list(labels), self.memory.device)
        context = target[:, :-1]
        n = context.shape[1]
        mask = torch.ones((n, n), device=context.device, dtype=torch.bool).triu(1)
        padding = context.eq(self.tokenizer.pad_id)
        memory = self.adapter(self.memory) + self.position[:, : self.memory.shape[1]]
        decoded = self.reader.decode(
            context,
            memory,
            tgt_mask=mask,
            tgt_padding_mask=padding,
            tgt_query=self.reader.pos_queries[:, :n].expand(len(labels), -1, -1),
            tgt_query_mask=mask,
        )
        logits = self.reader.head(decoded)
        return torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1),
            target[:, 1:].flatten(),
            ignore_index=self.tokenizer.pad_id,
        )
