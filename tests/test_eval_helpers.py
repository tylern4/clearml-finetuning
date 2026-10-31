import sys

sys.path.insert(0, ".")
from scripts.eval_agent import strip_think, chat

OPEN = "<" + "think" + ">"
CLOSE = "<" + "/think" + ">"

assert strip_think(OPEN + " reasoning... " + CLOSE + " code") == "code"
assert strip_think("text " + OPEN + " unterminated tail") == "text"
assert strip_think("no think here") == "no think here"
assert strip_think(None) is None
print("strip_think OK")

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
out = chat(tok, [{"role": "user", "content": "hi"}])
tail = out[out.rfind("assistant"):]
print("chat tail:", repr(tail[-60:]))
assert not tail.rstrip().endswith(OPEN)  # not an open think block
default = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                 tokenize=False, add_generation_prompt=True)
assert default != out
print("chat() thinking disabled OK")
