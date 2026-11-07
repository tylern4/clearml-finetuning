"""Create training data from a REMOTE model (OpenAI-compatible API).

Standalone: no ClearML task, no GPUs. This is the remote-model twin of
scripts/code_distill.py:

  1. load the problem set -- the same {"messages": [...], "meta": {...}}
     JSONL rows the code_data stage emits (local path, s3://, or https://
     URL; plain .jsonl or .zip),
  2. strip each row's reference assistant answer and context-fit the prompt,
  3. ask the remote model for its own assistant turn (concurrent, with
     retries and a resume journal so an interrupted run costs nothing),
  4. validate the assembled rows and upload one JSONL to S3.

Training chunks consume the result with `train_student.py --extra-data`;
mixing several models is just comma-separating several URLs in that flag.

Token accounting: reasoning traces are stored think-wrapped ahead of the
answer (R1-style distillation), so rows are much longer than answers-only
rows; the student trains with --max-len 16384 and tail-truncates longer rows
(rows with no supervised tokens at all are dropped). The row-size report at
the end shows how many rows exceed the training window.

Examples
--------
  python scripts/make_remote_distill.py \
      --url https://llm.api.example.com/v1 --token "$TOKEN" --model org/model-a \
      --input s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/clearml-finetuning/qwen3.8-coder-v2/finetune-code_data.b426ba99ad90419a8243282883ce2457/artifacts/code-data/code.zip \
      --s3-uri s3://s3-bucket-loadbalancer.das-redpanda.production.svc.spin.nersc.org:5432/dasml/clearml/remote_distills/model-a.jsonl

  # then, in a train stage task (clearml_stage --extra-data or directly):
  #   --extra-data s3://.../remote_distills/model-a.jsonl,s3://.../model-b.jsonl
"""

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clearml_finetuning import data  # noqa: E402  (stdlib-only helpers)


# ---------------------------------------------------------------- S3 plumbing
def parse_s3_uri(uri):
    """Split an s3 uri into (endpoint_or_None, bucket, key).

    Custom endpoints (self-hosted minions, the NERSC loadbalancer) carry a
    host[:port] netloc, so `s3://host:5432/bucket/key` means endpoint
    https://host:5432 with bucket/key inside the path.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    netloc, _, path = uri[len("s3://"):].partition("/")
    if not path:
        raise ValueError(f"s3 uri needs bucket/key: {uri}")
    if ":" in netloc or "." in netloc:
        bucket, _, key = path.partition("/")
        if not bucket:
            raise ValueError(f"s3 uri needs bucket/key after the endpoint: {uri}")
        return "https://" + netloc, bucket, key
    return None, netloc, path


def s3_client(args):
    import boto3
    from botocore.config import Config

    endpoint, _, _ = parse_s3_uri(args.s3_uri or args.input)
    if args.s3_endpoint:
        endpoint = args.s3_endpoint
    access = args.s3_access_key or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = args.s3_secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    session = args.s3_session_token or os.environ.get("AWS_SESSION_TOKEN")
    if not (access and secret):
        raise SystemExit("S3 credentials required: --s3-access-key/--s3-secret-key "
                         "or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY")
    cfg = Config(retries={"max_attempts": 5}, connect_timeout=30,
                 read_timeout=300)
    if endpoint:
        cfg = cfg.merge(Config(s3={"addressing_style": "path"}))
    if args.s3_insecure and endpoint:
        # endpoint speaks plain HTTP (e.g. the NERSC spin loadbalancer,
        # `secure: false` in clearml.conf)
        endpoint = endpoint.replace("https://", "http://")
        return boto3.client(
            "s3", endpoint_url=endpoint, aws_access_key_id=access,
            aws_secret_access_key=secret, aws_session_token=session, config=cfg)
    return boto3.client(
        "s3", endpoint_url=endpoint, aws_access_key_id=access,
        aws_secret_access_key=secret, aws_session_token=session, config=cfg)


def _sm_ops():
    """ClearML StorageManager as credential fallback: reads the bucket keys
    from clearml.conf (same as every stage artifact download). No ClearML
    task is created."""
    from clearml.storage import StorageManager

    def download(uri, dest):
        local = StorageManager.get_local_copy(remote_url=uri)
        if not local:
            raise SystemExit(f"could not fetch {uri} via StorageManager")
        if os.path.isdir(local):
            # StorageManager auto-extracts artifact zips into a folder;
            # resolve_input() locates the .jsonl inside it
            return local
        if local != dest:
            import shutil
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            shutil.copy(local, dest)
        return dest

    def upload(local_path, uri):
        if uri.endswith("/"):
            uri += os.path.basename(local_path)
        StorageManager.upload_file(local_path, uri)
        return uri

    return download, upload


def _boto_ops(args):
    c = s3_client(args)

    def download(uri, dest):
        _, bucket, key = parse_s3_uri(uri)
        print(f"downloading s3://{bucket}/{key}", flush=True)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        c.download_file(bucket, key, dest)
        return dest

    def upload(local_path, uri):
        if uri.endswith("/"):
            uri += os.path.basename(local_path)
        _, bucket, key = parse_s3_uri(uri)
        print(f"uploading {local_path} -> s3://{bucket}/{key}", flush=True)
        c.upload_file(local_path, bucket, key,
                      ExtraArgs={"ContentType": "application/octet-stream"})
        return uri

    return download, upload


def get_s3_ops(args):
    have_creds = bool(
        (args.s3_access_key or os.environ.get("AWS_ACCESS_KEY_ID"))
        and (args.s3_secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")))
    if have_creds:
        return _boto_ops(args)
    print("no explicit S3 creds given; using clearml StorageManager "
          "(bucket keys from clearml.conf)", flush=True)
    return _sm_ops()


def s3_upload(args, ops, local_path):
    return ops[1](local_path, args.s3_uri)


# ------------------------------------------------------------- input fetching
def http_download(url, dest, headers=None):
    print(f"downloading {url}", flush=True)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def extract_zip(zip_path, cache_dir):
    """Extract a .zip to the cache dir; returns the extraction directory."""
    out_dir = os.path.join(
        cache_dir, "unzipped_" + hashlib.sha1(zip_path.encode()).hexdigest()[:12])
    if not os.path.isdir(out_dir):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
    return out_dir


def locate_jsonl(path):
    """Pick the input .jsonl out of a directory (StorageManager unpacks
    artifact zips into folders). Train-split names preferred."""
    if os.path.isfile(path):
        return path
    jsonls = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".jsonl"):
                jsonls.append(os.path.join(root, f))
    if not jsonls:
        raise SystemExit(f"no .jsonl found under {path}")
    for name in ("code_train.jsonl", "train.jsonl"):
        for p in jsonls:
            if os.path.basename(p) == name:
                return p
    if len(jsonls) == 1:
        return jsonls[0]
    raise SystemExit(f"multiple .jsonl under {path}; pick one: {jsonls}")


def resolve_input(args, ops):
    """Local path to the input .jsonl (downloads/unzips remote inputs)."""
    cache_dir = os.path.expanduser(args.cache)
    os.makedirs(cache_dir, exist_ok=True)
    inp = args.input
    if os.path.exists(inp):
        local = inp
    elif inp.startswith("s3://"):
        local = ops[0](inp, os.path.join(
            cache_dir, hashlib.sha1(inp.encode()).hexdigest()[:16] + os.path.basename(inp)))
    elif inp.startswith(("http://", "https://")):
        headers = {}
        if args.input_header:
            for h in args.input_header:
                k, _, v = h.partition(":")
                headers[k.strip()] = v.strip()
        local = http_download(inp, os.path.join(
            cache_dir, hashlib.sha1(inp.encode()).hexdigest()[:16] + os.path.basename(inp)),
            headers)
    else:
        raise SystemExit(f"--input {inp!r} is not an existing file, s3:// or http(s):// url")
    if os.path.isdir(local):
        return locate_jsonl(local)
    if local.endswith(".zip"):
        return locate_jsonl(extract_zip(local, cache_dir))
    return local


# --------------------------------------------------------------- prompt prep
def fit_messages(messages, max_chars):
    """Trim until the prompt fits `max_chars` (roughly max_chars/3.5 tokens).

    max_chars <= 0 disables trimming entirely: the full problem is sent
    as-is (default --max-prompt-chars 0). When trimming is enabled, drop the
    oldest non-system turns first; hard-truncate the user text only if
    [system?, user] remains.
    """
    messages = data.normalize_messages([dict(m) for m in messages])
    if max_chars <= 0:
        return messages
    over = sum(len(m.get("content") or "") for m in messages) - max_chars
    while over > 0 and len(messages) > 2:
        del messages[1 if messages[0].get("role") == "system" else 0]
        over = sum(len(m.get("content") or "") for m in messages) - max_chars
    if over > 0:
        user = messages[-1]
        if user.get("role") != "user":
            return None
        keep = max(0, len(user.get("content") or "") - over - 40)
        if keep <= 0:
            return None
        user["content"] = user["content"][:keep] + "\n... [truncated]"
    return messages


# ------------------------------------------------------------------ remote API
# think tags are built from chr() so no shell/heredoc/filter pass can ever
# mangle the literal markers the training format depends on
THINK_OPEN = chr(60) + "think" + chr(62)
THINK_CLOSE = chr(60) + "/think" + chr(62)


def with_reasoning(reasoning, content):
    """Combine a reasoning trace + final answer into one assistant text.

    R1-style reasoning distillation: the trace is stored think-wrapped ahead
    of the answer so the student learns to reason before answering. Empty
    reasoning returns the content unchanged (non-thinking servers/models)."""
    reasoning = (reasoning or "").strip()
    if not reasoning:
        return content
    return f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n\n{content}"


def chat_remote_full(args, messages, extra_body=None):
    """One chat completion; returns (reasoning, content). Retries transient errors.

    Reasoning models (e.g. GLM) return thinking in a separate `message.reasoning`
    field; it is captured so it can be stored think-wrapped in the training row.
    Non-reasoning servers return ("", content)."""
    url = args.url.rstrip("/") + "/chat/completions"
    body = {"model": args.model, "messages": messages,
            "temperature": args.temperature, "max_tokens": args.max_tokens}
    if extra_body:
        body.update(extra_body)
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {args.token}"}
    last_err = None
    for attempt in range(args.retries + 1):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                payload = json.loads(r.read().decode())
            msg = payload["choices"][0]["message"]
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            return reasoning, msg.get("content") or ""
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")[:300]
            if e.code == 400 and "max_tokens" in err_body and "max_tokens" in body:
                # newer OpenAI-style endpoints want max_completion_tokens
                body["max_completion_tokens"] = body.pop("max_tokens")
                continue
            transient = e.code in (408, 409, 425, 429) or e.code >= 500
            last_err = f"HTTP {e.code}: {err_body}"
            if not transient:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < args.retries:
            time.sleep(min(60.0, 3.0 * (2 ** attempt)))
    raise RuntimeError(f"chat completion failed after {args.retries + 1} attempts: {last_err}")


def chat_remote(args, messages, extra_body=None):
    """One chat completion; returns assistant text only (reasoning discarded).

    Kept for callers that don't distill reasoning (make_remote_tasks); the
    distill generator uses chat_remote_full to keep the reasoning trace."""
    return chat_remote_full(args, messages, extra_body=extra_body)[1]


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(
        description="distill a remote model's answers into a training JSONL on S3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("REMOTE_BASE_URL", ""),
                    help="OpenAI-compatible API base, e.g. https://host/v1")
    ap.add_argument("--token", default=os.environ.get("REMOTE_API_TOKEN",
                    os.environ.get("OPENAI_API_KEY", "")),
                    help="bearer token for --url (env REMOTE_API_TOKEN)")
    ap.add_argument("--model", required=True, help="model name served at --url")
    ap.add_argument("--input", required=True,
                    help="problem rows JSONL: local path, s3:// or https:// url; "
                         ".zip accepted (rows: {'messages': [...], 'meta': {...}}; "
                         "trailing assistant turn is replaced)")
    ap.add_argument("--input-header", action="append", default=[],
                    help="extra header for https --input, 'Key: Value' (repeatable)")
    ap.add_argument("--s3-uri", required=True,
                    help="destination, s3://host:port/bucket/prefix/name.jsonl "
                         "(trailing / = prefix, basename appended)")
    ap.add_argument("--s3-endpoint", default="",
                    help="override the endpoint derived from --s3-uri, e.g. http://host:9000")
    ap.add_argument("--s3-access-key", default="", help="or AWS_ACCESS_KEY_ID")
    ap.add_argument("--s3-secret-key", default="", help="or AWS_SECRET_ACCESS_KEY")
    ap.add_argument("--s3-session-token", default="", help="or AWS_SESSION_TOKEN")
    ap.add_argument("--s3-insecure", action="store_true",
                    help="talk plain HTTP to the S3 endpoint (boto3 path only; "
                         "for endpoints with secure:false in clearml.conf)")
    ap.add_argument("--out", default="",
                    help="local output jsonl (default: ./remote_distill_<model>.jsonl)")
    ap.add_argument("--cache", default="~/.cache/remote_distill",
                    help="cache dir for downloaded inputs / resume journals")
    ap.add_argument("--limit", type=int, default=0, help="cap problems; 0 = all")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=900,
                    help="completion budget; reasoning + answer share it, and "
                         "only rows WITH a final answer are kept (reasoning "
                         "alone would train non-termination)")
    ap.add_argument("--train-max-len", type=int, default=49152,
                    help="the student's --max-len, for reporting only: how many "
                         "rows exceed the training window. Reasoning rows need "
                         "the raised window - a truncated row loses its answer")
    ap.add_argument("--max-prompt-chars", type=int, default=0,
                    help="prompt context cap in characters (~3.5 chars/token); "
                         "0 = send the full problem untruncated (default). The "
                         "trainer keeps every row and only cuts the token tail "
                         "at --max-len, so full prompts = more training signal.")
    ap.add_argument("--retry-kwargs", default="",
                    help="JSON merged into the request body for a RESCUE pass "
                         "over problems whose first attempt returned empty "
                         "content; empty = no rescue pass (empties are retried "
                         "for free on rerun). Prefer empty when distilling "
                         "reasoning: rescue kwargs that bound thinking would "
                         "store rows without a trace")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=300, help="per-request seconds")
    args = ap.parse_args()

    if not args.url:
        raise SystemExit("--url required (or env REMOTE_BASE_URL)")
    if not args.token:
        raise SystemExit("--token required (or env REMOTE_API_TOKEN / OPENAI_API_KEY)")
    tag = args.model.replace("/", "_")
    args.out = args.out or f"remote_distill_{tag}.jsonl"
    # journal is keyed by model + input identity: resuming with a different
    # problem set under the same model must never reuse old completions
    input_sig = hashlib.sha1(
        (args.input + "|" + args.model + f"|seed{args.seed}|limit{args.limit}"
         + f"|chars{args.max_prompt_chars}").encode()
    ).hexdigest()[:10]
    journal = os.path.join(os.path.expanduser(args.cache),
                           f"journal_{tag}_{input_sig}.jsonl")

    # ---- load problems --------------------------------------------------------
    ops = get_s3_ops(args)  # (download_fn, upload_fn)
    rows = data.read_jsonl(resolve_input(args, ops))
    if args.limit > 0:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]
    problems = []
    n_drop = 0
    for i, r in enumerate(rows):
        msgs = r.get("messages") or []
        # the prompt is everything up to the LAST user turn: it strips the
        # reference assistant answer AND any trailing assistant turns (some
        # trajectory rows end with several), and guarantees the prompt ends
        # with a user turn for the model to answer
        user_idx = [j for j, m in enumerate(msgs) if m.get("role") == "user"]
        if not user_idx:
            n_drop += 1
            continue
        msgs = msgs[: user_idx[-1] + 1]
        fitted = fit_messages(msgs, args.max_prompt_chars)
        if not fitted:
            n_drop += 1
            continue
        problems.append((i, r, fitted))
    print(f"{len(problems)} usable problems ({n_drop} dropped), model={args.model}", flush=True)

    # ---- resume journal -------------------------------------------------------
    done = {}
    rescued = set()
    if os.path.exists(journal):
        for line in open(journal, encoding="utf-8"):
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["idx"]] = rec["text"]
                if rec.get("rescued"):
                    rescued.add(rec["idx"])
        print(f"resuming: {len(done)} completions already in journal {journal}", flush=True)
    jlock = threading.Lock()

    todo = [(i, r, msgs) for i, r, msgs in problems if i not in done]
    print(f"{len(todo)} to generate ({len(done)} done)", flush=True)

    # ---- generate -------------------------------------------------------------
    counters = {"done": len(done), "fail": 0}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_pool(todo, extra_body, label):
        """Generate completions for `todo`, journaling successes as they land."""
        if not todo:
            return

        def work(i, r, msgs):
            reasoning, content = chat_remote_full(args, msgs, extra_body=extra_body)
            if reasoning.strip():
                # normalize: field reasoning wins; drop any inline think block
                # the content itself carried so tags aren't doubled
                answer = data.strip_think(content or "")
            else:
                # no field reasoning: keep the content as-is, including any
                # inline think block the model wrote itself (never lose traces)
                answer = content or ""
            text = with_reasoning(reasoning, answer)
            return i, r, msgs, text, answer

        jf = open(journal, "a", encoding="utf-8")
        try:
            with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
                futs = {ex.submit(work, i, r, msgs): i for i, r, msgs in todo}
                for n, fut in enumerate(as_completed(futs), 1):
                    i = futs[fut]
                    try:
                        i, r, msgs, text, answer = fut.result()
                    except Exception as e:
                        counters["fail"] += 1
                        print(f"  [{n}/{len(todo)}] idx {i} FAILED: {e}", flush=True)
                        continue
                    if not answer.strip():
                        # no final answer -> nothing to supervise; reasoning alone
                        # would train non-termination. Not journaled, so a rerun
                        # retries it for free (fresh sample at temp > 0)
                        counters["fail"] += 1
                        print(f"  [{n}/{len(todo)}] idx {i} empty generation",
                              flush=True)
                        continue
                    with jlock:
                        jf.write(json.dumps({"idx": i, "text": text,
                                             **({"rescued": True} if extra_body else {})},
                                            ensure_ascii=False) + "\n")
                        jf.flush()
                        done[i] = text
                    if extra_body:
                        rescued.add(i)
                    counters["done"] += 1
                    if n % 25 == 0 or n == len(todo):
                        print(f"  [{n}/{len(todo)}] solved={counters['done']} "
                              f"failed={counters['fail']}", flush=True)
        finally:
            jf.close()

    run_pool(todo, None, "pass 1")

    # ---- rescue pass: re-ask the empties with extra request kwargs ------------
    # opt-in via --retry-kwargs (default off). Empties are not journaled, so
    # `done` is missing exactly the ones to re-ask, and a plain rerun retries
    # them for free with a fresh sample. Note: a rescue that changes thinking
    # behaviour (e.g. chat_template_kwargs {"thinking": false}) would produce
    # rows WITHOUT a reasoning trace, so leave this off when distilling
    # reasoning unless the mixed format is acceptable.
    if args.retry_kwargs:
        todo2 = [(i, r, msgs) for i, r, msgs in problems if i not in done]
        if todo2:
            extra = json.loads(args.retry_kwargs)
            print(f"rescue pass: {len(todo2)} without completions, retrying "
                  f"with {args.retry_kwargs}", flush=True)
            run_pool(todo2, extra, "rescue")

    # ---- assemble + validate --------------------------------------------------
    out_rows = []
    for i, r, msgs in problems:
        text = done.get(i)
        if not text or not text.strip():
            print(f"  assembly: idx {i} has no completion", flush=True)
            continue
        out = {"messages": msgs + [{"role": "assistant", "content": text}],
               "meta": {**(r.get("meta") or {}),
                        "source": ((r.get("meta") or {}).get("source", "?")
                                   + f"-remote:{args.model}")}}
        if i in rescued:
            # recovered by the rescue pass (only exists when --retry-kwargs is
            # set) after pass 1 returned empty
            out["meta"]["rescued"] = True
        out["messages"] = data.normalize_messages(out["messages"])
        if not any(m.get("role") == "user" for m in out["messages"]) or \
                not out["messages"][-1].get("content", "").strip():
            roles = [m.get("role") for m in out["messages"]]
            print(f"  assembly: idx {i} dropped by validation, roles={roles}",
                  flush=True)
            continue
        out_rows.append(out)

    data.write_jsonl(args.out, out_rows)
    print(f"wrote {len(out_rows)}/{len(problems)} rows to {args.out} "
          f"(failed/empty: {counters['fail']}, input dropped: {n_drop})", flush=True)
    sizes = sorted(sum(len(m.get("content") or "") for m in r["messages"])
                   for r in out_rows)
    if sizes:
        q = lambda f: sizes[min(len(sizes) - 1, int(f * (len(sizes) - 1)))]
        window = int(args.train_max_len * 3.5)
        over = sum(1 for s in sizes if s > window)
        print(f"row size: p50={q(0.5)} p90={q(0.9)} p100={q(1.0)} chars "
              f"(~3.5 chars/token; {over}/{len(sizes)} rows exceed the "
              f"{args.train_max_len}-tok training window - raise the trainer's "
              f"--max-len or those rows lose their answer tail)",
              flush=True)
    if not out_rows:
        raise SystemExit("no rows produced; nothing to upload")

    uri = s3_upload(args, ops, args.out)
    print(f"\nuse in a train stage with:\n"
          f"  --extra-data {uri}\n"
          f"(mix several models by comma-separating URLs; each URL may also be "
          f"a local or https path)", flush=True)


if __name__ == "__main__":
    main()
