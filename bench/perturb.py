"""Text perturbations applied to the experiment samples. Typo, padding,
distractor are scripted; paraphrase comes from a generator model
that is not under test."""

import hashlib
import random
import string

from . import prompts

GREETINGS = ["Hi there, hope you're having a good day.", "Hello! Thanks in advance for reading this."]
SIGNOFFS = ["Thanks so much, Sam.", "Best regards, and have a lovely week. Alex"]
OFF_TOPIC = [
    "The weather has been unusually warm this week.",
    "My neighbour just adopted a very loud parrot.",
    "I've been meaning to repaint the kitchen for months.",
    "The train was ten minutes late again this morning.",
    "We are planning a small picnic for the weekend.",
    "My cousin recently started learning the cello.",
    "The local library extended its opening hours.",
    "I finally finished that long novel I was reading.",
]


def map_text(text, fn, fields=("passage", "question")):
    if isinstance(text, dict):
        return {k: fn(v) if k in fields else v for k, v in text.items()}
    return fn(text)


def typo(s: str, rng: random.Random, rate: float = 0.1) -> str:
    """Hasty-typing noise: in about `rate` of the words longer than three letters,
    swap two adjacent letters or drop one; then lower-case and strip punctuation."""
    words = s.split()
    for i, w in enumerate(words):
        if len(w) > 3 and rng.random() < rate:
            j = rng.randrange(len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :] if rng.random() < 0.5 else w[:j] + w[j + 1 :]
    out = " ".join(words).lower()
    return out.translate(str.maketrans("", "", string.punctuation))


def pad(s: str, rng: random.Random) -> str:
    """Bury the text in the middle of off-topic filler of about its own length
    (capped at 2000 chars), between a greeting and a sign-off."""
    filler: list[str] = []
    target = min(len(s), 2000)
    while sum(len(x) + 1 for x in filler) < target:
        filler.append(rng.choice(OFF_TOPIC))
    half = len(filler) // 2
    return " ".join([rng.choice(GREETINGS), *filler[:half], s, *filler[half:], rng.choice(SIGNOFFS)])


def distractor(task: str, item: dict, rng: random.Random, others: list[dict]):
    """Insert another class's keyword in negated form; the meaning is unchanged."""
    text, gold = item["text"], item["gold"]
    if task == "choice":
        other = rng.choice([k for k in prompts.banking77_labels() if k != gold])
        return f"{text} This isn't about {other.replace('_', ' ')}."
    if task == "score":
        g = int(gold)
        phrase = (
            "I wouldn't call it terrible."
            if g >= 3
            else "I can't say it was amazing."
            if g <= 1
            else "It wasn't perfect, but it wasn't awful either."
        )
        return f"{text} {phrase}"
    other_q = rng.choice([o for o in others if o["item_id"] != item["item_id"]])["text"]["question"]
    return {**text, "passage": f"{text['passage']} This passage is not about whether {other_q}."}


def paraphrase_prompt(task: str, text) -> str:
    what = "passage" if task == "noul" else "text"
    src = text["passage"] if task == "noul" else text
    return (
        f"Paraphrase this {what} using different wording and sentence structure. Keep every fact and the "
        "exact meaning and sentiment. Do not strengthen or soften any opinion, intensity or certainty "
        "('pretty good' must not become 'great'), and add nothing that is not in the original. "
        f"Return only the rewritten text, with no preamble or quotes.\n\n{src}"
    )


def paraphrase(task: str, source_id: str, text):
    """Generator-model paraphrase, cached on disk. The cache key includes the
    generator model, so configs with different generators never share output."""
    # Imported here so building experiments without --llm needs no HTTP stack.
    import asyncio

    import httpx

    from .adapters.base import Adapter, DailyLimitError
    from .cache import get, put
    from .config import headers, load

    cfg = load()
    model = cfg["perturb_generator"]["model"]
    prompt = paraphrase_prompt(task, text)
    k = f"generator|{model}|{task}|{source_id}|paraphrase|{hashlib.sha1(prompt.encode()).hexdigest()[:10]}"
    rec = get("_generator", task, k)
    if rec is None:
        gen = Adapter("_generator", {"kind": "llm", "model": model})
        body = {"model": model, "temperature": 0, "max_tokens": 1200, "messages": [{"role": "user", "content": prompt}]}

        async def post() -> dict:
            async with httpx.AsyncClient(timeout=cfg.get("timeout_s", 60)) as client:
                return await gen.call(client, body, headers(gen.spec))

        try:
            rec = {"response": asyncio.run(post())["response"]}
        except DailyLimitError as e:
            raise SystemExit(f"generator {model} hit its daily request limit; re-run tomorrow ({e})") from None
        put("_generator", task, k, rec)
    new = rec["response"]["choices"][0]["message"]["content"].strip()
    return {**text, "passage": new} if task == "noul" else new
