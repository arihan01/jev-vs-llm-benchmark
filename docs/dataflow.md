# Data flow

What is sent to each engine, how it is built, what comes back and how it is scored. One call = one text, one question, one engine, one repeat.

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 520, "nodeSpacing": 40, "rankSpacing": 40}}}%%
flowchart TB
  subgraph prep["A · prepared once"]
    direction LR
    D[("Banking77 · BoolQ · Yelp")] -->|"300 / task, seed 42"| F["data/{task}.jsonl<br/>{item_id, text, gold}"]
    F -->|"first 5 / task"| P["perturb.py<br/>typo · paraphrase · padding<br/>distractor · S5 clean"]
    P --> X["data/experiments.json — per sample:<br/>{id, perturbation, clean_text, text,<br/>questions: [{id, type, instructions,<br/>options: {key: description}, gold}]}"]
    H["human writes Q2 + T"] --> X
  end

  X --> J1 & L1

  subgraph send["B · one request per question — same instructions, option descriptions and text to both"]
    direction LR
    J1["Jev request (JSON body)<br/>{<br/>&nbsp;&nbsp;&quot;model&quot;: &quot;jev-1.13.0&quot;,<br/>&nbsp;&nbsp;&quot;state&quot;: &quot;&lt;sample text&gt;&quot;,<br/>&nbsp;&nbsp;&quot;questions&quot;: {&quot;Q1&quot;: {<br/>&nbsp;&nbsp;&nbsp;&nbsp;&quot;type&quot;: &quot;choice | score | noul&quot;,<br/>&nbsp;&nbsp;&nbsp;&nbsp;&quot;instructions&quot;: &quot;&lt;question&gt;&quot;,<br/>&nbsp;&nbsp;&nbsp;&nbsp;&quot;criteria&quot;: {key: description}<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score: [level descriptions]<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;noul: {true: yes-text, false: no-text}<br/>&nbsp;&nbsp;}}<br/>}"]
    L1["LLM request (chat body)<br/>messages:<br/>&nbsp;&nbsp;system: You are a classifier. Reply only with JSON matching the schema.<br/>&nbsp;&nbsp;user:&nbsp;&nbsp;&nbsp;&lt;instructions&gt;<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Options:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;key&gt;: &lt;description&gt;&nbsp;&nbsp;(one line per option)<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Input:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt;sample text&gt;<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return {&quot;label&quot;: &lt;one option key&gt;, &quot;confidence&quot;: &lt;0-1&gt;}.<br/>response_format: json_schema {label: enum(option keys), confidence: 0–1}<br/>temperature: 0 · max tokens 200–400 · reasoning off"]
  end

  J1 -->|"POST · rate limit · retry<br/>latency = final attempt"| JA["Jev API"]
  L1 -->|"POST · rate limit · retry<br/>latency = final attempt"| LA["Chat API<br/>OpenRouter / OpenAI"]

  subgraph recv["C · received"]
    direction LR
    JR["Jev response<br/>{&quot;answers&quot;: {&quot;Q1&quot;: {<br/>&nbsp;&nbsp;choice: {&quot;choice&quot;: key, &quot;probabilities&quot;: {key: p}}<br/>&nbsp;&nbsp;score:&nbsp;&nbsp;{&quot;score&quot;: 3.39, &quot;probabilities&quot;: {&quot;0&quot;: p, …, &quot;4&quot;: p}}<br/>&nbsp;&nbsp;noul:&nbsp;&nbsp;&nbsp;{&quot;noul&quot;: P(yes)}<br/>}},<br/>&nbsp;&quot;usage&quot;: {input_tokens, output_tokens, cost?}}"]
    LR["LLM response (expected)<br/>choices[0].message.content =<br/>&nbsp;&nbsp;'{&quot;label&quot;: &quot;&lt;option key&gt;&quot;, &quot;confidence&quot;: 0.96}'<br/>usage: {prompt_tokens, completion_tokens,<br/>&nbsp;&nbsp;cost?, prompt_tokens_details.cached_tokens?}<br/>anything else (empty, fenced, prose, key not in options)<br/>is parsed if possible, otherwise invalid"]
  end
  JA --> JR
  LA --> LR

  JR & LR --> C[(".cache/ {request, response, latency_ms, status}<br/>re-runs re-parse, never re-call")]

  subgraph parse["D · parse to one Prediction {label, probs, confidence, value, valid}"]
    direction LR
    JP["Jev<br/>choice: label = choice<br/>score: label = argmax(probabilities), value = score<br/>noul: label = yes if P ≥ 0.5, value = P(yes), probs = {yes: P, no: 1−P}<br/>confidence = probs[label] · valid = label ∈ options"]
    LP["LLM<br/>strip code fences → json.loads → need {label}<br/>label ∉ options → invalid (scored wrong)<br/>confidence clamped to [0,1], 1/n if missing<br/>probs = {label: c, others: (1−c)/(n−1)}<br/>score: value = float(label) · noul: value = probs[yes]"]
  end
  C --> JP & LP

  JP & LP --> R["raw.parquet row<br/>exp · sample · question · engine · repeat · gold<br/>label · value · probs · confidence · valid · correct = valid ∧ label == gold<br/>latency_ms · tokens_in/out · cost = usage.cost or tokens × price · status · error"]
  R --> M["metrics.py (status 200 only)<br/>cell = 3 repeats: accuracy · consistency · mean P(T) · invalid<br/>scorecard: ECE · Brier/AUROC (noul) · MAE (score)"]
  M --> O["cells.csv · scorecard.csv · verdict.md · report.md/.pdf · dashboard"]
```

Paraphrase is written by a generator model that is not under test; its output is frozen in `experiments.json` like every other sample.

Both engines see the same three strings: the question's instructions, every option's description, and the sample text. Jev gets them as fields, the LLM as one prompt plus a JSON schema. For BoolQ the `{passage, question}` object goes to Jev as JSON and to the LLM as JSON text inside the prompt.
