# AI Customer Support Agent — AmazonHelp

An end-to-end AI customer-support prototype built on the **Customer Support on Twitter** dataset.

The system takes an incoming customer message and:

1. Classifies it into a small support-intent taxonomy.
2. Retrieves a historically similar AmazonHelp interaction.
3. Drafts a response grounded in the historical response.
4. Decides whether to automatically handle the request or escalate it to a human, with a reason.

The goal of this project is not to build a production support system, but to demonstrate a measurable and reproducible approach to intent classification, historical grounding, and safe automation.

---

## 1. Problem Framing

I selected **AmazonHelp** from the Customer Support on Twitter dataset because it contains a large number of customer-support interactions and covers diverse support problems.

### Definition of a good system

A useful support agent should:

- correctly identify the customer's primary support need;
- retrieve evidence from historically similar support interactions;
- produce a response consistent with how the brand handled similar cases;
- avoid confidently automating uncertain or account-specific cases;
- provide an explicit reason when escalating.

### What I did not build

This is intentionally a prototype.

I did not build:

- a production-grade LLM agent;
- live Twitter/X integration;
- real Amazon account/order access;
- authentication or customer identity verification;
- automatic execution of refunds, cancellations, or account changes;
- a production deployment;
- a large local language model.

The emphasis is on **measurable behavior and evaluation rather than system complexity**.

---

## 2. Dataset

Source:

**Customer Support on Twitter — Kaggle**

The original dataset contains approximately 3 million tweets and multiple brands.

For this project, I selected AmazonHelp.

### Data preparation

The pipeline:

```text
Raw Twitter dataset
        ↓
AmazonHelp interactions
        ↓
Customer → Brand response pairs
        ↓
Cleaning / deduplication
        ↓
English-language filtering
        ↓
Historical retrieval corpus


The extracted AmazonHelp subset contained:

269,273 AmazonHelp tweets
99,433 customer → brand response pairs
98,026 cleaned pairs
72,785 English rows
70,512 rows in the final retrieval corpus

The extraction was performed in chunks because the original dataset is large.

3. Intent Taxonomy

I defined eight intents after exploring the AmazonHelp conversations.

Intent	Description
order_delivery	Late, missing, tracking, or incorrect delivery
return_refund	Returns, refunds, replacements
account_login	Login, password, locked or hacked account
prime	Prime membership, benefits, delivery, Prime Video
payment_billing	Charges, payment failures, billing
product_technical	Product, device, application, or technical problems
customer_service	Explicit requests for an agent, contact, callback, or escalation
general_feedback	Thanks, praise, complaints, or feedback without an actionable support request

The classifier is intended to identify the primary support need, rather than simply matching individual keywords.

4. Golden Evaluation Set

I manually labelled 200 examples sampled from the cleaned AmazonHelp corpus.

The golden set contains:

200 manually labelled examples
8 intent classes

Distribution:

Intent	Count
order_delivery	63
customer_service	62
general_feedback	35
return_refund	21
product_technical	11
payment_billing	4
prime	2
account_login	2
Sampling note

The examples were sampled from the cleaned AmazonHelp support corpus and manually assigned one primary intent.

Because the golden examples were also drawn from the historical corpus used by the prototype, there is potential for near-duplicate leakage. Exact duplicate customer messages are excluded during retrieval evaluation, but this does not completely eliminate semantic overlap.

A stronger future evaluation would create the golden set from a held-out source split before building the retrieval corpus.

5. System Architecture
Incoming customer message
          |
          v
   Intent Classifier
          |
          v
   Intent + Confidence
          |
          v
 Intent-aware Retrieval
          |
          v
Historical customer/response pair
          |
          v
 Draft Response
          |
          v
 Escalation Policy
     /          \
    /            \
AUTO_HANDLE     ESCALATE
Intent classifier

The main classical classifier is:

TF-IDF
   ↓
Logistic Regression

This was chosen because it is:

fast;
interpretable;
easy to reproduce;
appropriate for a relatively small manually-labelled dataset.
6. Retrieval

Historical customer-support conversations are used as grounding evidence.

The retrieval pipeline uses:

Customer messages
       ↓
TF-IDF representation
       ↓
Cosine similarity
       ↓
Intent-aware filtering
       ↓
Best historical interaction

Generic and obviously unhelpful responses were filtered from the retrieval corpus.

The final response is based on the historically retrieved AmazonHelp response rather than generated from unsupported information.

7. Escalation Policy

The system escalates when there is insufficient confidence or evidence.

Current rules include:

Intent confidence < 0.30
        → ESCALATE

Historical similarity < 0.35
        → ESCALATE

account_login
        → ESCALATE

payment_billing
        → ESCALATE

product_technical AND similarity < 0.50
        → ESCALATE

Otherwise:

AUTO_HANDLE

Each escalation includes a reason such as:

low intent confidence
weak historical evidence
issue may require account-specific information
technical issue lacks strong historical evidence

These thresholds are deliberately conservative prototype heuristics, not production safety guarantees.

8. Evaluation
Intent Classification

Two baselines were evaluated.

Baseline 1 — Majority Classifier

Always predicts the most frequent intent.

Accuracy: 30.0%
Macro F1: 6.59%
Baseline 2 — TF-IDF + Logistic Regression

A simple supervised text classifier.

Using stratified 5-fold cross-validation:

Accuracy: 52.5%
Macro F1: 28.01%

The classifier improves substantially over the majority baseline.

However, the dataset is highly imbalanced. In particular, prime, account_login, and payment_billing have very few manually labelled examples, so performance on those classes is not reliable.

Retrieval Evaluation

A leakage-aware diagnostic was performed on the 200 golden examples.

Exact duplicate customer messages were excluded.

Results:

Golden examples:       200
Historical rows:    70,512
Exact duplicates removed: 186

Mean Top-1 similarity: 0.4887
Mean Top-3 similarity: 0.4303
Mean Top-5 similarity: 0.4016

These similarity values should be treated as retrieval diagnostics, not as direct measures of response usefulness.

Some near-duplicate examples remain possible.

End-to-End Example
Customer message

My package hasn't arrived yet and the delivery is late

Prediction
Intent: order_delivery
Confidence: 0.3448
Historical evidence

My package still hasn’t arrived 😥

Historical response

Delivery delays are possible but should be rare. Please let us know if your order doesn't arrive tomorrow!

Decision
AUTO_HANDLE
Draft

Thanks for reaching out. Delivery delays are possible but should be rare. Please let us know if your order doesn't arrive tomorrow!

This illustrates the intended grounding behavior: the response is based on a historically similar AmazonHelp interaction.

9. Response Quality Evaluation

A heuristic response-quality diagnostic was implemented over the 200 golden examples.

The current diagnostic produced:

Average relevance:              0.114
Average groundedness:           0.832
Average helpfulness:             0.248
Average evidence similarity:     0.806
Average overall diagnostic:      0.496

Decision distribution:

AUTO_HANDLE: 73
ESCALATE:    127
Important limitation

These are heuristic diagnostic scores, not an LLM-as-judge evaluation.

The lexical relevance metric is particularly limited because semantically related support responses can use very different wording.

I also prepared a 29-example human evaluation set, but human ratings were not completed before submission. Therefore, I do not claim human/LLM judge agreement.

10. Top Failure Modes
1. Retrieval returns an unrelated support response

Example:

Customer

Can you track where my parcel is actually held?

Retrieved response

I've also responded to your DM.

Hypothesis

TF-IDF similarity can select responses that share generic support vocabulary without matching the actual problem.

2. Technical issue receives a vague response

Example:

Customer

Spotify podcast technical issue

Response

What exactly is happening Lauren?

Hypothesis

The historical response may be appropriate conversationally but does not provide meaningful troubleshooting information.

3. Very short / incomplete messages

Example:

I’m in the USA so I assume,

Response

Which marketplace are you utilizing: or Amazon.it?

Hypothesis

Short messages provide insufficient context for reliable intent classification or retrieval.

4. Feedback is confused with actionable support

Example:

Can u plz tell me ur policies...

Response

I'll ensure to share your feedback internally to work on.

Hypothesis

Generic customer-service language creates lexical overlap between feedback and actual support requests.

5. Resolved conversations can retrieve unrelated responses

Example:

I contacted customer service team and got it resolved. Thanks.

Retrieved response

A Prime-related response.

Hypothesis

The message contains generic support language and little information about the original issue.

11. What Is Misleading About My Headline Number?

The headline classification accuracy of 52.5% should not be interpreted as "the support agent is 52.5% correct."

There are several reasons:

The golden set contains only 200 examples.
The intent distribution is highly imbalanced.
Some classes have only 2–4 examples.
Macro F1 is much lower than accuracy.
Retrieval quality and response quality are separate from intent accuracy.
The golden set was sampled from the same historical corpus used for retrieval, creating potential near-duplicate leakage.
The response-quality scores are heuristic diagnostics rather than human or LLM-judge scores.

Therefore, the headline number demonstrates that the classifier learns useful signal over the majority baseline, but it is not evidence of production-level support quality.

12. Key Lessons
1. Simple models are useful baselines

TF-IDF + Logistic Regression provides a strong, fast baseline without requiring an LLM.

2. Retrieval quality matters as much as generation

A response cannot be reliably grounded if the retrieved historical interaction is unrelated.

3. Confidence should influence automation

Low-confidence predictions should not automatically produce customer-facing responses.

4. Rare intents require more labelled data

The manually-labelled dataset is too small for reliable estimates on several rare classes.

5. Evaluation design matters

A seemingly excellent score can be misleading when the evaluation set overlaps with the retrieval corpus.

13. One-Week Next Steps

If given another week, I would prioritize:

Create a strict train/evaluation split before retrieval.
Expand the golden set to at least 500–1,000 examples.
Add more examples for rare intents.
Replace TF-IDF retrieval with sentence embeddings.
Add a proper LLM-as-judge evaluation.
Measure agreement between human ratings and the LLM judge.
Improve response generation instead of directly reusing historical responses.
Add explicit citation/evidence tracking for every generated response.
Calibrate automation thresholds using a larger validation set.
Add adversarial tests for ambiguous and incomplete customer messages.
14. Repository Structure
hiver/
│
├── data/
│   ├── golden_set.csv
│   ├── human_evaluation.csv
│   └── ...
│
├── data_exploration.py
├── conversation_analysis.py
├── extract_amazon.py
├── inspect_amazon.py
├── conversation_structure.py
├── build_pairs.py
├── clean_data.py
├── language_analysis.py
├── filter_english.py
├── intent_exploration.py
├── intent_examples.py
├── intent_validation.py
├── golden_set_sampling.py
├── golden_distribution.py
├── baseline.py
├── strong_classifier.py
├── retrieval.py
├── prepare_retrieval.py
├── label_retrieval_data.py
├── intent_retrieval.py
├── evaluate_retrieval.py
├── response_generator.py
├── agent.py
├── evaluate_agent.py
├── evaluate_intent.py
├── evaluate_response.py
├── create_human_eval.py
│
├── report/
│   └── REPORT.md
│
├── decision_log.md
├── requirements.txt
└── README.md
15. Reproduction
Requirements

Python 3.11+ is recommended.

Create a virtual environment:

python -m venv venv

Activate it on Windows:

.\venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
Run the agent
python agent.py

This runs an example customer query through:

Intent classification
        ↓
Historical retrieval
        ↓
Response drafting
        ↓
Escalation decision
Run intent evaluation
python evaluate_intent.py

This performs stratified cross-validation on the golden set.

Expected headline result:

Accuracy ≈ 52.5%
Macro F1 ≈ 28.0%

Small differences may occur depending on package versions/random seeds.

Run retrieval evaluation
python evaluate_retrieval.py

Expected diagnostic:

Mean Top-1 similarity ≈ 0.49

Exact duplicate customer messages are excluded during this evaluation.

Run end-to-end evaluation
python evaluate_agent.py

This evaluates the agent's intent predictions and escalation behavior using out-of-fold predictions rather than training and evaluating on the same examples.

16. Reproducibility Note

The original Twitter dataset is very large, so the complete raw dataset is not required for the headline evaluation.

The evaluation artifacts are based on the selected AmazonHelp subset and manually-labelled golden set.

The full raw-data preprocessing pipeline is included separately for transparency.

For a fresh reproduction from the original dataset:

Raw dataset
    ↓
extract_amazon.py
    ↓
build_pairs.py
    ↓
clean_data.py
    ↓
filter_english.py
    ↓
prepare_retrieval.py
    ↓
label_retrieval_data.py

The full raw dataset processing can take considerably longer than the headline evaluation because the original dataset contains approximately 3 million tweets.

17. Conclusion

This project demonstrates a lightweight AI customer-support pipeline combining:

supervised intent classification;
historical-response retrieval;
grounded response drafting;
confidence-aware escalation;
baseline comparison;
leakage-aware retrieval diagnostics;
explicit failure analysis.

The main finding is that a simple TF-IDF + Logistic Regression model can learn meaningful support-intent signal over a majority baseline, but reliable customer-facing automation requires substantially better evaluation data, retrieval quality, and response-quality measurement.