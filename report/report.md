 AI Customer Support Agent

1. Problem Framing

The goal of this project is to turn noisy real-world customer-support conversations from Twitter into a support agent that can:

1. classify incoming customer messages into a small set of support intents,
2. draft a response grounded in how the brand historically handled similar issues, and
3. decide whether the message should be automatically handled or escalated to a human.

The primary dataset is the Customer Support on Twitter dataset, containing approximately 3M tweets and multi-turn customer-support conversations.

## Brand Selection

I selected **AmazonHelp** because it had a large number of customer-support interactions in the dataset, providing enough historical examples for both intent learning and retrieval.

The extracted AmazonHelp subset contained approximately:

- 269K AmazonHelp/customer tweets
- 99K customer → AmazonHelp response pairs
- 72.8K English customer messages after language filtering

### Definition of "Good"

A good support agent should:

- identify the customer's primary support need,
- retrieve evidence from similar historical Amazon interactions,
- avoid inventing unsupported resolutions,
- provide a useful and professional draft,
- escalate cases where confidence or evidence is insufficient.

### What I Did Not Build

This prototype intentionally does not attempt to:

- execute real Amazon account actions,
- access customer account information,
- issue refunds or replacements,
- reconstruct the complete multi-turn conversation,
- operate as a production customer-service system,
- use a large local LLM.

The current system focuses on classification, historical retrieval, grounded response drafting, and escalation.

---

# 2. Data Preparation

The raw Twitter dataset is noisy and contains inbound customer messages and outbound brand responses.

The preprocessing pipeline was:

```text
Raw Twitter dataset
        ↓
AmazonHelp filtering
        ↓
Customer → Brand response pairs
        ↓
Text cleaning
        ↓
English-language filtering
        ↓
Intent exploration
        ↓
Golden evaluation set
        ↓
Classifier + retrieval system


The customer → brand pairing uses the dataset's response relationships to create immediate support examples.

Text cleaning removes URLs, mentions, HTML artifacts, duplicate pairs, and other noise.

Only English examples were retained for the main prototype.



# 3. Intent Taxonomy

Eight support intents were defined from observed Amazon customer-support patterns.

Intent	Description
order_delivery	Late, missing, tracking, or delivery-status problems
return_refund	Returns, refunds, replacements
account_login	Login, password, locked or hacked accounts
prime	Prime membership and Prime-related issues
payment_billing	Charges, payment, billing, gift-card payment issues
product_technical	Product, device, application, or technical problems
customer_service	Requests for agents, callbacks, escalation, or support interaction problems
general_feedback	Thanks, praise, feedback, or non-actionable complaints

The classification rule is based on the customer's primary support need rather than simply matching keywords.

##4. Golden Evaluation Set

A hand-labelled evaluation set of 200 examples was created.

The set was sampled from the Amazon customer-support data and manually assigned one of the eight intents.

Distribution
Intent	Examples
order_delivery	63
customer_service	62
general_feedback	35
return_refund	21
product_technical	11
payment_billing	4
prime	2
account_login	2
Total	200

The class distribution reflects the observed support-message distribution rather than artificially balancing the evaluation set.

The golden set is stored in:

data/golden_set.csv




5. System Architecture

The agent consists of four main stages.

Customer Message
       │
       ▼
TF-IDF + Logistic Regression
       │
       ▼
Intent + Confidence
       │
       ▼
Intent-aware Historical Retrieval
       │
       ▼
Historical Amazon Response
       │
       ▼
Escalation Policy
       │
       ├───────────────┐
       ▼               ▼
 AUTO_HANDLE       ESCALATE
       │
       ▼
Grounded Draft Response
Intent Classification

The classifier uses:

TF-IDF features
unigram + bigram features
Logistic Regression
class weighting
Historical Retrieval

Historical customer messages are represented using TF-IDF.

The retrieval process first filters historical examples using the predicted intent and then calculates cosine similarity to find the most similar historical support case.

Response Drafting

The current prototype uses the retrieved historical brand response as the grounding source.

The system removes mentions, URLs, and agent signatures before presenting the response.

This is intentionally a retrieval-grounded response composer rather than an unconstrained generative model.

6. Escalation Policy

The agent considers both classifier confidence and historical evidence.

A message is escalated when:

intent confidence is below 0.30,
historical similarity is below 0.35,
the intent is account_login,
the intent is payment_billing, or
a technical issue has historical similarity below 0.50.

Otherwise, the system returns AUTO_HANDLE.

The purpose of this policy is to avoid automatically responding when either the predicted intent or historical evidence is weak.

7. Results
7.1 Baseline 1 — Majority Classifier

The majority-class baseline always predicts the most common intent.

Metric	Result
Accuracy	30.0%
Macro F1	6.59%

This provides a trivial lower baseline.

7.2 Baseline 2 — TF-IDF + Logistic Regression

The main classical classifier was evaluated using 5-fold stratified out-of-fold evaluation.

Metric	Result
Accuracy	52.5%
Macro F1	28.01%

The improvement over the majority baseline demonstrates that the text features contain useful information for distinguishing support intents.

However, rare intents remain difficult to estimate reliably because several classes have only 2–4 examples in the 200-example evaluation set.

8. Retrieval Evaluation

A TF-IDF historical retrieval system was evaluated against the golden examples.

Exact normalized duplicates were excluded from the evaluation to reduce direct leakage.

Results:

Metric	Result
Mean Top-1 similarity	0.489
Mean Top-3 similarity	0.430
Mean Top-5 similarity	0.402

These are retrieval-similarity diagnostics rather than end-to-end response-quality metrics.

The evaluation revealed that lexical similarity can retrieve highly similar customer messages while still returning a weak or context-dependent historical response.

9. End-to-End Example
Customer

My package hasn't arrived yet and the delivery is late

Predicted Intent

order_delivery

Intent Confidence

0.3448

Retrieval Similarity

0.6922

Decision

AUTO_HANDLE

Historical Evidence

My package still hasn’t arrived 😥

Historical Response

Delivery delays are possible but should be rare. Please let us know if your order doesn't arrive tomorrow!

Draft Response

Thanks for reaching out. Delivery delays are possible but should be rare. Please let us know if your order doesn't arrive tomorrow!

This demonstrates the intended behavior: the system identifies the delivery intent, retrieves a highly similar historical case, and grounds its response in the historical Amazon response.

10. Response-Quality Evaluation

An automated response-quality diagnostic was implemented using heuristic measures for:

relevance,
groundedness,
helpfulness,
similarity to the historical evidence.

The current diagnostic produced:

Metric	Result
Relevance	0.114
Groundedness	0.832
Helpfulness	0.248
Evidence similarity	0.806
Overall heuristic score	0.496

These scores are diagnostic only and are not presented as an LLM-as-judge score.

The low lexical relevance score demonstrates a limitation of simple word-overlap metrics: semantically relevant responses can use different vocabulary from the customer's message.

The relatively high groundedness/evidence-similarity measurements are consistent with the design of the system, which retrieves responses directly from historical Amazon interactions.

A 29-example human evaluation subset was also prepared for manual assessment.

A full LLM-as-judge versus human agreement study was not completed in the current implementation and is therefore treated as a limitation rather than reported as a completed result.

11. Failure Analysis
Failure 1 — Weak Historical Response
Customer

Can you track where my parcel is actually held?

Retrieved Response

I've also responded to your DM.

Hypothesis

The retrieval system can find a historically similar interaction whose response is only an operational acknowledgement rather than an actual resolution.

This shows that high retrieval similarity does not necessarily imply a useful response.

Failure 2 — Technical Issue Gets a Vague Response
Customer

It’s not recognizing the name of the podcast (Slate Presents Lexicon Valley) which can be found on Spotify.

Response

What exactly is happening Lauren?

Hypothesis

Historical customer-support data contains many short clarification responses. Retrieving such a response can produce a grounded but unhelpful draft.

Failure 3 — Loss of Multi-turn Context
Customer

I’m in the USA so I assume,

Response

Which marketplace are you utilizing: or Amazon.it?

Hypothesis

The current prototype primarily uses immediate customer → brand response pairs. A context-dependent message may be impossible to interpret correctly without earlier turns in the conversation.

Failure 4 — Ambiguous Short Messages
Customer

#LGQ6AppQuiz still not yet listed in your link.

Predicted Intent

general_feedback

Hypothesis

Short messages often contain insufficient information for reliable intent classification, particularly when the necessary context exists in earlier conversation turns.

Failure 5 — Information Request Misinterpreted as Feedback
Customer

Can u plz tell me ur policies for customers if customers is having problem continosly.

Response

I'll ensure to share your feedback internally to work on.

Hypothesis

Lexical retrieval can confuse a request for policy information with a complaint or feedback message. The system needs stronger semantic understanding of the customer's actual requested action.

12. What Is Misleading About My Headline Number?

The 52.5% intent accuracy should not be interpreted as production-level support-agent performance.

First, the golden evaluation set contains only 200 examples and several intents contain very few examples. Therefore, rare-class metrics are statistically unstable.

Second, retrieval evaluation is sensitive to duplicate and near-duplicate historical conversations. Exact normalized duplicates were excluded from the reported retrieval diagnostic, but near-duplicate leakage can still exist.

Third, response-quality measurements currently use heuristic automated metrics. They should not be interpreted as equivalent to human evaluation or an LLM judge.

Finally, intent accuracy alone does not measure whether the final response is useful. A correctly classified message can still retrieve a poor historical response, as demonstrated by the failure analysis.

Therefore, the headline classification result is useful for comparing the prototype with baselines, but it is not sufficient evidence for production deployment.

13. Key Lessons

The project showed that historical customer-support data can provide useful grounding for an AI support system, but three problems dominate the current limitations:

Context loss — individual tweets frequently depend on earlier conversation turns.
Weak historical responses — some brand responses are acknowledgements or clarification requests rather than resolutions.
Intent ambiguity — short or noisy messages are difficult to classify reliably.

The escalation mechanism is therefore important: the system should not treat retrieval alone as evidence that an interaction is safe to automate.

14. What I Would Do With One More Week
1. Reconstruct complete conversations

Instead of using only immediate customer → brand pairs, reconstruct multi-turn threads using the Twitter response IDs.

2. Improve retrieval

Replace TF-IDF-only retrieval with semantic embeddings and combine semantic similarity with lexical similarity.

3. Improve response generation

Use retrieved historical cases as grounding context for a controlled response-generation model rather than copying a single historical response.

4. Improve evaluation

Expand the golden set for rare intents and construct a stronger held-out retrieval evaluation split.

5. Add LLM-as-judge evaluation

Use a structured rubric covering relevance, groundedness, helpfulness, and appropriateness, and measure agreement against human ratings.

6. Calibrate escalation

Tune escalation thresholds using a larger validation set and optimize for safe automation rather than maximizing raw auto-handling rate.

15. Conclusion

The final prototype demonstrates an end-to-end customer-support agent for AmazonHelp:

data extraction and cleaning,
intent taxonomy creation,
intent classification,
historical evidence retrieval,
grounded response drafting,
automated escalation decisions,
baseline comparison,
evaluation diagnostics,
and real failure analysis.

The main conclusion is not that the system is production-ready. Instead, the experiments identify where a retrieval-grounded support agent works and where historical conversational context and response quality remain limiting factors.