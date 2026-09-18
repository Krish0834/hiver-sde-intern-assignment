# Decision Log

## 1. Selected AmazonHelp as the target brand

**Decision:** Build the support agent for AmazonHelp.

**Why:** AmazonHelp had a large number of customer-support interactions in the dataset, providing enough historical examples for both intent classification and retrieval.

---

## 2. Used customer → brand response pairs

**Decision:** Use immediate customer-to-brand response pairs as the initial training/retrieval unit.

**Why:** This provided a simple, reproducible mapping between a customer problem and the historical brand response.

**Trade-off:** This loses information from earlier turns in multi-turn conversations.

---

## 3. Restricted the initial system to English

**Decision:** Use English-language customer messages for the main prototype.

**Why:** Language detection showed that English represented the largest portion of the extracted AmazonHelp interactions and allowed the initial system to focus on one language.

**Trade-off:** Non-English conversations are excluded.

---

## 4. Defined eight support intents

**Decision:** Use eight intents:
`order_delivery`, `return_refund`, `account_login`, `prime`,
`payment_billing`, `product_technical`, `customer_service`,
and `general_feedback`.

**Why:** The taxonomy was derived from recurring support needs observed in the AmazonHelp data.

**Trade-off:** A small taxonomy improves simplicity but merges some more specific customer problems.

---

## 5. Created a 200-example hand-labelled golden set

**Decision:** Manually label 200 customer messages.

**Why:** The assignment requires 150–250 hand-labelled examples. 200 provides enough examples for evaluation while remaining feasible to label manually.

**Trade-off:** Several rare intents still have very few examples.

---

## 6. Used a majority-class classifier as the trivial baseline

**Decision:** Always predict the most frequent intent.

**Why:** This establishes the minimum baseline that a useful classifier should outperform.

---

## 7. Used TF-IDF + Logistic Regression as the simple baseline

**Decision:** Use TF-IDF features with Logistic Regression.

**Why:** It is fast, interpretable, reproducible, and provides a strong classical text-classification baseline without requiring a large language model.

---

## 8. Added class weighting

**Decision:** Use `class_weight="balanced"` in Logistic Regression.

**Why:** The golden set is highly imbalanced, with `order_delivery` and `customer_service` much more common than several other intents.

**Trade-off:** Class weighting can improve minority-class sensitivity but does not solve the problem of extremely small classes.

---

## 9. Used out-of-fold evaluation for intent classification

**Decision:** Evaluate the classifier using stratified 5-fold out-of-fold predictions rather than evaluating on the same examples used for training.

**Why:** Evaluating on training examples would produce an overly optimistic estimate of performance.

**Trade-off:** Very rare intents with fewer than five examples make 5-fold estimates unstable.

---

## 10. Added intent-aware retrieval

**Decision:** First predict the intent and then retrieve historical examples from the predicted intent.

**Why:** Restricting retrieval to the predicted support category reduces obviously unrelated historical responses.

**Trade-off:** An incorrect intent prediction can prevent the correct historical examples from being considered.

---

## 11. Filtered generic historical responses

**Decision:** Remove some historical responses that were purely generic acknowledgements such as simple thanks or generic support greetings.

**Why:** Such responses provide little useful grounding for drafting a support response.

**Trade-off:** Conservative filtering can also remove some responses that might be useful in specific contexts.

---

## 12. Used retrieval similarity as an escalation signal

**Decision:** Escalate when the best historical match has weak similarity.

**Why:** A system should avoid automatically generating responses when it has insufficient historical evidence.

---

## 13. Escalated account and payment issues

**Decision:** Automatically escalate `account_login` and `payment_billing` issues.

**Why:** These cases may require account-specific or transaction-specific information that is unavailable to the prototype.

**Trade-off:** This reduces the potential automation rate but provides a safer initial policy.

---

## 14. Required stronger evidence for technical issues

**Decision:** Use a higher retrieval threshold for `product_technical`.

**Why:** Technical problems are often more specific and can be poorly served by a loosely similar historical response.

---

## 15. Avoided a large local LLM

**Decision:** Use a retrieval-grounded response composer rather than installing and running a large local language model.

**Why:** The development environment is CPU-only with approximately 8 GB of RAM. A large local model would add significant setup and runtime cost and would not improve the reproducibility of the core experiment under the assignment's time constraint.

**Trade-off:** The current response generation is less flexible than a dedicated generative model and can reproduce weaknesses present in historical responses.