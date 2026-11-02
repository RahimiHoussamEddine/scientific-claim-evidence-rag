# Methodology and evaluation protocol

## Task

Given a short scientific claim, retrieve abstracts from the SciFact corpus and generate a concise verdict conditioned only on those retrieved abstracts. The user interface always displays retrieved source IDs and abstracts next to the model response.

## Data splits

The pinned source archive contains 809 labeled training claims, 300 labeled development claims, 300 unlabeled test claims, and 5,183 corpus abstracts. The corpus is the retrieval knowledge base. Training claims are not used to train the generator or dense encoder. The labeled development split is reported as validation evidence; the test split is never scored because its labels are not released in the archive.

Claim labels are derived from evidence annotations: SUPPORT-only becomes SUPPORTS; CONTRADICT-only becomes REFUTES; an empty evidence map becomes NOT_ENOUGH_EVIDENCE; and conflicting annotation types become MIXED. Evidence document IDs and rationale sentence IDs are evaluated separately.

## Retrieval systems

- BM25 over lowercased alphanumeric tokens is the sparse lexical baseline.
- Dense retrieval uses a pinned all-MiniLM-L6-v2 encoder, CPU inference, cosine similarity, and normalized embeddings.
- Hybrid retrieval uses reciprocal-rank fusion with a fixed constant of 60. This fusion rule is specified before development-split evaluation rather than selected by the hidden test.

For annotated claims, report document-level Recall@5, Precision@5, MRR, and nDCG@5. Claims without annotated evidence are counted separately and excluded from positive evidence-retrieval denominators. Rationale sentence IDs are retained for error inspection but are not treated as independent gold documents.

## Local generation

FLAN-T5-small receives the claim and up to the first three retrieved abstracts, with up to three sentences per abstract. Sampling is disabled. Its fixed prompt requests one of SUPPORTS, REFUTES, MIXED, or NOT_ENOUGH_EVIDENCE. The model is not fine-tuned, and generated output is not rewritten to force agreement with dataset labels. The interface displays the retrieved document IDs independently of the model's output.

Report verdict accuracy, macro-F1 over classes observed in the dev gold labels, unparseable fraction, and citation checks for any bracketed document IDs the model emits despite not being asked to cite. Citation-source validity is not factuality. Citation gold precision is conditional on citations the model emitted; zero citations are reported as zero. No LLM-as-judge score is used. The measured generator results are poor and are presented as a negative finding, not as a reliable verification capability.

## Model selection and leakage controls

No model weights are trained in this project. Retrieval variants and the fixed RRF configuration are predeclared; the development split is reported as validation evidence. The hidden test set is unlabeled and never used for tuning. The original dataset claim IDs and evidence annotations are not included in document text or query features. The generator sees only each user claim and retrieved abstract text.

## Limits

SciFact is small and based on evidence-containing abstracts. Some claims may require full papers, study quality assessment, or a systematic review. Publication-level abstracts can omit caveats. The benchmark's non-commercial license applies to its source content; raw data are excluded from Git. Model outputs may hallucinate or overstate evidence and require expert review.
