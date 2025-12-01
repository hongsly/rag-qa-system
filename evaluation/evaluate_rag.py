import json

from datasets import Dataset
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (answer_correctness, answer_relevancy,
                           context_precision, context_recall, faithfulness)
from src.rag_pipeline import RagAssistant
from src.utils import EVAL_DATA_DIR, EVAL_OUTPUT_DIR, get_openai_api_key
from tqdm import tqdm

RAGAS_TESTSET_PATH = EVAL_DATA_DIR / "ragas_testset.jsonl"
MANUAL_TESTSET_PATH = EVAL_DATA_DIR / "test_questions.json"


def _generate_responses(
    questions: list[str], assistant: RagAssistant, retrieval_mode: str, rag_model: str
):
    response_list = []
    retrieved_contexts_list = []
    for question in tqdm(questions, desc="Generating responses"):
        answer, context = assistant.query(
            question, model=rag_model, retrieval_mode=retrieval_mode
        )
        response_list.append(answer)
        if context:
            retrieved_contexts_list.append([c["chunk_text"] for c in context])
        else:
            retrieved_contexts_list.append([])

    response_dict = {
        "user_input": questions,
        "response": response_list,
    }
    if retrieval_mode != "none":
        response_dict["retrieved_contexts"] = retrieved_contexts_list
    return response_dict


def _evaluate(
    assistant: RagAssistant,
    ragas_testset: list[dict],
    manual_testset: list[dict],
    mode: str,
    eval_llm: LangchainLLMWrapper,
    rag_model: str = "gpt-4o-mini",
):
    questions = [item["question"] for item in manual_testset] + [
        item["user_input"] for item in ragas_testset
    ]
    dataset_dict = _generate_responses(
        questions, assistant, retrieval_mode=mode, rag_model=rag_model
    )
    dataset_dict["reference"] = [item["reference"] for item in manual_testset] + [
        item["reference"] for item in ragas_testset
    ]
    dataset = Dataset.from_dict(dataset_dict)
    dataset.to_json(str(EVAL_OUTPUT_DIR / f"response_dataset_{mode}.jsonl"))
    print(
        f"Response dataset written to {EVAL_OUTPUT_DIR / f'response_dataset_{mode}.jsonl'}"
    )

    # answer_relevancy need: user_input, response
    # faithfulness need: user_input, response, retrieved_contexts
    # answer_correctness need: user_input, response, reference
    # context_precision, context_recall need: user_input, retrieved_contexts, reference
    if mode != "none":
        metrics = [
            answer_correctness,
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ]
    else:
        metrics = [answer_correctness, answer_relevancy]
    print("Evaluating...")
    results = evaluate(dataset, metrics, llm=eval_llm)
    results.to_pandas().to_json(str(EVAL_OUTPUT_DIR / f"eval_results_{mode}.json"))
    print(results)


def _get_ollama_llm(model: str = "qwen2.5-coder:7b"):
    ollama_llm = LangchainLLMWrapper(
        ChatOllama(model=model, reasoning=False, temperature=0.0, num_ctx=8192)
    )
    return ollama_llm


def _get_openai_llm(model: str = "gpt-4o-mini"):
    get_openai_api_key()
    openai_llm = LangchainLLMWrapper(ChatOpenAI(model=model))
    return openai_llm


def evaluate_rag():
    with open(RAGAS_TESTSET_PATH, "r") as f:
        ragas_testset = [json.loads(line) for line in f]
    with open(MANUAL_TESTSET_PATH, "r") as f:
        manual_testset = json.load(f)
    print("Loaded ragas_testset and manual_testset")

    eval_llm = _get_openai_llm()
    rag_model = "gpt-4o-mini"

    # eval_llm = _get_ollama_llm()
    # rag_model = "qwen2.5-coder:7b"

    assistant = RagAssistant()
    modes = ["hybrid", "dense", "sparse", "none"]
    for mode in modes:
        print(f"\n{'='*80}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*80}\n")
        _evaluate(
            assistant,
            ragas_testset,
            manual_testset,
            mode,
            eval_llm,
            rag_model=rag_model,
        )


if __name__ == "__main__":
    evaluate_rag()
