"""
Prompt templates for V-CUE framework.
All prompts are derived from the paper appendices.
"""

# ============================================================
# Appendix A: Uncertainty Detection (UD) Prompts
# ============================================================

UD_SYSTEM_PROMPT = (
    "You are an expert evaluator for cultural understanding tasks. "
    "Your job is to assess whether the given answer is culturally "
    "reliable and confident."
)

UD_USER_PROMPT = (
    "[Query]: {question}\n"
    "[Answer]: {answer}\n\n"
    "Please evaluate the answer according to the following criteria:\n"
    "1. Cultural correctness (Is the answer aligned with the cultural "
    "background implied by the query?)\n"
    "2. Cultural specificity (Does the answer rely on concrete cultural "
    "knowledge rather than generic statements?)\n"
    "3. Confidence and consistency (Is the answer internally consistent "
    "and decisive, without hedging or uncertainty?)\n\n"
    "Return only one label: True or False. Do not explain your decision."
)

# ============================================================
# Appendix B: Cultural Cue Extraction Prompts
# ============================================================

CUE_EXTRACTION_SYSTEM_PROMPT = (
    "You are a cultural information parser. Your task is to extract "
    "culturally relevant elements from the input question according "
    "to a predefined schema.\n\n"
    "Rules:\n"
    "- Only use information explicitly contained in the question text.\n"
    "- Do not infer or predict the correct answer.\n"
    "- If an element is not mentioned or cannot be inferred, output None.\n\n"
    'Output Format (JSON only):\n'
    '{\n'
    '  "region": "...",\n'
    '  "object": "...",\n'
    '  "symbol": "..."\n'
    '}'
)

CUE_EXTRACTION_USER_PROMPT = "[Question]: {question}"

# ============================================================
# Appendix C: Neutral Visuals Construction Prompts
# ============================================================

NEUTRAL_CUE_SYSTEM_PROMPT = (
    "You are a cultural cue rewriter. Your task is to neutralize "
    "culturally specific visual elements by removing identifiable "
    "cultural meaning while preserving generic visual content.\n\n"
    "Rules:\n"
    "- Rewrite each element to remove culture-specific meaning.\n"
    "- Replace culturally identifiable terms with culturally neutral, "
    "generic alternatives.\n"
    "- Do not add new information or attributes.\n"
    "- If an element is already culturally neutral, keep it unchanged.\n\n"
    'Output Format (JSON only):\n'
    '{\n'
    '  "region": "...",\n'
    '  "object": "...",\n'
    '  "symbol": "..."\n'
    '}'
)

NEUTRAL_CUE_USER_PROMPT = "[Input Elements]: {region}, {object}, {symbol}"

# ============================================================
# Appendix D: MAPS Baseline Prompts
# ============================================================

MAPS_KEYWORD_PROMPT = "Extract the keywords in the following sentence:\n{text}"

MAPS_TOPIC_PROMPT = (
    "Use a few words to describe the cultural topics of the "
    "following input sentence:\n{text}"
)

MAPS_DEMO_PROMPT = (
    "Write a sentence related to but different from the input "
    "cultural query and answer:\n{text}"
)

MAPS_FINAL_PROMPT = (
    "Given the following cultural knowledge context:\n"
    "Keywords: {keywords}\n"
    "Topics: {topics}\n"
    "Related example: {demo}\n\n"
    "Now answer the following question:\n{question}"
)

# ============================================================
# Appendix D: Self-Correct Baseline Prompts
# ============================================================

SELF_CORRECT_ACTOR_PROMPT = (
    "You are tasked with answering cultural knowledge questions. "
    "Use the following feedback from previous attempts to improve "
    "your answer.\n\n"
    "Previous feedback:\n{feedback}\n\n"
    "Question: {question}\n"
    "Please provide your answer."
)

SELF_CORRECT_EVALUATOR_PROMPT = (
    "Evaluate the following answer to a cultural question. "
    "Score it from 1-10 on cultural accuracy.\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "Return only the score as an integer."
)

SELF_CORRECT_REFLECTION_PROMPT = (
    "The following answer was given to a cultural question and "
    "received a low score. Identify what went wrong and provide "
    "guidance for improvement.\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Score: {score}\n\n"
    "Provide a brief reflection on what cultural aspects were "
    "missed or incorrect."
)

# ============================================================
# Visual-based Reconstruction (VR) Prompts
# ============================================================

VR_TEXT_SYSTEM_PROMPT = (
    "You are an expert in cultural understanding. You are given a "
    "cultural question along with a description of a culturally "
    "relevant image. Use both the textual context and the visual "
    "description to answer the question accurately."
)

VR_TEXT_USER_PROMPT = (
    "Question: {question}\n\n"
    "Visual Description: {caption}\n\n"
    "Based on the question and the visual cultural cues described "
    "above, please provide your answer."
)

VR_VL_SYSTEM_PROMPT = (
    "You are an expert in cultural understanding. You are given a "
    "cultural question along with a culturally relevant image. "
    "Use both the textual context and the visual information to "
    "answer the question accurately."
)

VR_VL_USER_PROMPT = (
    "Question: {question}\n\n"
    "An image with cultural cues relevant to this question is "
    "provided. Based on the question and the visual cultural cues, "
    "please provide your answer."
)

# ============================================================
# Image Captioning Prompt
# ============================================================

CAPTION_SYSTEM_PROMPT = (
    "You are an image description expert. Describe the cultural "
    "elements visible in this image in detail, including clothing, "
    "decorations, symbols, architecture, and any culturally "
    "significant items."
)

CAPTION_USER_PROMPT = (
    "Please describe the cultural elements in this image in detail."
)

# ============================================================
# CARE Evaluation Judge Prompts
# ============================================================

CARE_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for cultural text generation. "
    "You will evaluate the given response on three dimensions:\n"
    "1. Generality (1-10): How well does the response cover general "
    "knowledge and context?\n"
    "2. Cultural Relevance (1-10): How well does the response reflect "
    "the specific cultural context?\n"
    "3. Literary Quality (1-10): How well-written and coherent is "
    "the response?\n\n"
    "Return your evaluation as JSON:\n"
    '{{"generality": <score>, "cultural_relevance": <score>, '
    '"literary_quality": <score>}}'
)

CARE_JUDGE_USER_PROMPT = (
    "Question: {question}\n"
    "Response: {response}\n\n"
    "Please evaluate the response on generality, cultural relevance, "
    "and literary quality."
)

# ============================================================
# CulturalBench Task Formatting Prompts
# ============================================================

CULTURALBENCH_MC_PROMPT = (
    "Answer the following cultural knowledge question by selecting "
    "the correct option(s). Return ONLY the option letter(s).\n\n"
    "{question}\n\n"
    "Options:\n{options}"
)

CULTURALBENCH_TF_PROMPT = (
    "Determine whether the following cultural statement is True or "
    "False. Return ONLY 'True' or 'False'.\n\n"
    "{question}"
)
