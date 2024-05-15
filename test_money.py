# from annotated_text import annotated_text
# from streamlit_tags import st_tags
# import logging
# import os
# import traceback
# from openai_fake_data_generator import OpenAIParams
# from presidio_analyzer import Pattern, PatternRecognizer
# from presidio_analyzer.recognizer_registry import RecognizerRegistry
# from presidio_analyzer.analyzer_engine import AnalyzerEngine
# from presidio_analyzer.predefined_recognizers import predefined_recognizers
# import dotenv
# import pandas as pd
# import streamlit as st
# import streamlit.components.v1 as components
# from PIL import Image
# import textract
# from pdf2image import convert_from_path
# import fitz
# import re
# import uuid  # Import uuid library to generate unique keys

# from openai_fake_data_generator import OpenAIParams
# from presidio_helpers import (
#     get_supported_entities,
#     analyze,
#     anonymize,
#     annotate,
#     create_fake_data,
#     analyzer_engine,
# )

# st.set_page_config(
#     page_title="RT SECURE",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# dotenv.load_dotenv()
# logger = logging.getLogger("presidio-streamlit")

# allow_other_models = os.getenv("ALLOW_OTHER_MODELS", False)

# model_help_text = """
#     Select which Named Entity Recognition (NER) model to use for PII detection, in parallel to rule-based recognizers.
#     Presidio supports multiple NER packages off-the-shelf, such as spaCy, Huggingface, Stanza and Flair,
#     as well as service such as Azure Text Analytics PII.
#     """
# st_ta_key = st_ta_endpoint = ""

# model_list = [
#     "spaCy/en_core_web_lg",
#     "flair/ner-english-large",
#     "HuggingFace/obi/deid_roberta_i2b2",
#     "HuggingFace/StanfordAIMI/stanford-deidentifier-base",
#     "stanza/en",
#     "Azure AI Language",
#     "Other",
# ]
# if not allow_other_models:
#     model_list.pop()

# # Select model
# st_model = st.sidebar.selectbox(
#     "NER model package",
#     model_list,
#     index=2,
#     help=model_help_text,
# )

# # Extract model package.
# st_model_package = st_model.split("/")[0]

# # Remove package prefix (if needed)
# st_model = (
#     st_model
#     if st_model_package.lower() not in ("spacy", "stanza", "huggingface")
#     else "/".join(st_model.split("/")[1:])
# )

# if st_model == "Other":
#     st_model_package = st.sidebar.selectbox(
#         "NER model OSS package", options=["spaCy", "stanza", "Flair", "HuggingFace"]
#     )
#     st_model = st.sidebar.text_input(f"NER model name", value="")

# if st_model == "Azure AI Language":
#     st_ta_key = st.sidebar.text_input(
#         f"Azure AI Language key", value=os.getenv("TA_KEY", ""), type="password"
#     )
#     st_ta_endpoint = st.sidebar.text_input(
#         f"Azure AI Language endpoint",
#         value=os.getenv("TA_ENDPOINT", default="")
#     )

# st.sidebar.warning("Note: Models might take some time to download. ")

# analyzer_params = (st_model_package, st_model, st_ta_key, st_ta_endpoint)
# logger.debug(f"analyzer_params: {analyzer_params}")

# st_operator = st.sidebar.selectbox(
#     "De-identification approach",
#     ["redact", "replace", "synthesize", "highlight", "mask", "hash", "encrypt"],
#     index=1,
#     help="""
#     Select which manipulation to the text is requested after PII has been identified.\n
#     - Redact: Completely remove the PII text\n
#     - Replace: Replace the PII text with a constant, e.g. <PERSON>\n
#     - Synthesize: Replace with fake values (requires an OpenAI key)\n
#     - Highlight: Shows the original text with PII highlighted in colors\n
#     - Mask: Replaces a requested number of characters with an asterisk (or other mask character)\n
#     - Hash: Replaces with the hash of the PII string\n
#     - Encrypt: Replaces with an AES encryption of the PII string, allowing the process to be reversed
#          """,
# )
# st_mask_char = "*"
# st_number_of_chars = 15
# st_encrypt_key = "WmZq4t7w!z%C&F)J"

# open_ai_params = None

# logger.debug(f"st_operator: {st_operator}")

# def set_up_openai_synthesis():
#     """Set up the OpenAI API key and model for text synthesis."""

#     if os.getenv("OPENAI_TYPE", default="openai") == "Azure":
#         openai_api_type = "azure"
#         st_openai_api_base = st.sidebar.text_input(
#             "Azure OpenAI base URL",
#             value=os.getenv("AZURE_OPENAI_ENDPOINT", default=""),
#         )
#         openai_key = os.getenv("AZURE_OPENAI_KEY", default="")
#         st_deployment_id = st.sidebar.text_input(
#             "Deployment name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT", default="")
#         )
#         st_openai_version = st.sidebar.text_input(
#             "OpenAI version",
#             value=os.getenv("OPENAI_API_VERSION", default="2023-05-15"),
#         )
#     else:
#         openai_api_type = "openai"
#         st_openai_version = st_openai_api_base = None
#         st_deployment_id = ""
#         openai_key = os.getenv("OPENAI_KEY", default="")
#     st_openai_key = st.sidebar.text_input(
#         "OPENAI_KEY",
#         value=openai_key,
#         help="See https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key for more info.",
#         type="password",
#     )
#     st_openai_model = st.sidebar.text_input(
#         "OpenAI model for text synthesis",
#         value=os.getenv("OPENAI_MODEL", default="gpt-3.5-turbo-instruct"),
#         help="See more here: https://platform.openai.com/docs/models/",
#     )
#     return (
#         openai_api_type,
#         st_openai_api_base,
#         st_deployment_id,
#         st_openai_version,
#         st_openai_key,
#         st_openai_model,
#     )

# if st_operator == "mask":
#     st_number_of_chars = st.sidebar.number_input(
#         "number of chars", value=st_number_of_chars, min_value=0, max_value=100
#     )
#     st_mask_char = st.sidebar.text_input(
#         "Mask character", value=st_mask_char, max_chars=1
#     )
# elif st_operator == "encrypt":
#     st_encrypt_key = st.sidebar.text_input("AES key", value=st_encrypt_key)
# elif st_operator == "synthesize":
#     (
#         openai_api_type,
#         st_openai_api_base,
#         st_deployment_id,
#         st_openai_version,
#         st_openai_key,
#         st_openai_model,
#     ) = set_up_openai_synthesis()

#     open_ai_params = OpenAIParams(
#         openai_key=st_openai_key,
#         model=st_openai_model,
#         api_base=st_openai_api_base,
#         deployment_id=st_deployment_id,
#         api_version=st_openai_version,
#         api_type=openai_api_type,
#     )

# st_threshold = st.sidebar.slider(
#     label="Acceptance threshold",
#     min_value=0.0,
#     max_value=1.0,
#     value=0.35,
#     help="Define the threshold for accepting a detection as PII. See more here: ",
# )

# st_return_decision_process = st.sidebar.checkbox(
#     "Add analysis explanations to findings",
#     value=False,
#     help="Add the decision process to the output table. "
#     "More information can be found here: https://microsoft.github.io/presidio/analyzer/decision_process/",
# )

# # Allow and deny lists
# st_deny_allow_expander = st.sidebar.expander(
#     "Allowlists and denylists",
#     expanded=False,
# )

# with st_deny_allow_expander:
#     st_allow_list = st_tags(
#         label="Add words to the allowlist", text="Enter word and press enter."
#     )
#     st.caption(
#         "Allowlists contain words that are not considered PII, but are detected as such."
#     )

#     st_deny_list = st_tags(
#         label="Add words to the denylist", text="Enter word and press enter."
#     )
#     st.caption(
#         "Denylists contain words that are considered PII, but are not detected as such."
#     )

# # Search option
# search_query = st.sidebar.text_input("Search Text", "")
# search_button = st.sidebar.button("Search")

# # Main panel
# with st.expander("About this demo", expanded=False):
#     # About section content
#     pass

# analyzer_load_state = st.info("Starting Presidio analyzer...")

# analyzer_load_state.empty()

# # Read default text
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as doc:
#         for page in doc:
#             text += page.get_text()
#     return text

# def extract_text_from_image(image_path):
#     with Image.open(image_path) as img:
#         text = textract.process(image_path, method='tesseract')
#     return text.decode('utf-8')

# def extract_text_from_doc(doc_path):
#     text = textract.process(doc_path).decode('utf-8')
#     return text

# def extract_text_from_file(file_path):
#     file_extension = os.path.splitext(file_path)[1].lower()
#     if file_extension in ('.pdf'):
#         return extract_text_from_pdf(file_path)
#     elif file_extension in ('.jpg', '.jpeg', '.png'):
#         return extract_text_from_image(file_path)
#     elif file_extension in ('.txt', '.doc', '.docx'):
#         return extract_text_from_doc(file_path)
#     else:
#         return ""  # Unsupported file format

# uploaded_files = st.file_uploader("Upload files", type=["pdf", "jpg", "jpeg", "png", "txt", "doc", "docx"], accept_multiple_files=True)

# if uploaded_files:
#     for idx, uploaded_file in enumerate(uploaded_files):
#         uploaded_file_path = f"uploaded_file_{idx}_{uuid.uuid4()}{os.path.splitext(uploaded_file.name)[1]}"
#         with open(uploaded_file_path, "wb") as f:
#             f.write(uploaded_file.getvalue())

#         pdf_text = extract_text_from_file(uploaded_file_path)

#         # Create two columns for before and after
#         col1, col2 = st.columns(2)

#         # Before:
#         col1.subheader("Input")
#         st_text = col1.text_area(
#             label=f"Enter text {idx}", value=pdf_text, height=400, key=f"text_input_{idx}"
#         )

#         if search_button:
#             search_results = []
#             if search_query:
#                 search_results = [(m.start(), m.end()) for m in re.finditer(re.escape(search_query), st_text, re.IGNORECASE)]

#             if search_results:
#                 for start, end in search_results:
#                     st_text = f"{st_text[:start]}**{st_text[start:end]}**{st_text[end:]}"
#                 st.text_area("Highlighted Search Results", st_text, height=400)
#             else:
#                 st.warning("No matches found.")

#         try:
#             # Choose entities
#             st_entities_expander = st.sidebar.expander("Choose entities to look for")
#             st_entities = st_entities_expander.multiselect(
#                 label="Which entities to look for?",
#                 options=get_supported_entities(*analyzer_params),
#                 default=list(get_supported_entities(*analyzer_params)),
#                 help="Limit the list of PII entities detected. "
#                 "This list is dynamic and based on the NER model and registered recognizers. "
#                 "More information can be found here: https://microsoft.github.io/presidio/analyzer/adding_recognizers/",
#             )
#             # Initialize Presidio Analyzer with money recognizer
#             registry = RecognizerRegistry()
#             registry.load_predefined_recognizers()

#             # Define a pattern recognizer for the 'Money' entity if it's not already available
#             money_pattern = Pattern(name="money_pattern", regex=r"\b(?:\$|US\$|C\$|A\$|£|€|¥|₹)?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b", score=0.85)
#             money_recognizer = PatternRecognizer(supported_entity="MONEY", patterns=[money_pattern])
#             registry.add_recognizer(money_recognizer)
#             # Before
#             analyzer_load_state = st.info("Starting Presidio analyzer...")
#             analyzer = analyzer_engine(*analyzer_params,registry=registry)
#             analyzer_load_state.empty()

#             st_analyze_results = analyze(
#                 *analyzer_params,
#                 text=st_text,
#                 entities=st_entities,
#                 language="en",
#                 score_threshold=st_threshold,
#                 return_decision_process=st_return_decision_process,
#                 allow_list=st_allow_list,
#                 deny_list=st_deny_list,
#             )

#             # After
#             if st_operator not in ("highlight", "synthesize"):
#                 with col2:
#                     st.subheader(f"Output")
#                     st_anonymize_results = anonymize(
#                         text=st_text,
#                         operator=st_operator,
#                         mask_char=st_mask_char,
#                         number_of_chars=st_number_of_chars,
#                         encrypt_key=st_encrypt_key,
#                         analyze_results=st_analyze_results,
#                     )
#                     st.text_area(
#                         label="De-identified", value=st_anonymize_results.text, height=400
#                     )
#             elif st_operator == "synthesize":
#                 with col2:
#                     st.subheader(f"OpenAI Generated output")
#                     fake_data = create_fake_data(
#                         st_text,
#                         st_analyze_results,
#                         open_ai_params,
#                     )
#                     st.text_area(label="Synthetic data", value=fake_data, height=400)
#             else:
#                 st.subheader("Highlighted")
#                 annotated_tokens = annotate(text=st_text, analyze_results=st_analyze_results)
#                 # annotated_tokens
#                 annotated_text(*annotated_tokens)

#             # table result
#             st.subheader(
#                 "Findings"
#                 if not st_return_decision_process
#                 else "Findings with decision factors"
#             )
#             if st_analyze_results:
#                 df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
#                 df["text"] = [st_text[res.start : res.end] for res in st_analyze_results]

#                 df_subset = df[["entity_type", "text", "start", "end", "score"]].rename(
#                     {
#                         "entity_type": "Entity type",
#                         "text": "Text",
#                         "start": "Start",
#                         "end": "End",
#                         "score": "Confidence",
#                     },
#                     axis=1,
#                 )
#                 df_subset["Text"] = [st_text[res.start : res.end] for res in st_analyze_results]
#                 if st_return_decision_process:
#                     analysis_explanation_df = pd.DataFrame.from_records(
#                         [r.analysis_explanation.to_dict() for r in st_analyze_results]
#                     )
#                     df_subset = pd.concat([df_subset, analysis_explanation_df], axis=1)
#                 st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)
#             else:
#                 st.text("No findings")

#         except Exception as e:
#             print(e)
#             traceback.print_exc()
#             st.error(e)

# components.html(
#     """
#     <script type="text/javascript">
#     (function(c,l,a,r,i,t,y){
#         c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
#         t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
#         y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
#     })(window, document, "clarity", "script", "h7f8bp42n8");
#     </script>
#     """
# )












import logging
import os
import traceback
import dotenv
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import textract
from pdf2image import convert_from_path
import fitz
import re

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.recognizer_registry import RecognizerRegistry
from presidio_analyzer.predefined_recognizers import get_recognizers
from money_recognizer import MoneyRecognizer 

st.set_page_config(
    page_title="RT SECURE",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "https://microsoft.github.io/presidio/",
    },
)

dotenv.load_dotenv()
logger = logging.getLogger("presidio-streamlit")

allow_other_models = os.getenv("ALLOW_OTHER_MODELS", False)

model_help_text = """
    Select which Named Entity Recognition (NER) model to use for PII detection, in parallel to rule-based recognizers.
    Presidio supports multiple NER packages off-the-shelf, such as spaCy, Huggingface, Stanza and Flair,
    as well as service such as Azure Text Analytics PII.
    """
st_ta_key = st_ta_endpoint = ""

model_list = [
    "spaCy/en_core_web_lg",
    "flair/ner-english-large",
    "HuggingFace/obi/deid_roberta_i2b2",
    "HuggingFace/StanfordAIMI/stanford-deidentifier-base",
    "stanza/en",
    "Azure AI Language",
    "Other",
]
if not allow_other_models:
    model_list.pop()

# Select model
st_model = st.sidebar.selectbox(
    "NER model package",
    model_list,
    index=2,
    help=model_help_text,
)

# Extract model package.
st_model_package = st_model.split("/")[0]

# Remove package prefix (if needed)
st_model = (
    st_model
    if st_model_package.lower() not in ("spacy", "stanza", "huggingface")
    else "/".join(st_model.split("/")[1:])
)

if st_model == "Other":
    st_model_package = st.sidebar.selectbox(
        "NER model package",
        model_list,
        index=2,
        help=model_help_text,
    )
    st_model = st.sidebar.text_input("NER model")

uploaded_file = st.sidebar.file_uploader("Upload a file", type=["txt", "jpg", "png", "jpeg", "pdf"])

def annotate(text, analyze_results):
    """ Annotate the text based on the analysis results """
    annotations = []
    last_idx = 0
    for result in analyze_results:
        annotations.append(text[last_idx:result.start])
        annotations.append((text[result.start:result.end], result.entity_type))
        last_idx = result.end
    annotations.append(text[last_idx:])
    return annotations

def annotated_text(*args):
    """ Display annotated text in Streamlit """
    st.markdown(
        "".join([f"<span style='color: red;'>{text}</span>" if isinstance(text, tuple) else text for text in args]),
        unsafe_allow_html=True,
    )

try:
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            pdf_pages = convert_from_path(uploaded_file)
            text = "\n".join([str(textract.process(page).decode("utf-8")) for page in pdf_pages])
        elif uploaded_file.type in ["image/jpg", "image/jpeg", "image/png"]:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
        else:
            text = uploaded_file.getvalue().decode("utf-8")

        st.subheader("Extracted Text")
        st.text_area(label="Extracted text", value=text, height=300)

        # Initialize the Presidio Analyzer Engine
        analyzer = AnalyzerEngine()

        # Add the predefined recognizers to the engine
        for recognizer in predefined_recognizers():
            analyzer.registry.add_recognizer(recognizer)

        # Add the custom Money recognizer
        money_recognizer = MoneyRecognizer()
        analyzer.registry.add_recognizer(money_recognizer)

        # Analyze text with the custom recognizers included
        results = analyzer.analyze(
            text=text,
            language="en",
            return_decision_process=True,
        )
        st_analyze_results = results

        # Display annotated text
        annotated_tokens = annotate(text=text, analyze_results=st_analyze_results)
        annotated_text(*annotated_tokens)

        # Display findings
        df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
        df["text"] = [text[res.start : res.end] for res in st_analyze_results]
        df_subset = df[["entity_type", "text", "start", "end", "score"]].rename({
            "entity_type": "Entity type",
            "text": "Text",
            "start": "Start",
            "end": "End",
            "score": "Confidence",
        }, axis=1)
        st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)

except Exception as e:
    print(e)
    traceback.print_exc()
    st.error(e)

components.html(
    """
    <script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "h7f8bp42n8");
    </script>
    """
)
