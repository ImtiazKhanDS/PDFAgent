from typing import Any, Dict, Optional, Union
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.retrievers import (
    BaseRetriever,
    RetrieverOutput,
)
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
    _validate_prompt,
)


def create_stuff_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate,
    *,
    output_parser: Optional[BaseOutputParser] = None,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = DEFAULT_DOCUMENT_SEPARATOR,
    document_variable_name: str = DOCUMENTS_KEY,
) -> Runnable[Dict[str, Any], Any]:
    """Create a chain for passing a list of Documents to a model.

    Args:
        llm: Language model.
        prompt: Prompt template. Must contain input variable "context" (override by
            setting document_variable), which will be used for passing in the formatted documents.
        output_parser: Output parser. Defaults to StrOutputParser.
        document_prompt: Prompt used for formatting each document into a string. Input
            variables can be "page_content" or any metadata keys that are in all
            documents. "page_content" will automatically retrieve the
            `Document.page_content`, and all other inputs variables will be
            automatically retrieved from the `Document.metadata` dictionary. Default to
            a prompt that only contains `Document.page_content`.
        document_separator: String separator to use between formatted document strings.
        document_variable_name: Variable name to use for the formatted documents in the prompt.
            Defaults to "context".

    Returns:
        An LCEL Runnable. The input is a dictionary that must have a "context" key that
        maps to a List[Document], and any other input variables expected in the prompt.
        The Runnable return type depends on output_parser used.
    """

    _validate_prompt(prompt, document_variable_name)
    _document_prompt = document_prompt or DEFAULT_DOCUMENT_PROMPT
    _output_parser = output_parser or StrOutputParser()

    def format_docs(inputs: dict) -> str:
        return document_separator.join(
            format_document(doc, _document_prompt)
            for doc in inputs[document_variable_name]
        )

    return (
        RunnablePassthrough.assign(**{document_variable_name: format_docs}).with_config(
            run_name="format_inputs"
        )
        | prompt
        | llm
        | _output_parser
    ).with_config(run_name="stuff_documents_chain")


def create_retrieval_chain(
    retriever: Union[BaseRetriever, Runnable[dict, RetrieverOutput]],
    combine_docs_chain: Runnable[Dict[str, Any], str],
) -> Runnable:
    """Create retrieval chain that retrieves documents and then passes them on.

    Args:
        retriever: Retriever-like object that returns list of documents. Should
            either be a subclass of BaseRetriever or a Runnable that returns
            a list of documents. If a subclass of BaseRetriever, then it
            is expected that an `input` key be passed in - this is what
            is will be used to pass into the retriever. If this is NOT a
            subclass of BaseRetriever, then all the inputs will be passed
            into this runnable, meaning that runnable should take a dictionary
            as input.
        combine_docs_chain: Runnable that takes inputs and produces a string output.
            The inputs to this will be any original inputs to this chain, a new
            context key with the retrieved documents, and chat_history (if not present
            in the inputs) with a value of `[]` (to easily enable conversational
            retrieval.

    Returns:
        An LCEL Runnable. The Runnable return is a dictionary containing at the very
        least a `context` and `answer` key.
    """
    if not isinstance(retriever, BaseRetriever):
        retrieval_docs: Runnable[dict, RetrieverOutput] = retriever
    else:
        retrieval_docs = (lambda x: x["input"]) | retriever

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=retrieval_docs.with_config(run_name="retrieve_documents"),
        ).assign(answer=combine_docs_chain)
    ).with_config(run_name="retrieval_chain")

    return retrieval_chain