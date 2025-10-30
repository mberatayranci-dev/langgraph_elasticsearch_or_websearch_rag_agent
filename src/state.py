"""
LangGraph State tanımları
"""
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """
    Agent'ın state yapısı
    
    Attributes:
        messages: Konuşma geçmişi
        is_elasticsearch_related: Sorunun Elasticsearch ile ilgili olup olmadığı
        context: Retrieve edilen veya arama yapılan bilgiler
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    is_elasticsearch_related: bool
    context: str