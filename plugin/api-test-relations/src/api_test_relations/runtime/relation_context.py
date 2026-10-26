"""Shared causal owner, usable before importing torch or the native adapter."""
from contextvars import ContextVar
OWNER=ContextVar('api_relation_owner',default=None)
