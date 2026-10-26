"""Activated only for children carrying an explicit parent-test context."""
import os
if os.environ.get('API_RELATION_PARENT'):
    from relation_process import bootstrap
    bootstrap()
