"""Sanitized policy-processing failures without source text or local paths."""


class PolicyExtractionError(ValueError):
    """Reject unreadable PDFs or documents without usable numbered sections."""


class PolicyIndexError(RuntimeError):
    """Report a failed atomic index replacement without database diagnostics."""


class PolicyIndexMissing(RuntimeError):
    """Indicate that policy tables or a completed policy index are unavailable."""


class PolicySearchError(RuntimeError):
    """Report an operational retrieval failure without query or source data."""
