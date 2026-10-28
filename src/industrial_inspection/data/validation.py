"""Compatibility module for dataset audit functions."""

from industrial_inspection.data.contracts import AuditReport, validate_manifest, write_audit_report

__all__ = ["AuditReport", "validate_manifest", "write_audit_report"]
