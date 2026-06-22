"""Learning hooks and loss utilities."""

from .logit_adjustment import adjust_logits, log_prior_offsets, margin_distortion

__all__ = ["adjust_logits", "log_prior_offsets", "margin_distortion"]
