from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta


THRESHOLD_DIRECTIONS: Dict[str, str] = {
    "msp": ">=",
    "margin": ">=",
    "entropy": "<=",
    "energy": "<=",
}


@dataclass(frozen=True)
class Candidate:
    score_name: str
    threshold: float
    threshold_direction: str
    gamma: float
    alpha: float
    prop_coverage: float
    prop_risk: float
    prop_n: int
    prop_k: int


@dataclass
class CertificationResult:
    alpha: float
    delta: float
    delta_total: float
    delta_risk: float
    delta_coverage: float
    delta_allocation: str
    gamma: float
    certified: bool
    score_name: Optional[str]
    threshold: Optional[float]
    threshold_direction: Optional[str]
    prop_coverage: float
    prop_risk: float
    prop_n: int
    prop_k: int
    cert_n: int
    cert_k: int
    cert_risk_emp: float
    cert_risk_ucb: float
    cert_coverage_emp: float
    cert_coverage_lcb: float
    certified_coverage_at_alpha: float
    test_n: int
    test_k: int
    test_risk: float
    test_coverage: float
    test_accuracy_all: float
    certificate_scope: str
    n_certified_candidates: int
    zero_error_min_cert_n: int
    cert_n_minus_zero_error_min_n: int
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _validate_delta(delta: float) -> None:
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0,1)")


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def clopper_pearson_ucb(k: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson UCB for a binomial error rate."""
    _validate_delta(delta)
    if n == 0:
        return 1.0
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"Invalid k={k}, n={n}")
    if k == n:
        return 1.0
    if k == 0:
        return _clip01(1.0 - float(delta) ** (1.0 / float(n)))
    return _clip01(float(beta.ppf(1.0 - delta, k + 1, n - k)))


def clopper_pearson_lcb(k: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson LCB for a binomial success rate."""
    _validate_delta(delta)
    if n == 0:
        return 0.0
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"Invalid k={k}, n={n}")
    if k == 0:
        return 0.0
    return _clip01(float(beta.ppf(delta, k, n - k + 1)))


def zero_error_min_n(alpha: float, delta_risk: float) -> int:
    """Minimum accepted N needed to certify risk alpha with zero accepted errors."""
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1)")
    _validate_delta(delta_risk)
    return int(np.ceil(np.log(delta_risk) / np.log(1.0 - alpha)))


# Backward-compatible names used by older tests/scripts.
cp_upper = clopper_pearson_ucb
cp_lower = clopper_pearson_lcb
zero_error_n_min = zero_error_min_n


def threshold_direction(score_name: str) -> str:
    return THRESHOLD_DIRECTIONS.get(score_name, "<=")


def acceptance_mask(scores: np.ndarray, threshold: float, direction: str = "<=") -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if direction == "<=":
        return scores <= threshold
    if direction == ">=":
        return scores >= threshold
    raise ValueError(f"Unknown threshold direction: {direction}")


def empirical_counts(
    scores: np.ndarray,
    errors: np.ndarray,
    threshold: float,
    direction: str = "<=",
) -> Tuple[int, int, float, float]:
    mask = acceptance_mask(scores, threshold, direction)
    n = int(mask.sum())
    k = int(np.asarray(errors)[mask].sum()) if n > 0 else 0
    coverage = n / len(scores) if len(scores) else 0.0
    risk = k / n if n > 0 else float("nan")
    return n, k, coverage, risk


def make_thresholds(scores: np.ndarray, num_thresholds: int = 200) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.array([], dtype=float)
    qs = np.linspace(0.001, 1.0, num_thresholds)
    return np.unique(np.quantile(scores, qs)).astype(float)


def _conservative_threshold_key(threshold: float, direction: str) -> float:
    return float(threshold) if direction == ">=" else -float(threshold)


def propose_candidate(
    prop_scores: Dict[str, np.ndarray],
    prop_errors: np.ndarray,
    alpha: float,
    gamma: float,
    num_thresholds: int = 200,
    min_prop_accept: int = 1,
) -> Optional[Candidate]:
    """Select max-coverage candidate using only the proposal split."""
    best: Optional[Candidate] = None
    risk_target = gamma * alpha
    for score_index, (score_name, scores) in enumerate(prop_scores.items()):
        direction = threshold_direction(score_name)
        for threshold_index, thr in enumerate(make_thresholds(scores, num_thresholds=num_thresholds)):
            n, k, cov, risk = empirical_counts(scores, prop_errors, float(thr), direction)
            if n < min_prop_accept or not isfinite(risk):
                continue
            if risk <= risk_target:
                cand = Candidate(
                    score_name=score_name,
                    threshold=float(thr),
                    threshold_direction=direction,
                    gamma=float(gamma),
                    alpha=float(alpha),
                    prop_coverage=float(cov),
                    prop_risk=float(risk),
                    prop_n=int(n),
                    prop_k=int(k),
                )
                cand_key = (
                    cand.prop_coverage,
                    -cand.prop_risk,
                    _conservative_threshold_key(cand.threshold, cand.threshold_direction),
                    -score_index,
                    -threshold_index,
                )
                if best is None:
                    best = cand
                else:
                    best_key = (
                        best.prop_coverage,
                        -best.prop_risk,
                        _conservative_threshold_key(best.threshold, best.threshold_direction),
                        0.0,
                        0.0,
                    )
                    if cand_key > best_key:
                        best = cand
    return best


def certification_reason(cert_n: int, cert_k: int, cert_risk_ucb: float, alpha: float) -> str:
    if cert_n == 0:
        return "zero_cert_accepts"
    if cert_risk_ucb <= alpha:
        return "certified"
    if cert_k == 0:
        return "insufficient_cert_n"
    if cert_risk_ucb > alpha:
        return "high_cert_error_ucb"
    return "not_certified"


def certify_candidate(
    cand: Candidate,
    cert_scores: Dict[str, np.ndarray],
    cert_errors: np.ndarray,
    test_scores: Dict[str, np.ndarray],
    test_errors: np.ndarray,
    alpha: float,
    delta_risk: float,
    delta_coverage: float,
    delta_total: float,
    delta_allocation: str,
    certificate_scope: str,
    n_certified_candidates: int,
) -> CertificationResult:
    s_cert = cert_scores[cand.score_name]
    s_test = test_scores[cand.score_name]
    direction = cand.threshold_direction

    cert_n, cert_k, cert_cov, cert_risk = empirical_counts(s_cert, cert_errors, cand.threshold, direction)
    cert_ucb = clopper_pearson_ucb(cert_k, cert_n, delta=delta_risk)
    cert_cov_lcb = clopper_pearson_lcb(cert_n, len(s_cert), delta=delta_coverage)

    test_n, test_k, test_cov, test_risk = empirical_counts(s_test, test_errors, cand.threshold, direction)
    test_acc_all = 1.0 - float(np.asarray(test_errors).mean())

    certified = bool(cert_n > 0 and cert_ucb <= alpha)
    certified_coverage = float(cert_cov_lcb) if certified else 0.0
    zero_min = zero_error_min_n(alpha, delta_risk)
    reason = certification_reason(cert_n, cert_k, cert_ucb, alpha)
    return CertificationResult(
        alpha=float(alpha),
        delta=float(delta_risk),
        delta_total=float(delta_total),
        delta_risk=float(delta_risk),
        delta_coverage=float(delta_coverage),
        delta_allocation=delta_allocation,
        gamma=float(cand.gamma),
        certified=certified,
        score_name=cand.score_name,
        threshold=float(cand.threshold),
        threshold_direction=direction,
        prop_coverage=float(cand.prop_coverage),
        prop_risk=float(cand.prop_risk),
        prop_n=int(cand.prop_n),
        prop_k=int(cand.prop_k),
        cert_n=int(cert_n),
        cert_k=int(cert_k),
        cert_risk_emp=float(cert_risk) if isfinite(cert_risk) else float("nan"),
        cert_risk_ucb=float(cert_ucb),
        cert_coverage_emp=float(cert_cov),
        cert_coverage_lcb=float(cert_cov_lcb),
        certified_coverage_at_alpha=certified_coverage,
        test_n=int(test_n),
        test_k=int(test_k),
        test_risk=float(test_risk) if isfinite(test_risk) else float("nan"),
        test_coverage=float(test_cov),
        test_accuracy_all=float(test_acc_all),
        certificate_scope=certificate_scope,
        n_certified_candidates=int(n_certified_candidates),
        zero_error_min_cert_n=int(zero_min),
        cert_n_minus_zero_error_min_n=int(cert_n - zero_min),
        reason=reason,
    )


def failed_result(
    alpha: float,
    delta_risk: float,
    delta_coverage: float,
    delta_total: float,
    delta_allocation: str,
    gamma: float,
    reason: str,
    certificate_scope: str,
    n_certified_candidates: int,
    test_errors: Optional[np.ndarray] = None,
) -> CertificationResult:
    test_acc_all = float("nan") if test_errors is None else 1.0 - float(np.asarray(test_errors).mean())
    zero_min = zero_error_min_n(alpha, delta_risk)
    return CertificationResult(
        alpha=float(alpha),
        delta=float(delta_risk),
        delta_total=float(delta_total),
        delta_risk=float(delta_risk),
        delta_coverage=float(delta_coverage),
        delta_allocation=delta_allocation,
        gamma=float(gamma),
        certified=False,
        score_name=None,
        threshold=None,
        threshold_direction=None,
        prop_coverage=0.0,
        prop_risk=float("nan"),
        prop_n=0,
        prop_k=0,
        cert_n=0,
        cert_k=0,
        cert_risk_emp=float("nan"),
        cert_risk_ucb=1.0,
        cert_coverage_emp=0.0,
        cert_coverage_lcb=0.0,
        certified_coverage_at_alpha=0.0,
        test_n=0,
        test_k=0,
        test_risk=float("nan"),
        test_coverage=0.0,
        test_accuracy_all=test_acc_all,
        certificate_scope=certificate_scope,
        n_certified_candidates=int(n_certified_candidates),
        zero_error_min_cert_n=int(zero_min),
        cert_n_minus_zero_error_min_n=-int(zero_min),
        reason=reason,
    )


def run_certification_for_alpha(
    prop_scores: Dict[str, np.ndarray],
    prop_errors: np.ndarray,
    cert_scores: Dict[str, np.ndarray],
    cert_errors: np.ndarray,
    test_scores: Dict[str, np.ndarray],
    test_errors: np.ndarray,
    alpha: float,
    delta: Optional[float] = None,
    gammas: Sequence[float] = (0.5, 0.7, 1.0),
    num_thresholds: int = 200,
    min_prop_accept: int = 1,
    bonferroni_over_gammas: bool = True,
    delta_total: Optional[float] = None,
    delta_risk: Optional[float] = None,
    delta_coverage: Optional[float] = None,
    delta_allocation: Optional[str] = None,
) -> List[CertificationResult]:
    """Run proposal/certification rows without using cert labels for proposal."""
    gammas = list(gammas)
    if delta_allocation is None:
        delta_allocation = "risk_only_legacy" if delta is not None else "joint_split"
    if delta_allocation == "risk_only_legacy":
        if delta is None:
            delta = 0.05
        delta_total = float(delta if delta_total is None else delta_total)
        base_delta_risk = float(delta if delta_risk is None else delta_risk)
        base_delta_coverage = float(delta if delta_coverage is None else delta_coverage)
    else:
        delta_total = float(0.05 if delta_total is None else delta_total)
        base_delta_risk = float(delta_total / 2.0 if delta_risk is None else delta_risk)
        base_delta_coverage = float(delta_total / 2.0 if delta_coverage is None else delta_coverage)
    _validate_delta(base_delta_risk)
    _validate_delta(base_delta_coverage)

    n_rows = max(1, len(gammas))
    if bonferroni_over_gammas and n_rows > 1:
        row_delta_risk = base_delta_risk / n_rows
        row_delta_coverage = base_delta_coverage / n_rows
        certificate_scope = "simultaneous_rows"
    else:
        row_delta_risk = base_delta_risk
        row_delta_coverage = base_delta_coverage
        certificate_scope = "individual_row" if n_rows > 1 else "single_selector"
    n_certified_candidates = n_rows if certificate_scope == "simultaneous_rows" else 1

    results: List[CertificationResult] = []
    for gamma in gammas:
        cand = propose_candidate(
            prop_scores=prop_scores,
            prop_errors=prop_errors,
            alpha=alpha,
            gamma=gamma,
            num_thresholds=num_thresholds,
            min_prop_accept=min_prop_accept,
        )
        if cand is None:
            results.append(
                failed_result(
                    alpha,
                    row_delta_risk,
                    row_delta_coverage,
                    float(delta_total),
                    delta_allocation,
                    gamma,
                    "no_proposal_candidate",
                    certificate_scope,
                    n_certified_candidates,
                    test_errors,
                )
            )
            continue
        results.append(
            certify_candidate(
                cand,
                cert_scores=cert_scores,
                cert_errors=cert_errors,
                test_scores=test_scores,
                test_errors=test_errors,
                alpha=alpha,
                delta_risk=row_delta_risk,
                delta_coverage=row_delta_coverage,
                delta_total=float(delta_total),
                delta_allocation=delta_allocation,
                certificate_scope=certificate_scope,
                n_certified_candidates=n_certified_candidates,
            )
        )
    return results


def best_certified_result(results: Sequence[CertificationResult]) -> Optional[CertificationResult]:
    certified = [r for r in results if r.certified]
    if not certified:
        return None
    return max(certified, key=lambda r: (r.certified_coverage_at_alpha, -r.cert_risk_ucb, r.prop_coverage))
