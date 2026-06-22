from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class Candidate:
    score_name: str
    threshold: float
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
    gamma: float
    certified: bool
    score_name: Optional[str]
    threshold: Optional[float]
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
    test_n: int
    test_k: int
    test_risk: float
    test_coverage: float
    test_accuracy_all: float
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def cp_upper(k: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper confidence bound for binomial proportion.

    Valid conditional on n accepted samples. If n=0, returns 1.0.
    """
    if n <= 0:
        return 1.0
    if k < 0 or k > n:
        raise ValueError(f"Invalid k={k}, n={n}")
    if k == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def cp_lower(k: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson lower confidence bound for binomial proportion."""
    if n <= 0:
        return 0.0
    if k < 0 or k > n:
        raise ValueError(f"Invalid k={k}, n={n}")
    if k == 0:
        return 0.0
    return float(beta.ppf(delta, k, n - k + 1))


def zero_error_n_min(alpha: float, delta: float) -> int:
    """Minimum accepted N needed to certify risk alpha with zero accepted errors."""
    if not (0 < alpha < 1) or not (0 < delta < 1):
        raise ValueError("alpha and delta must be in (0,1)")
    return int(np.ceil(np.log(delta) / np.log(1.0 - alpha)))


def acceptance_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores) <= threshold


def empirical_counts(scores: np.ndarray, errors: np.ndarray, threshold: float) -> Tuple[int, int, float, float]:
    mask = acceptance_mask(scores, threshold)
    n = int(mask.sum())
    k = int(np.asarray(errors)[mask].sum()) if n > 0 else 0
    coverage = n / len(scores) if len(scores) else 0.0
    risk = k / n if n > 0 else 1.0
    return n, k, coverage, risk


def make_thresholds(scores: np.ndarray, num_thresholds: int = 200) -> np.ndarray:
    """Create candidate thresholds for scores where lower is safer.

    Thresholds are quantiles over the proposal scores, plus the minimum score.
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.array([], dtype=float)
    qs = np.linspace(0.001, 1.0, num_thresholds)
    thresholds = np.quantile(scores, qs)
    thresholds = np.unique(thresholds)
    return thresholds.astype(float)


def propose_candidate(
    prop_scores: Dict[str, np.ndarray],
    prop_errors: np.ndarray,
    alpha: float,
    gamma: float,
    num_thresholds: int = 200,
    min_prop_accept: int = 1,
) -> Optional[Candidate]:
    """Select max-coverage candidate using proposal split and risk buffer."""
    best: Optional[Candidate] = None
    risk_target = gamma * alpha
    for score_name, scores in prop_scores.items():
        thresholds = make_thresholds(scores, num_thresholds=num_thresholds)
        for thr in thresholds:
            n, k, cov, risk = empirical_counts(scores, prop_errors, float(thr))
            if n < min_prop_accept:
                continue
            if risk <= risk_target:
                cand = Candidate(
                    score_name=score_name,
                    threshold=float(thr),
                    gamma=float(gamma),
                    alpha=float(alpha),
                    prop_coverage=float(cov),
                    prop_risk=float(risk),
                    prop_n=int(n),
                    prop_k=int(k),
                )
                if best is None:
                    best = cand
                else:
                    # Max coverage; tie-break by lower prop risk, then larger accepted N.
                    if (cand.prop_coverage, -cand.prop_risk, cand.prop_n) > (
                        best.prop_coverage,
                        -best.prop_risk,
                        best.prop_n,
                    ):
                        best = cand
    return best


def certify_candidate(
    cand: Candidate,
    cert_scores: Dict[str, np.ndarray],
    cert_errors: np.ndarray,
    test_scores: Dict[str, np.ndarray],
    test_errors: np.ndarray,
    alpha: float,
    delta: float,
    coverage_delta: Optional[float] = None,
) -> CertificationResult:
    if coverage_delta is None:
        coverage_delta = delta
    s_cert = cert_scores[cand.score_name]
    s_test = test_scores[cand.score_name]

    cert_n, cert_k, cert_cov, cert_risk = empirical_counts(s_cert, cert_errors, cand.threshold)
    cert_ucb = cp_upper(cert_k, cert_n, delta=delta)
    cert_cov_lcb = cp_lower(cert_n, len(s_cert), delta=coverage_delta)

    test_n, test_k, test_cov, test_risk = empirical_counts(s_test, test_errors, cand.threshold)
    test_acc_all = 1.0 - float(np.asarray(test_errors).mean())

    certified = cert_ucb <= alpha
    reason = "certified" if certified else "risk_ucb_exceeds_alpha"
    return CertificationResult(
        alpha=float(alpha),
        delta=float(delta),
        gamma=float(cand.gamma),
        certified=bool(certified),
        score_name=cand.score_name,
        threshold=float(cand.threshold),
        prop_coverage=float(cand.prop_coverage),
        prop_risk=float(cand.prop_risk),
        prop_n=int(cand.prop_n),
        prop_k=int(cand.prop_k),
        cert_n=int(cert_n),
        cert_k=int(cert_k),
        cert_risk_emp=float(cert_risk),
        cert_risk_ucb=float(cert_ucb),
        cert_coverage_emp=float(cert_cov),
        cert_coverage_lcb=float(cert_cov_lcb),
        test_n=int(test_n),
        test_k=int(test_k),
        test_risk=float(test_risk),
        test_coverage=float(test_cov),
        test_accuracy_all=float(test_acc_all),
        reason=reason,
    )


def failed_result(
    alpha: float,
    delta: float,
    gamma: float,
    reason: str,
    test_errors: Optional[np.ndarray] = None,
) -> CertificationResult:
    test_acc_all = float("nan") if test_errors is None else 1.0 - float(np.asarray(test_errors).mean())
    return CertificationResult(
        alpha=float(alpha),
        delta=float(delta),
        gamma=float(gamma),
        certified=False,
        score_name=None,
        threshold=None,
        prop_coverage=0.0,
        prop_risk=1.0,
        prop_n=0,
        prop_k=0,
        cert_n=0,
        cert_k=0,
        cert_risk_emp=1.0,
        cert_risk_ucb=1.0,
        cert_coverage_emp=0.0,
        cert_coverage_lcb=0.0,
        test_n=0,
        test_k=0,
        test_risk=1.0,
        test_coverage=0.0,
        test_accuracy_all=test_acc_all,
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
    delta: float,
    gammas: Sequence[float],
    num_thresholds: int = 200,
    min_prop_accept: int = 1,
    bonferroni_over_gammas: bool = True,
) -> List[CertificationResult]:
    """Run proposal/certification for all gammas.

    If bonferroni_over_gammas=True, each gamma receives delta/len(gammas), allowing the
    analyst to select among gamma results post hoc while preserving a union-bound guarantee.
    """
    results: List[CertificationResult] = []
    gammas = list(gammas)
    delta_each = delta / len(gammas) if bonferroni_over_gammas and len(gammas) > 0 else delta
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
            results.append(failed_result(alpha, delta_each, gamma, "no_candidate_passed_prop_buffer", test_errors))
            continue
        res = certify_candidate(
            cand,
            cert_scores=cert_scores,
            cert_errors=cert_errors,
            test_scores=test_scores,
            test_errors=test_errors,
            alpha=alpha,
            delta=delta_each,
            coverage_delta=delta_each,
        )
        results.append(res)
    return results


def best_certified_result(results: Sequence[CertificationResult]) -> Optional[CertificationResult]:
    certified = [r for r in results if r.certified]
    if not certified:
        return None
    return max(certified, key=lambda r: (r.cert_coverage_lcb, r.test_coverage, -r.cert_risk_ucb))
