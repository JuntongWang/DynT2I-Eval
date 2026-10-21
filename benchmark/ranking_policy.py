import math
from dataclasses import asdict, dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class RankingConfig:
    mu_init: float = 1000.0
    sigma_init: float = 300.0
    beta: float = 70.0
    eta: float = 3.0
    alpha: float = 190.0
    sigma_conv: float = 20.0
    decisive_p_low: float = 0.42
    decisive_p_high: float = 0.58
    tie_sigma_shrink: float = 0.98
    tie_mean_blend: float = 0.05
    weight_scale: float = 2.0
    max_weight: float = 2.0
    sigma_tiebreak_power: float = 1.0
    eta_warmup_matches: float = 20.0
    eta_floor_ratio: float = 0.20


DEFAULT_RANKING_CONFIG = RankingConfig()


def config_as_dict(cfg: RankingConfig = DEFAULT_RANKING_CONFIG) -> dict:
    return asdict(cfg)


def effective_eta(matches: float, cfg: RankingConfig = DEFAULT_RANKING_CONFIG) -> float:
    if cfg.eta_warmup_matches <= 0:
        return cfg.eta
    progress = max(cfg.eta_floor_ratio, min(1.0, matches / cfg.eta_warmup_matches))
    return cfg.eta * progress


def conservative_score(mu: float, sigma: float, matches: float, cfg: RankingConfig = DEFAULT_RANKING_CONFIG) -> float:
    return mu - effective_eta(matches, cfg) * sigma


def sync_model_record(model_data: dict, cfg: RankingConfig = DEFAULT_RANKING_CONFIG) -> dict:
    model_data.setdefault("mu", cfg.mu_init)
    model_data.setdefault("sigma", cfg.sigma_init)
    model_data.setdefault("status", "active")
    model_data.setdefault("matches", 0)
    model_data.setdefault("match_history", {})
    model_data.setdefault("wins", 0)
    model_data.setdefault("losses", 0)
    model_data.setdefault("ties", 0)
    model_data["score"] = conservative_score(
        model_data["mu"],
        model_data["sigma"],
        model_data["matches"],
        cfg,
    )
    return model_data


def opponent_selection_tuple(
    pivot_mu: float,
    opponent_mu: float,
    opponent_sigma: float,
    total_rounds: int,
    pair_matches: int,
    rng_jitter: float,
    cfg: RankingConfig = DEFAULT_RANKING_CONFIG,
) -> tuple[float, float, float]:
    exploit = -abs(pivot_mu - opponent_mu)
    explore = cfg.alpha * math.sqrt(math.log(max(1, total_rounds)) / max(1, pair_matches))
    sigma_bias = opponent_sigma ** cfg.sigma_tiebreak_power
    return (round(exploit + explore, 6), sigma_bias, rng_jitter)


def apply_bayesian_update(
    winner_data: dict,
    loser_data: dict,
    cfg: RankingConfig = DEFAULT_RANKING_CONFIG,
    *,
    is_tie: bool = False,
    weight: float = 1.0,
) -> None:
    mu_w, sigma_w = winner_data["mu"], winner_data["sigma"]
    mu_l, sigma_l = loser_data["mu"], loser_data["sigma"]

    if is_tie:
        midpoint = 0.5 * (mu_w + mu_l)
        blend = cfg.tie_mean_blend
        winner_data["mu"] = (1.0 - blend) * mu_w + blend * midpoint
        loser_data["mu"] = (1.0 - blend) * mu_l + blend * midpoint
        winner_data["sigma"] = math.sqrt(max(cfg.sigma_conv**2, sigma_w**2 * cfg.tie_sigma_shrink))
        loser_data["sigma"] = math.sqrt(max(cfg.sigma_conv**2, sigma_l**2 * cfg.tie_sigma_shrink))
        return

    c12 = math.sqrt(sigma_w**2 + sigma_l**2 + 2.0 * cfg.beta**2)
    t = max(-5.0, min(5.0, (mu_w - mu_l) / c12))
    cdf_t = max(norm.cdf(t), 1e-12)
    v_val = norm.pdf(t) / cdf_t
    w_val = v_val * (v_val + t)

    winner_data["mu"] += (sigma_w**2 / c12) * v_val * weight
    loser_data["mu"] -= (sigma_l**2 / c12) * v_val * weight

    gamma = min(1.2, 1.0 + (weight - 1.0) * 0.5)
    winner_data["sigma"] = math.sqrt(
        max(cfg.sigma_conv**2, sigma_w**2 * (1.0 - (sigma_w**2 / c12**2) * w_val * gamma))
    )
    loser_data["sigma"] = math.sqrt(
        max(cfg.sigma_conv**2, sigma_l**2 * (1.0 - (sigma_l**2 / c12**2) * w_val * gamma))
    )


def resolve_micro_batch(wins_a: int, wins_b: int, ties: int, cfg: RankingConfig = DEFAULT_RANKING_CONFIG) -> dict:
    total_valid = wins_a + wins_b + ties
    if total_valid <= 0:
        return {
            "total_valid": 0,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "p_a": None,
            "macro_winner": "NoResult",
            "is_tie": True,
            "batch_weight": 1.0,
        }

    p_a = (wins_a + 0.5 * ties) / float(total_valid)
    if p_a > cfg.decisive_p_high:
        macro_winner = "A"
        p_w = p_a
        is_tie = False
    elif p_a < cfg.decisive_p_low:
        macro_winner = "B"
        p_w = 1.0 - p_a
        is_tie = False
    else:
        macro_winner = "Tie"
        p_w = 0.5
        is_tie = True

    batch_weight = min(cfg.max_weight, 1.0 + cfg.weight_scale * (p_w - 0.5))
    return {
        "total_valid": total_valid,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "p_a": p_a,
        "macro_winner": macro_winner,
        "is_tie": is_tie,
        "batch_weight": batch_weight,
    }


def pairwise_win_probability(mean_a: float, var_a: float, mean_b: float, var_b: float, eps: float = 1e-5) -> float:
    denom = math.sqrt(max(0.0, var_a) + max(0.0, var_b) + eps)
    z_val = (mean_a - mean_b) / denom
    return float(norm.cdf(z_val))


def winner_from_probability(prob_a: float, cfg: RankingConfig = DEFAULT_RANKING_CONFIG) -> str:
    if prob_a > cfg.decisive_p_high:
        return "A"
    if prob_a < cfg.decisive_p_low:
        return "B"
    return "Tie"
