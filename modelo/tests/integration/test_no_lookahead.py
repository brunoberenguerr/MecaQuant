"""O teste definitivo contra look-ahead.

Ideia: se a estratégia realmente só usa informação passada, então apagar todo o
futuro depois de uma data D não pode alterar NENHUMA decisão tomada até D.

Rodamos o backtest duas vezes — uma com o painel completo, outra com tudo após D
substituído por NaN — e exigimos pesos bit-idênticos até D. Se qualquer módulo
espiar o futuro, os pesos divergem e o teste quebra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mare.backtest.asof import LookAheadError, MarketData
from mare.backtest.engine import run_backtest
from mare.config import load_config
from mare.io.universe import Membership
from mare.portfolio.construct import make_signal_fn
from mare.signal.naive import candidate_table_naive

N_DAYS, N_SYMBOLS = 900, 70
CUTOFF = 700


@pytest.fixture(scope="module")
def config():
    """Config reduzida para o teste rodar rápido sem mudar a lógica."""
    return load_config(**{
        "factor.window": 250, "factor.halflife": 60,
        "factor.k_min": 3, "factor.k_max": 10,
        "residual.beta_window": 250, "residual.beta_gap": 40,
        "residual.resid_window": 40,
        "ou.halflife_min": 5, "ou.halflife_max": 30,
        "portfolio.n_names": 10,
        "universe.min_dollar_volume": 0, "universe.min_price": 0,
        "universe.min_valid_returns": 100, "universe.adv_window": 20,
    })


@pytest.fixture(scope="module")
def market() -> MarketData:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2015-01-01", periods=N_DAYS)
    symbols = [f"S{i:02d}" for i in range(N_SYMBOLS)]

    market_factor = rng.standard_normal((N_DAYS, 1)) * 0.010
    loads = rng.uniform(0.6, 1.4, (1, N_SYMBOLS))
    idio = rng.standard_normal((N_DAYS, N_SYMBOLS)) * 0.012
    # Componente idiossincrática com reversão, para haver sinal a capturar.
    for t in range(1, N_DAYS):
        idio[t] += -0.05 * idio[t - 1]

    returns = pd.DataFrame(market_factor @ loads + idio, index=dates, columns=symbols)
    close = (1 + returns).cumprod() * 50.0
    volume = pd.DataFrame(1e7, index=dates, columns=symbols)
    members = Membership(((pd.Timestamp.min + pd.Timedelta(days=1),
                           frozenset(symbols)),))
    return MarketData(returns=returns, close_raw=close, volume=volume,
                      membership=members)


def _truncate(market: MarketData, cutoff: pd.Timestamp) -> MarketData:
    """Apaga todo dado após `cutoff`, preservando o índice de datas."""
    def blank(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out.loc[out.index > cutoff] = np.nan
        return out

    return MarketData(returns=blank(market.returns),
                      close_raw=blank(market.close_raw),
                      volume=blank(market.volume),
                      membership=market.membership)


def test_weights_are_identical_when_future_is_erased(market, config):
    """Nenhuma decisão até D pode depender de dado posterior a D."""
    cutoff = market.dates[CUTOFF]

    full = run_backtest(market, config, make_signal_fn(), end=str(market.dates[-1].date()))
    truncated = run_backtest(_truncate(market, cutoff), config, make_signal_fn(),
                             end=str(cutoff.date()))

    common = truncated.weights.index[truncated.weights.index <= cutoff]
    assert len(common) >= 5, "poucos rebalanceamentos para o teste ser informativo"

    a = full.weights.reindex(common).fillna(0.0)
    b = truncated.weights.reindex(common).fillna(0.0)
    a, b = a.align(b, axis=1, fill_value=0.0)

    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_returns_are_identical_before_cutoff(market, config):
    """Consequência do anterior: a curva de patrimônio até D também coincide."""
    cutoff = market.dates[CUTOFF]
    full = run_backtest(market, config, make_signal_fn())
    truncated = run_backtest(_truncate(market, cutoff), config, make_signal_fn(),
                             end=str(cutoff.date()))

    common = truncated.returns.index[truncated.returns.index < cutoff]
    np.testing.assert_allclose(full.returns.reindex(common).to_numpy(),
                               truncated.returns.reindex(common).to_numpy(),
                               rtol=0, atol=0)


def test_view_refuses_future_access(market):
    """Consulta explícita ao futuro levanta exceção, não devolve NaN.

    NaN silencioso vira zero em algum lugar e desaparece do relatório.
    """
    view = market.view(market.dates[300])
    view.at(market.dates[300])                      # presente: ok
    view.at(market.dates[299])                      # passado: ok
    with pytest.raises(LookAheadError):
        view.at(market.dates[301])


def test_view_is_physically_sliced(market):
    """A view não CONTÉM o futuro — não é questão de não olhar."""
    as_of = market.dates[300]
    view = market.view(as_of)
    assert view.returns.index.max() == as_of
    assert view.close_raw.index.max() == as_of
    assert view.volume.index.max() == as_of


def test_schedule_never_trades_on_signal_date(market, config):
    """O lag entre sinal e execução é estrutural, não convenção."""
    from mare.calendar import schedule

    plan = schedule(market.dates, weekday=config.backtest.rebalance_weekday,
                    lag_days=config.backtest.signal_lag_days)
    assert (plan.signal_date < plan.trade_date).all()


def test_determinism(market, config):
    """Mesmo config e mesmo dado devem dar exatamente o mesmo resultado."""
    first = run_backtest(market, config, make_signal_fn())
    second = run_backtest(market, config, make_signal_fn())
    assert first.fingerprint() == second.fingerprint()


def test_naive_baseline_has_no_lookahead(market, config):
    """O baseline de reversão simples (sem PCA/resíduo) é a mesma checagem.

    `table_fn` troca só a construção da tabela de candidatos; o resto do
    pipeline (livro, sizing) é idêntico ao sinal principal. Precisa da mesma
    garantia estrutural.
    """
    cutoff = market.dates[CUTOFF]
    signal_fn = make_signal_fn(table_fn=candidate_table_naive)

    full = run_backtest(market, config, signal_fn, end=str(market.dates[-1].date()))
    truncated = run_backtest(_truncate(market, cutoff), config,
                             make_signal_fn(table_fn=candidate_table_naive),
                             end=str(cutoff.date()))

    common = truncated.weights.index[truncated.weights.index <= cutoff]
    assert len(common) >= 5, "poucos rebalanceamentos para o teste ser informativo"

    a = full.weights.reindex(common).fillna(0.0)
    b = truncated.weights.reindex(common).fillna(0.0)
    a, b = a.align(b, axis=1, fill_value=0.0)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
