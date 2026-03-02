from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import datetime
import time
import threading

app = Flask(__name__)
CORS(app)

# Simple in-memory cache to avoid hammering Yahoo Finance
_cache = {}
_cache_lock = threading.Lock()
CACHE_TTL = 300  # seconds (5 minutes)

def cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry['ts'] < CACHE_TTL):
            return entry['data']
    return None

def cache_set(key, data):
    with _cache_lock:
        _cache[key] = {'data': data, 'ts': time.time()}

TICKER_ALIASES = {
    'SPX': '^GSPC',
    'NDX': '^NDX',
    'RUT': '^RUT',
    'VIX': '^VIX',
    'DJI': '^DJI',
    'SPY': 'SPY',
    'QQQ': 'QQQ',
}

def bs_price(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def calc_iv(mkt, S, K, T, r, opt):
    if T <= 0 or mkt <= 0: return np.nan
    try:
        return brentq(lambda s: bs_price(S,K,T,r,s,opt)-mkt, 1e-6, 10.0, xtol=1e-6, maxiter=500)
    except:
        return np.nan

def calc_greeks(S, K, T, r, sigma, opt):
    if T <= 0 or sigma <= 0:
        return dict(delta=0, gamma=0, theta=0, vega=0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    if opt == 'call':
        delta = norm.cdf(d1)
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
    return dict(delta=round(delta,4), gamma=round(gamma,4), theta=round(theta,4), vega=round(vega,4))

def fetch_with_retry(func, retries=3, delay=2):
    """Call func(), retrying on rate-limit or transient errors."""
    for attempt in range(retries):
        try:
            result = func()
            return result
        except Exception as e:
            msg = str(e).lower()
            if 'rate' in msg or '429' in msg or 'too many' in msg:
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
            raise
    return None

@app.route('/scan', methods=['POST'])
def scan():
    try:
        data = request.json
        raw_ticker = data.get('ticker','SPY').upper().strip()
        ticker = TICKER_ALIASES.get(raw_ticker, raw_ticker)
        start_str = data.get('start_date')
        end_str   = data.get('end_date')
        opt_type  = data.get('option_type','call').lower()
        rfr = float(data.get('risk_free_rate', 4.5)) / 100

        start_dt = datetime.date.fromisoformat(start_str)
        end_dt   = datetime.date.fromisoformat(end_str)

        # --- Fetch spot price (cached) ---
        price_key = f'price:{ticker}'
        S = cache_get(price_key)
        if S is None:
            stock = yf.Ticker(ticker)
            hist = fetch_with_retry(lambda: stock.history(period='5d'))
            if hist is None or hist.empty:
                return jsonify({'error': f'No price data for {raw_ticker}'}), 400
            S = float(hist['Close'].iloc[-1])
            cache_set(price_key, S)
        else:
            stock = yf.Ticker(ticker)

        # --- Fetch expiration list (cached) ---
        exp_key = f'exps:{ticker}'
        expirations = cache_get(exp_key)
        if expirations is None:
            expirations = fetch_with_retry(lambda: stock.options)
            if not expirations:
                return jsonify({'error': f'No options data for {raw_ticker}'}), 400
            cache_set(exp_key, expirations)

        candidates = []
        today = datetime.date.today()

        for exp_str in expirations:
            exp_dt = datetime.date.fromisoformat(exp_str)
            if not (start_dt <= exp_dt <= end_dt):
                continue
            T = (exp_dt - today).days / 365.0
            if T <= 0:
                continue

            # --- Fetch option chain (cached per expiry) ---
            chain_key = f'chain:{ticker}:{exp_str}:{opt_type}'
            opts_data = cache_get(chain_key)
            if opts_data is None:
                try:
                    time.sleep(0.3)  # polite delay between chain fetches
                    chain = fetch_with_retry(lambda e=exp_str: stock.option_chain(e))
                    if chain is None:
                        continue
                    opts = chain.calls if opt_type == 'call' else chain.puts
                    opts_data = opts[['strike','lastPrice','volume']].to_dict('records')
                    cache_set(chain_key, opts_data)
                except Exception:
                    continue

            for row in opts_data:
                K   = float(row['strike'])
                mkt = float(row.get('lastPrice', 0))
                vol = float(row.get('volume', 0) or 0)
                sigma = calc_iv(mkt, S, K, T, rfr, opt_type)
                if np.isnan(sigma) or sigma <= 0:
                    continue
                g = calc_greeks(S, K, T, rfr, sigma, opt_type)
                delta = g['delta']
                gamma = g['gamma']
                if opt_type == 'call':
                    if not (0.35 <= delta <= 0.40): continue
                else:
                    if not (-0.40 <= delta <= -0.35): continue
                gamma_diff = abs(gamma - 0.025)
                candidates.append({
                    'expiration': exp_str,
                    'strike': K,
                    'price': round(mkt, 3),
                    'delta': delta,
                    'gamma': gamma,
                    'theta': g['theta'],
                    'vega':  g['vega'],
                    'iv_pct': round(sigma * 100, 2),
                    'volume': int(vol),
                    'gamma_diff': gamma_diff
                })

        if not candidates:
            return jsonify({'error': f'No options matched delta 0.35-0.40 / -0.40 to -0.35 for {raw_ticker} in that date range. Try widening the date range.'}), 200

        candidates.sort(key=lambda x: x['gamma_diff'])
        top5 = candidates[:5]
        for c in top5:
            del c['gamma_diff']

        return jsonify({'results': top5, 'ticker': raw_ticker, 'spot': round(S,2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def health():
    return 'Options Scanner API is running.'

if __name__ == '__main__':
    app.run(debug=False)
