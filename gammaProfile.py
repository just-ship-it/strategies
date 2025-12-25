import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import requests
import sys
import json

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed. Run 'pip install yfinance' for futures translation.")

pd.options.display.float_format = '{:,.4f}'.format

# Futures mapping
FUTURES_MAP = {
    "SPX": {"futures": "ES", "yahoo": "ES=F", "method": "spread"},
    "QQQ": {"futures": "NQ", "yahoo": "NQ=F", "method": "multiplier"},
}

def get_futures_spot(yahoo_ticker):
    """Fetch current futures price from Yahoo Finance."""
    if not HAS_YFINANCE:
        return None
    try:
        ticker = yf.Ticker(yahoo_ticker)
        # Try multiple methods to get price
        price = ticker.info.get('regularMarketPrice')
        if price is None:
            price = ticker.fast_info.get('lastPrice')
        if price is None:
            # Fallback to recent history
            hist = ticker.history(period="1d")
            if len(hist) > 0:
                price = hist['Close'].iloc[-1]
        return price
    except Exception as e:
        print(f"Warning: Could not fetch {yahoo_ticker}: {e}")
        return None

def translate_level_to_futures(level, spot_price, futures_spot, method):
    """Translate an index/ETF level to futures level."""
    if level == 0 or futures_spot is None:
        return 0
    
    if method == "spread":
        # SPX -> ES: add the spread
        spread = futures_spot - spot_price
        return int(round(level + spread))
    elif method == "multiplier":
        # QQQ -> NQ: multiply by ratio
        multiplier = futures_spot / spot_price
        return int(round(level * multiplier))
    return level

def translate_all_levels(levels_dict, spot_price, futures_spot, method):
    """Translate all levels to futures."""
    return {
        "zeroGamma": translate_level_to_futures(levels_dict["zeroGamma"], spot_price, futures_spot, method),
        "callWall": translate_level_to_futures(levels_dict["callWall"], spot_price, futures_spot, method),
        "putWall": translate_level_to_futures(levels_dict["putWall"], spot_price, futures_spot, method),
        "resistance": [translate_level_to_futures(l, spot_price, futures_spot, method) for l in levels_dict["resistance"]],
        "support": [translate_level_to_futures(l, spot_price, futures_spot, method) for l in levels_dict["support"]],
    }

# Black-Scholes European-Options Gamma
def calcGammaEx(S, K, vol, T, r, q, optType, OI):
    if T <= 0 or vol <= 0 or np.isnan(vol) or S <= 0 or K <= 0 or np.isnan(OI):
        return 0

    try:
        dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
        dm = dp - vol*np.sqrt(T) 

        if optType == 'call':
            gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
            return OI * 100 * S * S * 0.01 * gamma 
        else:
            gamma = K * np.exp(-r*T) * norm.pdf(dm) / (S * S * vol * np.sqrt(T))
            return OI * 100 * S * S * 0.01 * gamma
    except:
        return 0

def isThirdFriday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21

def findKeyLevels(dfAgg, spotPrice, n_levels=5):
    """
    Find secondary support/resistance levels based on gamma and OI concentrations.
    """
    nearMoney = dfAgg[(dfAgg.index >= spotPrice * 0.92) & (dfAgg.index <= spotPrice * 1.08)].copy()
    
    if len(nearMoney) == 0:
        return {"resistance": [], "support": []}
    
    # RESISTANCE LEVELS (above spot)
    above_spot = nearMoney[nearMoney.index > spotPrice].copy()
    
    resistance_levels = []
    if len(above_spot) > 0:
        above_spot['call_score'] = (
            above_spot['CallGEX'] / above_spot['CallGEX'].abs().max() * 0.5 +
            above_spot['CallOpenInt'] / above_spot['CallOpenInt'].max() * 0.5
        )
        top_resistance = above_spot.nlargest(n_levels, 'call_score')
        for strike in sorted(top_resistance.index):
            resistance_levels.append({
                "strike": int(strike),
                "callOI": int(above_spot.loc[strike, 'CallOpenInt']),
                "callGamma": round(above_spot.loc[strike, 'CallGEX'] / 1e9, 2),
                "type": "call_wall"
            })
    
    # SUPPORT LEVELS (below spot)
    below_spot = nearMoney[nearMoney.index < spotPrice].copy()
    
    support_levels = []
    if len(below_spot) > 0:
        below_spot['put_score'] = (
            below_spot['PutGEX'].abs() / below_spot['PutGEX'].abs().max() * 0.5 +
            below_spot['PutOpenInt'] / below_spot['PutOpenInt'].max() * 0.5
        )
        top_support = below_spot.nlargest(n_levels, 'put_score')
        for strike in sorted(top_support.index, reverse=True):
            support_levels.append({
                "strike": int(strike),
                "putOI": int(below_spot.loc[strike, 'PutOpenInt']),
                "putGamma": round(abs(below_spot.loc[strike, 'PutGEX']) / 1e9, 2),
                "type": "put_wall"
            })
    
    # GAMMA MAGNETS
    nearMoney['absGamma'] = nearMoney['TotalGamma'].abs()
    top_gamma = nearMoney.nlargest(n_levels, 'absGamma')
    
    gamma_magnets = []
    for strike in top_gamma.index:
        gamma_magnets.append({
            "strike": int(strike),
            "totalGamma": round(nearMoney.loc[strike, 'TotalGamma'], 2),
            "netGammaType": "positive" if nearMoney.loc[strike, 'TotalGamma'] > 0 else "negative"
        })
    
    return {
        "resistance": resistance_levels,
        "support": support_levels,
        "gammaMagnets": sorted(gamma_magnets, key=lambda x: abs(x['totalGamma']), reverse=True)
    }

# ============================================================================
# MAIN SCRIPT
# ============================================================================

# Parse command line args
index = "SPX"
json_only = False
tv_output = False
no_charts = False

for arg in sys.argv[1:]:
    if arg.startswith("--"):
        if arg == "--json":
            json_only = True
        elif arg == "--tv":
            tv_output = True
        elif arg == "--no-charts":
            no_charts = True
    else:
        index = arg.upper()

print(f"Fetching {index} options data from CBOE...")

# CBOE uses underscore prefix for indices, no prefix for ETFs
INDEX_SYMBOLS = ["SPX", "NDX", "VIX", "RUT", "DJX", "OEX", "XSP"]  # Index options
cboe_symbol = f"_{index}" if index in INDEX_SYMBOLS else index

# Get options data
url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{cboe_symbol}.json"
response = requests.get(url=url)

if response.status_code != 200:
    print(f"Error: CBOE API returned status {response.status_code}")
    print(f"URL: {url}")
    sys.exit(1)

try:
    options = response.json()
except Exception as e:
    print(f"Error parsing CBOE response: {e}")
    print(f"Response text: {response.text[:500]}")
    sys.exit(1)

# Extract timestamp info
data_timestamp = options["data"].get("current_datetime") or options["data"].get("last_trade_time") or "unknown"
seqno = options["data"].get("seqno", "")

# Get Spot Price
spotPrice = options["data"]["close"]
print(f"Spot Price: {spotPrice}")
fromStrike = 0.8 * spotPrice
toStrike = 1.2 * spotPrice

todayDate = date.today()

# Process options data
data_df = pd.DataFrame(options["data"]["options"])

data_df['CallPut'] = data_df['option'].str.slice(start=-9,stop=-8)
data_df['ExpirationDate'] = data_df['option'].str.slice(start=-15,stop=-9)
data_df['ExpirationDate'] = pd.to_datetime(data_df['ExpirationDate'], format='%y%m%d')
data_df['Strike'] = data_df['option'].str.slice(start=-8,stop=-3)
data_df['Strike'] = data_df['Strike'].str.lstrip('0')

data_df_calls = data_df.loc[data_df['CallPut'] == "C"]
data_df_puts = data_df.loc[data_df['CallPut'] == "P"]
data_df_calls = data_df_calls.reset_index(drop=True)
data_df_puts = data_df_puts.reset_index(drop=True)

df = data_df_calls[['ExpirationDate','option','last_trade_price','change','bid','ask','volume','iv','delta','gamma','open_interest','Strike']]
df_puts = data_df_puts[['ExpirationDate','option','last_trade_price','change','bid','ask','volume','iv','delta','gamma','open_interest','Strike']]
df_puts.columns = ['put_exp','put_option','put_last_trade_price','put_change','put_bid','put_ask','put_volume','put_iv','put_delta','put_gamma','put_open_interest','put_strike']

df = pd.concat([df, df_puts], axis=1)

df['check'] = np.where((df['ExpirationDate'] == df['put_exp']) & (df['Strike'] == df['put_strike']), 0, 1)

if df['check'].sum() != 0:
    print("PUT CALL MERGE FAILED - OPTIONS ARE MISMATCHED.")
    exit()

df.drop(['put_exp', 'put_strike', 'check'], axis=1, inplace=True)

df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
              'CallIV','CallDelta','CallGamma','CallOpenInt','StrikePrice','Puts','PutLastSale',
              'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutOpenInt']

df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y', errors='coerce')
df['ExpirationDate'] = df['ExpirationDate'] + timedelta(hours=16)
df['StrikePrice'] = pd.to_numeric(df['StrikePrice'], errors='coerce')
df['CallIV'] = pd.to_numeric(df['CallIV'], errors='coerce').fillna(0)
df['PutIV'] = pd.to_numeric(df['PutIV'], errors='coerce').fillna(0)
df['CallGamma'] = pd.to_numeric(df['CallGamma'], errors='coerce').fillna(0)
df['PutGamma'] = pd.to_numeric(df['PutGamma'], errors='coerce').fillna(0)
df['CallOpenInt'] = pd.to_numeric(df['CallOpenInt'], errors='coerce').fillna(0)
df['PutOpenInt'] = pd.to_numeric(df['PutOpenInt'], errors='coerce').fillna(0)

df = df[df['StrikePrice'] > 0]
df = df[df['ExpirationDate'].notna()]

print(f"Processing {len(df)} options...")

# Calculate spot gamma
df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1

df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 10**9
dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)
strikes = dfAgg.index.values

# Calculate key levels
totalGammaValue = df['TotalGamma'].sum()
callWall = int(dfAgg['CallOpenInt'].idxmax())
putWall = int(dfAgg['PutOpenInt'].idxmax())

# Get secondary levels
keyLevels = findKeyLevels(dfAgg, spotPrice, n_levels=5)

# Pad levels to ensure we have 5 of each
resistance = [l['strike'] for l in keyLevels['resistance']]
support = [l['strike'] for l in keyLevels['support']]

while len(resistance) < 5:
    resistance.append(0)
while len(support) < 5:
    support.append(0)

resistance = resistance[:5]
support = support[:5]

# Calculate gamma profile for zero gamma
print(f"Calculating gamma profile...")
levels = np.linspace(fromStrike, toStrike, 30)

df['daysTillExp'] = [1/262 if (np.busday_count(todayDate, x.date())) == 0 \
                           else np.busday_count(todayDate, x.date())/262 for x in df.ExpirationDate]

nextExpiry = df['ExpirationDate'].min()

df['IsThirdFriday'] = [isThirdFriday(x) for x in df.ExpirationDate]
thirdFridays = df.loc[df['IsThirdFriday'] == True]
nextMonthlyExp = thirdFridays['ExpirationDate'].min() if len(thirdFridays) > 0 else nextExpiry

totalGamma = []
totalGammaExNext = []
totalGammaExFri = []

for i, level in enumerate(levels):
    df['callGammaEx'] = df.apply(lambda row : calcGammaEx(level, row['StrikePrice'], row['CallIV'], 
                                                          row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis = 1)

    df['putGammaEx'] = df.apply(lambda row : calcGammaEx(level, row['StrikePrice'], row['PutIV'], 
                                                         row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis = 1)    

    totalGamma.append(df['callGammaEx'].sum() - df['putGammaEx'].sum())

    exNxt = df.loc[df['ExpirationDate'] != nextExpiry]
    totalGammaExNext.append(exNxt['callGammaEx'].sum() - exNxt['putGammaEx'].sum())

    exFri = df.loc[df['ExpirationDate'] != nextMonthlyExp]
    totalGammaExFri.append(exFri['callGammaEx'].sum() - exFri['putGammaEx'].sum())

totalGamma = np.array(totalGamma) / 10**9
totalGammaExNext = np.array(totalGammaExNext) / 10**9
totalGammaExFri = np.array(totalGammaExFri) / 10**9

# Find Gamma Flip Point
zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]

if len(zeroCrossIdx) == 0:
    if totalGamma[0] > 0:
        zeroGamma = 0  # Positive everywhere
    else:
        zeroGamma = 0  # Negative everywhere
else:
    negGamma = totalGamma[zeroCrossIdx]
    posGamma = totalGamma[zeroCrossIdx+1]
    negStrike = levels[zeroCrossIdx]
    posStrike = levels[zeroCrossIdx+1]
    zeroGamma = posStrike - ((posStrike - negStrike) * posGamma/(posGamma-negGamma))
    zeroGamma = int(round(zeroGamma[0]))

gammaStatus = "positive" if totalGammaValue > 0 else "negative"

# ============================================================================
# OUTPUT
# ============================================================================

# Prepare levels dict for translation
index_levels = {
    "zeroGamma": zeroGamma,
    "callWall": callWall,
    "putWall": putWall,
    "resistance": resistance,
    "support": support,
}

# Get futures translation if available
futures_spot = None
futures_levels = None
futures_name = None
translation_info = ""

if index in FUTURES_MAP and HAS_YFINANCE:
    futures_config = FUTURES_MAP[index]
    futures_name = futures_config["futures"]
    yahoo_ticker = futures_config["yahoo"]
    method = futures_config["method"]
    
    print(f"\nFetching {futures_name} futures price...")
    futures_spot = get_futures_spot(yahoo_ticker)
    
    if futures_spot:
        if method == "spread":
            spread = futures_spot - spotPrice
            translation_info = f"spread: {spread:+.1f}"
        else:
            multiplier = futures_spot / spotPrice
            translation_info = f"multiplier: {multiplier:.2f}x"
        
        futures_levels = translate_all_levels(index_levels, spotPrice, futures_spot, method)
        print(f"{futures_name} Spot: {futures_spot:,.2f} ({translation_info})")

print(f"\n{'='*60}")
print(f"GEX SUMMARY FOR {index} - {todayDate.strftime('%Y-%m-%d')}")
print(f"{'='*60}")
print(f"Data Timestamp:  {data_timestamp}")
print(f"Spot Price:      {spotPrice:,.2f}")
if futures_spot:
    print(f"{futures_name} Spot:       {futures_spot:,.2f} ({translation_info})")
print(f"Total Gamma:     ${totalGammaValue:,.2f} Bn per 1% move")
print(f"Gamma Status:    {'POSITIVE ✓ (stabilizing)' if totalGammaValue > 0 else 'NEGATIVE ✗ (volatile)'}")
print(f"Zero Gamma:      {zeroGamma if zeroGamma > 0 else 'Not in range (gamma ' + gammaStatus + ' everywhere)'}")

print(f"\nPRIMARY LEVELS:")
print(f"  Call Wall:     {callWall:,}")
print(f"  Put Wall:      {putWall:,}")

print(f"\nRESISTANCE LEVELS (above spot):")
for i, level in enumerate(keyLevels['resistance'][:5]):
    print(f"  R{i+1}: {level['strike']:,} - Call OI: {level['callOI']:,}, Gamma: ${level['callGamma']:.2f}Bn")

print(f"\nSUPPORT LEVELS (below spot):")
for i, level in enumerate(keyLevels['support'][:5]):
    print(f"  S{i+1}: {level['strike']:,} - Put OI: {level['putOI']:,}, Gamma: ${level['putGamma']:.2f}Bn")

# ============================================================================
# TRADINGVIEW CSV OUTPUT
# ============================================================================

# Format: gammaStatus,totalGamma,zeroGamma,callWall,putWall,r1,r2,r3,r4,r5,s1,s2,s3,s4,s5
tv_csv = f"{gammaStatus},{totalGammaValue:.2f},{zeroGamma},{callWall},{putWall},{resistance[0]},{resistance[1]},{resistance[2]},{resistance[3]},{resistance[4]},{support[0]},{support[1]},{support[2]},{support[3]},{support[4]}"

print(f"\n{'='*60}")
print(f"TRADINGVIEW CSV - {index} (copy and paste into indicator):")
print(f"{'='*60}")
print(tv_csv)

# Futures CSV output
if futures_levels:
    fl = futures_levels
    futures_tv_csv = f"{gammaStatus},{totalGammaValue:.2f},{fl['zeroGamma']},{fl['callWall']},{fl['putWall']},{fl['resistance'][0]},{fl['resistance'][1]},{fl['resistance'][2]},{fl['resistance'][3]},{fl['resistance'][4]},{fl['support'][0]},{fl['support'][1]},{fl['support'][2]},{fl['support'][3]},{fl['support'][4]}"
    
    print(f"\n{'='*60}")
    print(f"TRADINGVIEW CSV - {futures_name} (copy and paste into indicator):")
    print(f"{'='*60}")
    print(futures_tv_csv)

print(f"{'='*60}")

if tv_output:
    # Also save to file
    tv_filename = f"gex_levels_{index}_{todayDate.strftime('%Y%m%d')}.txt"
    with open(tv_filename, 'w') as f:
        f.write(f"# GEX Levels for {index} - {todayDate.strftime('%Y-%m-%d')}\n")
        f.write(f"# Data Timestamp: {data_timestamp}\n")
        f.write(f"# Total Gamma: ${totalGammaValue:.2f}Bn ({gammaStatus})\n")
        f.write(f"# Spot: {spotPrice}\n")
        if futures_spot:
            f.write(f"# {futures_name} Spot: {futures_spot} ({translation_info})\n")
        f.write(f"#\n")
        f.write(f"# CSV Format: gammaStatus,totalGamma,zeroGamma,callWall,putWall,r1,r2,r3,r4,r5,s1,s2,s3,s4,s5\n")
        f.write(f"#\n")
        f.write(f"# {index} levels:\n")
        f.write(tv_csv + "\n")
        if futures_levels:
            f.write(f"#\n")
            f.write(f"# {futures_name} levels:\n")
            f.write(futures_tv_csv + "\n")
    print(f"\nSaved to: {tv_filename}")

# JSON output
if json_only:
    summary = {
        "index": index,
        "date": todayDate.strftime('%Y-%m-%d'),
        "dataTimestamp": data_timestamp,
        "spotPrice": spotPrice,
        "totalGamma": round(totalGammaValue, 2),
        "gammaStatus": gammaStatus,
        "zeroGamma": zeroGamma if zeroGamma > 0 else None,
        "callWall": callWall,
        "putWall": putWall,
        "resistance": resistance,
        "support": support,
        "tvCSV": tv_csv
    }
    
    if futures_levels and futures_spot:
        summary["futures"] = {
            "symbol": futures_name,
            "spot": futures_spot,
            "translation": translation_info,
            "zeroGamma": futures_levels["zeroGamma"] if futures_levels["zeroGamma"] > 0 else None,
            "callWall": futures_levels["callWall"],
            "putWall": futures_levels["putWall"],
            "resistance": futures_levels["resistance"],
            "support": futures_levels["support"],
            "tvCSV": futures_tv_csv
        }
    
    print(f"\nJSON OUTPUT:")
    print(json.dumps(summary, indent=2))

# ============================================================================
# CHARTS
# ============================================================================

if not json_only and not no_charts:
    # Chart 1: Gamma Exposure with levels
    plt.figure(figsize=(14, 7))
    plt.grid()
    plt.bar(strikes, dfAgg['TotalGamma'].to_numpy(), width=6, linewidth=0.1, edgecolor='k', label="Gamma Exposure", alpha=0.7)
    plt.xlim([fromStrike, toStrike])
    chartTitle = f"Total Gamma: ${totalGammaValue:.2f} Bn per 1% {index} Move"
    plt.title(chartTitle, fontweight="bold", fontsize=16)
    plt.xlabel('Strike', fontweight="bold")
    plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
    
    plt.axvline(x=spotPrice, color='r', lw=2, label=f"Spot: {spotPrice:,.0f}")
    
    # Resistance levels
    for i, strike in enumerate(resistance[:3]):
        if strike > 0:
            plt.axvline(x=strike, color='orange', lw=1, linestyle='--', alpha=0.7,
                        label=f"Resistance" if i == 0 else "")
    
    # Support levels
    for i, strike in enumerate(support[:3]):
        if strike > 0:
            plt.axvline(x=strike, color='green', lw=1, linestyle='--', alpha=0.7,
                        label=f"Support" if i == 0 else "")
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Chart 2: Gamma Profile
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.grid()
    plt.plot(levels, totalGamma, label="All Expiries", linewidth=2, color='blue')
    plt.plot(levels, totalGammaExNext, label="Ex-Next Expiry", linewidth=1, alpha=0.6)
    plt.plot(levels, totalGammaExFri, label="Ex-Next Monthly", linewidth=1, alpha=0.6)
    
    chartTitle = f"Gamma Exposure Profile - {index} - {todayDate.strftime('%d %b %Y')}"
    plt.title(chartTitle, fontweight="bold", fontsize=16)
    plt.xlabel('Index Price', fontweight="bold")
    plt.ylabel('Gamma Exposure ($ billions/1% move)', fontweight="bold")
    
    plt.axvline(x=spotPrice, color='r', lw=2, label=f"Spot: {spotPrice:,.0f}")
    if zeroGamma > 0:
        plt.axvline(x=zeroGamma, color='g', lw=2, label=f"Zero Gamma: {zeroGamma:,}")
    plt.axhline(y=0, color='grey', lw=1)
    
    plt.xlim([fromStrike, toStrike])
    trans = ax.get_xaxis_transform()
    if zeroGamma > 0:
        plt.fill_between([fromStrike, zeroGamma], min(totalGamma), max(totalGamma), facecolor='red', alpha=0.1, transform=trans)
        plt.fill_between([zeroGamma, toStrike], min(totalGamma), max(totalGamma), facecolor='green', alpha=0.1, transform=trans)
    elif gammaStatus == "positive":
        plt.fill_between([fromStrike, toStrike], min(totalGamma), max(totalGamma), facecolor='green', alpha=0.1, transform=trans)
    else:
        plt.fill_between([fromStrike, toStrike], min(totalGamma), max(totalGamma), facecolor='red', alpha=0.1, transform=trans)
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

print("\nDone!")
