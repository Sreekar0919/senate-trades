"""
SenateTrades v3 — Central Configuration
All weights, thresholds, and API settings live here.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────
ALPACA_API_KEY     = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY  = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL    = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
QUIVER_API_KEY     = os.getenv("QUIVER_API_KEY", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
POLYGON_API_KEY    = os.getenv("POLYGON_API_KEY", "")   # optional, for intraday data

# ─────────────────────────────────────────────────────────────
# Data windows and CSV paths
# ─────────────────────────────────────────────────────────────
DISCLOSURE_LOOKBACK_DAYS    = 6000   # capture all historical data (back to 2010)
PREDICTION_LOOKBACK_DAYS    = 30    # hearing / contract window for prediction layer
PERFORMANCE_LOOKBACK_MONTHS = 24    # how many months of trades to score a senator
ALPHA_WINDOW_DAYS           = 30    # days forward to measure return after a trade
ALPHA_DECAY_HALFLIFE_MONTHS = 6     # recent trades weighted 2x every 6 months

# CSV Data Paths
TRADE_DATA_CSV = "data/Copy of congress-trading-all (3).csv"
PRICE_DATA_CSV = "data/stock_prices_2023_to_now.csv"

# ─────────────────────────────────────────────────────────────
# Signal scoring weights (V4.1 Balanced)
# ─────────────────────────────────────────────────────────────
W_CROWD_SCORE       = 0.10
W_PERF_SCORE        = 0.25
W_COMMITTEE_SCORE   = 0.20
W_PREDICTION_SCORE  = 0.10
W_TIES_SCORE        = 0.20
W_MOMENTUM_SCORE    = 0.15

# ─────────────────────────────────────────────────────────────
# V4 Additive Bonuses (Base + Base * sum(Bonuses))
# ─────────────────────────────────────────────────────────────
B_TOP_TRADER       = 0.20
B_CHAIR            = 0.15
B_SUBCOMMITTEE     = 0.12
B_PRE_HEARING      = 0.18
B_CONTRACT         = 0.20
B_TRANSCRIPT       = 0.15
BONUS_POOL_CAP     = 0.60

# ─────────────────────────────────────────────────────────────
# Signal thresholds
# ─────────────────────────────────────────────────────────────
MIN_SIGNAL_SCORE        = 0.35   # Buy threshold
MIN_SELL_SIGNAL_SCORE   = 0.32   # Sell threshold (lower as per description)
MIN_SENATORS_TO_QUALIFY = 1

# ─────────────────────────────────────────────────────────────
# TiesAgent: State-Company Mappings
# ─────────────────────────────────────────────────────────────
STATE_COMPANY_TIES = {
    "Washington": ["MSFT", "AMZN", "BA", "COST", "SBUX"],
    "California": ["AAPL", "GOOGL", "META", "NVDA", "INTC", "CRM", "NFLX", "TSLA"],
    "Texas": ["XOM", "CVX", "COP", "TSLA", "ORCL", "AAL"],
    "Virginia": ["LMT", "GD", "RTX", "BAH", "SAIC"],
    "Georgia": ["GD", "RTX", "DAL", "UPS", "HD"],
    "Mississippi": ["HII"],
    "Maryland": ["LMT", "MAR"],
    "New York": ["JPM", "GS", "MS", "C", "AXP", "IBM"],
    "Massachusetts": ["RTX", "MRNA", "STT"],
    "Pennsylvania": ["CMCSA", "VLO"],
    "Illinois": ["BA", "CAT", "ABBV", "ADM"],
    "Florida": ["LMT", "RTX", "NCLH", "RCL"],
}

# ─────────────────────────────────────────────────────────────
# Senator → State Mapping (Inferred from CSV or static for key senators)
# ─────────────────────────────────────────────────────────────
SENATOR_STATES = {
    "Tommy Tuberville": "Alabama",
    "Mark Warner": "Virginia",
    "Marco Rubio": "Florida",
    "Roger Wicker": "Mississippi",
    "Jack Reed": "Rhode Island",
    "Maria Cantwell": "Washington",
    "Joe Manchin": "West Virginia",
    "Tim Scott": "South Carolina",
    "Elizabeth Warren": "Massachusetts",
    "Shelley Moore Capito": "West Virginia",
    "Nancy Pelosi": "California",
    "Thomas Carper": "Delaware",
    "Suzan Delbene": "Washington",
    "Josh Gottheimer": "New Jersey",
}


# ─────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────
BASE_ORDER_NOTIONAL     = 300    # USD — scaled up by Kelly
MAX_ORDER_NOTIONAL      = 2000   # USD cap per position
MAX_POSITIONS           = 12
MAX_SECTOR_EXPOSURE     = 0.35   # max fraction of portfolio in one sector
ENTRY_DELAY_MINUTES     = 35     # wait after open before entering (avoid open pop)
USE_KELLY_SIZING        = True
KELLY_FRACTION          = 0.25   # fractional Kelly (conservative)

# ─────────────────────────────────────────────────────────────
# Sector → ETF benchmark (for alpha calculation)
# ─────────────────────────────────────────────────────────────
SECTOR_ETF: dict[str, str] = {
    "defense":    "XAR",
    "aerospace":  "XAR",
    "finance":    "XLF",
    "tech":       "XLK",
    "energy":     "XLE",
    "health":     "XLV",
    "telecom":    "XLC",
    "materials":  "XLB",
    "utilities":  "XLU",
    "real_estate":"XLRE",
    "consumer":   "XLY",
}

# ─────────────────────────────────────────────────────────────
# Ticker → sector classification
# ─────────────────────────────────────────────────────────────
TICKER_SECTORS: dict[str, str] = {
    # Defense / Aerospace
    "LMT":"defense","RTX":"defense","NOC":"defense","GD":"defense",
    "BA":"aerospace","HII":"defense","KTOS":"defense","CACI":"defense",
    "SAIC":"defense","BAH":"defense","LDOS":"defense","PLTR":"defense",
    "RCAT":"defense","ACHR":"aerospace","JOBY":"aerospace",
    # Finance
    "JPM":"finance","BAC":"finance","GS":"finance","MS":"finance",
    "WFC":"finance","C":"finance","BLK":"finance","SCHW":"finance",
    "V":"finance","MA":"finance","AXP":"finance","COF":"finance",
    # Tech
    "AAPL":"tech","MSFT":"tech","GOOGL":"tech","GOOG":"tech","META":"tech",
    "NVDA":"tech","AMZN":"tech","CRM":"tech","ORCL":"tech","INTC":"tech",
    "AMD":"tech","QCOM":"tech","CSCO":"tech","NOW":"tech","PANW":"tech",
    "CRWD":"tech","ZS":"tech","FTNT":"tech","NET":"tech","DDOG":"tech",
    # Energy
    "XOM":"energy","CVX":"energy","COP":"energy","SLB":"energy",
    "NEE":"energy","BP":"energy","OXY":"energy","MPC":"energy",
    "VLO":"energy","PSX":"energy","EOG":"energy","PXD":"energy",
    # Health / Pharma / Biotech
    "JNJ":"health","PFE":"health","MRK":"health","ABBV":"health",
    "UNH":"health","CVS":"health","AMGN":"health","MRNA":"health",
    "GILD":"health","BIIB":"health","REGN":"health","LLY":"health",
    "BMY":"health","HCA":"health","CI":"health","ELV":"health",
    # Telecom / Media
    "T":"telecom","VZ":"telecom","TMUS":"telecom","CHTR":"telecom",
    "DIS":"telecom","NFLX":"telecom","CMCSA":"telecom",
    # Materials / Industrials
    "NEM":"materials","FCX":"materials","AA":"materials",
    "CAT":"materials","DE":"materials","HON":"materials","MMM":"materials",
}

# ─────────────────────────────────────────────────────────────
# Committee → sector relevance
# ─────────────────────────────────────────────────────────────
COMMITTEE_SECTORS: dict[str, list[str]] = {
    "Armed Services":                           ["defense","aerospace"],
    "Intelligence":                             ["defense","tech","telecom"],
    "Finance":                                  ["finance"],
    "Banking, Housing, and Urban Affairs":      ["finance","real_estate"],
    "Commerce, Science, and Transportation":    ["tech","telecom","aerospace"],
    "Energy and Natural Resources":             ["energy"],
    "Environment and Public Works":             ["energy","materials","utilities"],
    "Health, Education, Labor, and Pensions":   ["health"],
    "Appropriations":                           ["defense","energy","health"],
    "Foreign Relations":                        ["defense","energy"],
    "Judiciary":                                ["tech"],
}

# Subcommittee → sectors (more specific than full committee)
SUBCOMMITTEE_SECTORS: dict[str, list[str]] = {
    "Subcommittee on Cybersecurity":            ["tech","defense"],
    "Subcommittee on Emerging Threats":         ["tech","defense"],
    "Subcommittee on Seapower":                 ["defense"],
    "Subcommittee on Strategic Forces":         ["defense","aerospace"],
    "Subcommittee on Airland":                  ["defense","aerospace"],
    "Subcommittee on Health Care":              ["health"],
    "Subcommittee on Financial Institutions":   ["finance"],
    "Subcommittee on Securities":               ["finance","tech"],
    "Subcommittee on Energy":                   ["energy"],
    "Subcommittee on Communications":           ["telecom","tech"],
    "Subcommittee on Space and Science":        ["aerospace","tech"],
}

# ─────────────────────────────────────────────────────────────
# Senator committee assignments (119th Congress)
# Tuple: (committees, subcommittees, is_chair)
# ─────────────────────────────────────────────────────────────
SENATOR_COMMITTEES: dict[str, tuple[list[str], list[str], bool]] = {
    "Shelley Moore Capito": (["Environment and Public Works", "Appropriations", "Commerce, Science, and Transportation"],
                             [], True),
    "Nancy Pelosi":      (["Financial Services"], [], False),
    "Thomas Carper":     (["Environment and Public Works", "Finance"], [], True),
    "Tommy Tuberville":  (["Armed Services"],
                          ["Subcommittee on Airland","Subcommittee on Emerging Threats"], False),
    "Jack Reed":         (["Armed Services"],
                          ["Subcommittee on Seapower"], True),
    "Roger Wicker":      (["Armed Services","Commerce, Science, and Transportation"],
                          ["Subcommittee on Strategic Forces"], False),
    "Jeanne Shaheen":    (["Armed Services","Foreign Relations"],
                          [], False),
    "Dan Sullivan":      (["Armed Services","Commerce, Science, and Transportation"],
                          ["Subcommittee on Cybersecurity"], False),
    "Mark Warner":       (["Intelligence","Finance"],
                          ["Subcommittee on Securities"], True),
    "Marco Rubio":       (["Intelligence","Commerce, Science, and Transportation"],
                          ["Subcommittee on Communications"], False),
    "Richard Burr":      (["Intelligence"],
                          [], False),
    "Angus King":        (["Intelligence","Armed Services"],
                          ["Subcommittee on Cybersecurity"], False),
    "Mike Crapo":        (["Finance","Banking, Housing, and Urban Affairs"],
                          [], True),
    "Ron Wyden":         (["Finance"],
                          [], False),
    "Sherrod Brown":     (["Banking, Housing, and Urban Affairs"],
                          [], True),
    "Tim Scott":         (["Banking, Housing, and Urban Affairs","Finance"],
                          ["Subcommittee on Financial Institutions"], False),
    "Elizabeth Warren":  (["Finance","Banking, Housing, and Urban Affairs"],
                          [], False),
    "Maria Cantwell":    (["Commerce, Science, and Transportation","Energy and Natural Resources"],
                          ["Subcommittee on Communications","Subcommittee on Space and Science"], True),
    "Ted Cruz":          (["Commerce, Science, and Transportation"],
                          ["Subcommittee on Space and Science"], False),
    "John Thune":        (["Commerce, Science, and Transportation","Finance"],
                          [], False),
    "Joe Manchin":       (["Energy and Natural Resources"],
                          ["Subcommittee on Energy"], True),
    "John Barrasso":     (["Energy and Natural Resources"],
                          [], False),
    "Lisa Murkowski":    (["Energy and Natural Resources","Appropriations"],
                          ["Subcommittee on Energy"], False),
    "Bernie Sanders":    (["Health, Education, Labor, and Pensions"],
                          [], True),
    "Patty Murray":      (["Health, Education, Labor, and Pensions","Appropriations"],
                          ["Subcommittee on Health Care"], False),
    "Bill Cassidy":      (["Health, Education, Labor, and Pensions","Finance"],
                          ["Subcommittee on Health Care"], False),
    "Rand Paul":         (["Finance","Foreign Relations"],
                          [], False),
    "Dianne Feinstein":  (["Intelligence","Judiciary"],
                          [], False),
    "James Inhofe":      (["Armed Services","Environment and Public Works"],
                          [], False),
    "John Hoeven":       (["Appropriations","Energy and Natural Resources"],
                          [], False),
    "Kelly Loeffler":    (["Finance","Banking, Housing, and Urban Affairs"],
                          [], False),
    "David Perdue":      (["Armed Services","Banking, Housing, and Urban Affairs"],
                          [], False),
    "Sheldon Whitehouse":(["Environment and Public Works","Judiciary"],
                          [], False),
}

# ─────────────────────────────────────────────────────────────
# Known star senators (will be validated/overridden by real alpha)
# ─────────────────────────────────────────────────────────────
KNOWN_STAR_SENATORS = {
    "Tommy Tuberville","Kelly Loeffler","David Perdue","Richard Burr",
    "Dianne Feinstein","James Inhofe","Mark Warner","John Hoeven",
    "Rand Paul","Sheldon Whitehouse",
}
