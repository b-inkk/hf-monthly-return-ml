"""
Stock to Industry ETF Mapping

Maps each of the 132 S&P stocks to their most appropriate industry ETF
from the available 48 ETFs. Mapping is based on primary business sector.

Industry ETF Categories:
- XLB: Materials (Basic Materials, Chemicals, Mining)
- XLE: Energy (Oil, Gas, Energy Equipment)
- XLF: Financials (Banks, Insurance, Financial Services)
- XLI: Industrials (Aerospace, Defense, Industrial Equipment, Transportation)
- XLK: Technology (Software, Hardware, Semiconductors, IT Services)
- XLP: Consumer Staples (Food, Beverages, Household Products, Tobacco)
- XLU: Utilities (Electric, Gas, Water Utilities)
- XLV: Healthcare (Pharmaceuticals, Biotechnology, Medical Devices)
- XLY: Consumer Discretionary (Retail, Media, Automotive, Restaurants)
- IYR: Real Estate (REITs, Real Estate Services)
- GDX: Gold Miners (Gold mining companies)
- OIH: Oil Services (Oil field services and equipment)
- XME: Metals & Mining (Metal mining and processing)
- XOP: Oil & Gas Exploration (Oil and gas exploration companies)
- XRT: Retail (Retail companies)
"""

# Comprehensive stock-to-industry ETF mapping
STOCK_TO_INDUSTRY_ETF = {
    # Materials (XLB)
    'AA': 'XLB',        # Alcoa - Aluminum
    'DD': 'XLB',        # DuPont - Chemicals
    'DOW': 'XLB',       # Dow Chemical - Chemicals
    'FCX': 'XME',       # Freeport-McMoRan - Copper/Gold Mining
    'MON': 'XLB',       # Monsanto - Agricultural Chemicals
    'WY': 'XLB',        # Weyerhaeuser - Forest Products
    
    # Energy (XLE)
    'APA': 'XLE',       # Apache - Oil & Gas
    'APC': 'XLE',       # Anadarko Petroleum - Oil & Gas
    'BHI': 'OIH',       # Baker Hughes - Oil Services
    'COP': 'XLE',       # ConocoPhillips - Oil & Gas
    'CVX': 'XLE',       # Chevron - Oil & Gas
    'DVN': 'XLE',       # Devon Energy - Oil & Gas
    'HAL': 'OIH',       # Halliburton - Oil Services
    'NOV': 'OIH',       # National Oilwell Varco - Oil Services
    'OXY': 'XLE',       # Occidental Petroleum - Oil & Gas
    'SLB': 'OIH',       # Schlumberger - Oil Services
    'XOM': 'XLE',       # ExxonMobil - Oil & Gas
    
    # Financials (XLF)
    'AIG': 'XLF',       # AIG - Insurance
    'ALL': 'XLF',       # Allstate - Insurance
    'AXP': 'XLF',       # American Express - Financial Services
    'BAC': 'XLF',       # Bank of America - Banking
    'BK': 'XLF',        # Bank of New York Mellon - Banking
    'BRK.B': 'XLF',     # Berkshire Hathaway - Diversified Financials
    'C': 'XLF',         # Citigroup - Banking
    'COF': 'XLF',       # Capital One - Banking
    'GS': 'XLF',        # Goldman Sachs - Investment Banking
    'JPM': 'XLF',       # JPMorgan Chase - Banking
    'MA': 'XLF',        # Mastercard - Payment Processing
    'MET': 'XLF',       # MetLife - Insurance
    'MS': 'XLF',        # Morgan Stanley - Investment Banking
    'NYX': 'XLF',       # NYSE Euronext - Financial Exchange
    'RF': 'XLF',        # Regions Financial - Banking
    'SLE': 'XLF',       # Sara Lee - Insurance (Discontinued)
    'USB': 'XLF',       # US Bancorp - Banking
    'V': 'XLF',         # Visa - Payment Processing
    'WFC': 'XLF',       # Wells Fargo - Banking
    
    # Industrials (XLI)
    'BA': 'XLI',        # Boeing - Aerospace
    'CAT': 'XLI',       # Caterpillar - Heavy Machinery
    'EMR': 'XLI',       # Emerson Electric - Industrial Equipment
    'FDX': 'XLI',       # FedEx - Transportation
    'GD': 'XLI',        # General Dynamics - Aerospace/Defense
    'GE': 'XLI',        # General Electric - Industrial Conglomerate
    'HON': 'XLI',       # Honeywell - Industrial Equipment
    'LMT': 'XLI',       # Lockheed Martin - Aerospace/Defense
    'MMM': 'XLI',       # 3M - Industrial Products
    'NSC': 'XLI',       # Norfolk Southern - Railroad
    'RTN': 'XLI',       # Raytheon - Defense
    'UNP': 'XLI',       # Union Pacific - Railroad
    'UPS': 'XLI',       # UPS - Transportation
    'UTX': 'XLI',       # United Technologies - Aerospace
    'BNI': 'XLI',       # Burlington Northern - Railroad
    
    # Technology (XLK)
    'AAPL': 'XLK',      # Apple - Technology Hardware
    'ACN': 'XLK',       # Accenture - IT Services
    'CSCO': 'XLK',      # Cisco - Networking Equipment
    'DELL': 'XLK',      # Dell - Computer Hardware
    'EBAY': 'XLK',      # eBay - Internet Services
    'EMC': 'XLK',       # EMC - Data Storage
    'FB': 'XLK',        # Facebook - Social Media
    'GOOG': 'XLK',      # Google - Internet Services
    'GOOGL': 'XLK',     # Google Class A - Internet Services
    'HPQ': 'XLK',       # HP - Computer Hardware
    'IBM': 'XLK',       # IBM - Technology Services
    'INTC': 'XLK',      # Intel - Semiconductors
    'MSFT': 'XLK',      # Microsoft - Software
    'ORCL': 'XLK',      # Oracle - Software
    'QCOM': 'XLK',      # Qualcomm - Semiconductors
    'TXN': 'XLK',       # Texas Instruments - Semiconductors
    'XRX': 'XLK',       # Xerox - Technology Services
    
    # Consumer Staples (XLP)
    'BUD': 'XLP',       # Anheuser-Busch - Beverages
    'CL': 'XLP',        # Colgate-Palmolive - Household Products
    'CPB': 'XLP',       # Campbell Soup - Food Products
    'HNZ': 'XLP',       # Heinz - Food Products
    'KFT': 'XLP',       # Kraft Foods - Food Products
    'KO': 'XLP',        # Coca-Cola - Beverages
    'MDLZ': 'XLP',      # Mondelez - Food Products
    'MO': 'XLP',        # Altria - Tobacco
    'PEP': 'XLP',       # PepsiCo - Beverages/Snacks
    'PG': 'XLP',        # Procter & Gamble - Household Products
    'PM': 'XLP',        # Philip Morris - Tobacco
    'WMT': 'XLP',       # Walmart - Retail (Consumer Staples)
    
    # Utilities (XLU)
    'AEP': 'XLU',       # American Electric Power - Electric Utility
    'AES': 'XLU',       # AES - Electric Utility
    'ETR': 'XLU',       # Entergy - Electric Utility
    'EXC': 'XLU',       # Exelon - Electric Utility
    'SO': 'XLU',        # Southern Company - Electric Utility
    'WMB': 'XLU',       # Williams Companies - Pipeline Utility
    'KMI': 'XLU',       # Kinder Morgan - Pipeline Utility
    
    # Healthcare (XLV)
    'ABBV': 'XLV',      # AbbVie - Pharmaceuticals
    'ABT': 'XLV',       # Abbott Laboratories - Medical Devices
    'AMGN': 'XLV',      # Amgen - Biotechnology
    'BAX': 'XLV',       # Baxter - Medical Devices
    'BIIB': 'XLV',      # Biogen - Biotechnology
    'BMY': 'XLV',       # Bristol-Myers Squibb - Pharmaceuticals
    'GILD': 'XLV',      # Gilead Sciences - Biotechnology
    'JNJ': 'XLV',       # Johnson & Johnson - Healthcare Conglomerate
    'LLY': 'XLV',       # Eli Lilly - Pharmaceuticals
    'MDT': 'XLV',       # Medtronic - Medical Devices
    'MRK': 'XLV',       # Merck - Pharmaceuticals
    'PFE': 'XLV',       # Pfizer - Pharmaceuticals
    'SGP': 'XLV',       # Schering-Plough - Pharmaceuticals
    'UNH': 'XLV',       # UnitedHealth - Health Insurance
    'WYE': 'XLV',       # Wyeth - Pharmaceuticals
    
    # Consumer Discretionary (XLY)
    'AMZN': 'XLY',      # Amazon - E-commerce/Retail
    'CBS': 'XLY',       # CBS - Media
    'CMCSA': 'XLY',     # Comcast - Media/Cable
    'COST': 'XLY',      # Costco - Retail
    'CVS': 'XLY',       # CVS - Retail Pharmacy
    'DIS': 'XLY',       # Disney - Media/Entertainment
    'F': 'XLY',         # Ford - Automotive
    'FOXA': 'XLY',      # Fox - Media
    'GM': 'XLY',        # General Motors - Automotive
    'HD': 'XLY',        # Home Depot - Retail
    'LOW': 'XLY',       # Lowe's - Retail
    'MCD': 'XLY',       # McDonald's - Restaurants
    'NKE': 'XLY',       # Nike - Apparel
    'NWSA': 'XLY',      # News Corp - Media
    'SBUX': 'XLY',      # Starbucks - Restaurants
    'TGT': 'XLY',       # Target - Retail
    'TWX': 'XLY',       # Time Warner - Media
    'WAG': 'XLY',       # Walgreens - Retail Pharmacy
    'WBA': 'XLY',       # Walgreens Boots Alliance - Retail Pharmacy
    
    # Real Estate (IYR)
    'SPG': 'IYR',       # Simon Property Group - REITs
    
    # Telecommunications (Use XLY as closest match)
    'T': 'XLY',         # AT&T - Telecommunications
    'VZ': 'XLY',        # Verizon - Telecommunications
    
    # Conglomerates/Other (Use SPY as broad market proxy)
    'ARNC': 'XLI',      # Arconic - Industrial Materials
    'ATI': 'XLB',       # Allegheny Technologies - Specialty Materials
    'AVP': 'XLF',       # Avon Products - Personal Care (Financial proxy)
    'BDK': 'XLY',       # Black & Decker - Consumer Tools
    'CCU': 'XLP',       # Compania Cervecerias Unidas - Beverages
    'COV': 'XLV',       # Covidien - Medical Devices
    'S': 'XLY',         # Sprint - Telecommunications
    'ACT': 'XLV',       # Actavis - Pharmaceuticals
}

def get_industry_etf(stock_symbol):
    """
    Get the industry ETF for a given stock symbol.
    
    Args:
        stock_symbol (str): Stock symbol (e.g., 'AAPL')
        
    Returns:
        str: Industry ETF symbol (e.g., 'XLK') or 'SPY' as fallback
    """
    return STOCK_TO_INDUSTRY_ETF.get(stock_symbol, 'SPY')

def get_all_mappings():
    """
    Get all stock-to-industry ETF mappings.
    
    Returns:
        dict: Complete mapping dictionary
    """
    return STOCK_TO_INDUSTRY_ETF.copy()

def has_industry_mapping(stock_symbol):
    """
    Check if a stock has an explicit industry ETF mapping.
    
    Args:
        stock_symbol (str): Stock symbol to check
        
    Returns:
        bool: True if stock has explicit mapping, False if would use SPY fallback
    """
    return stock_symbol in STOCK_TO_INDUSTRY_ETF

def get_stocks_by_etf(etf_symbol):
    """
    Get all stocks mapped to a specific ETF.
    
    Args:
        etf_symbol (str): ETF symbol (e.g., 'XLK')
        
    Returns:
        list: List of stock symbols mapped to this ETF
    """
    return [stock for stock, etf in STOCK_TO_INDUSTRY_ETF.items() if etf == etf_symbol]

def get_mapping_statistics():
    """
    Get statistics about the mapping distribution.
    
    Returns:
        dict: Statistics including ETF usage counts and coverage
    """
    etf_counts = {}
    for stock, etf in STOCK_TO_INDUSTRY_ETF.items():
        etf_counts[etf] = etf_counts.get(etf, 0) + 1
    
    return {
        'total_stocks_mapped': len(STOCK_TO_INDUSTRY_ETF),
        'total_etfs_used': len(etf_counts),
        'etf_usage': etf_counts,
        'coverage_percentage': (len(STOCK_TO_INDUSTRY_ETF) / 132) * 100
    }

def validate_mappings():
    """
    Validate that all mappings use available ETFs.
    
    Returns:
        tuple: (valid_mappings, invalid_mappings)
    """
    available_etfs = {
        'DIA', 'EEM', 'EFA', 'EWZ', 'FAS', 'FAZ', 'FXE', 'FXI', 'GDX', 'GLD',
        'HYG', 'IVV', 'IWF', 'IWM', 'IWO', 'IYR', 'JNK', 'LQD', 'MDY', 'OIH',
        'QLD', 'QQQ', 'RSX', 'SDS', 'SH', 'SHY', 'SLV', 'SPY', 'SSO', 'TLT',
        'TZA', 'UPRO', 'USO', 'VTI', 'VWO', 'VXX', 'XLB', 'XLE', 'XLF', 'XLI',
        'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOP', 'XRT'
    }
    
    valid_mappings = {}
    invalid_mappings = {}
    
    for stock, etf in STOCK_TO_INDUSTRY_ETF.items():
        if etf in available_etfs:
            valid_mappings[stock] = etf
        else:
            invalid_mappings[stock] = etf
            
    return valid_mappings, invalid_mappings

if __name__ == "__main__":
    # Validate mappings
    valid, invalid = validate_mappings()
    print(f"Valid mappings: {len(valid)}")
    print(f"Invalid mappings: {len(invalid)}")
    
    if invalid:
        print("Invalid mappings found:")
        for stock, etf in invalid.items():
            print(f"  {stock} -> {etf}")
    
    # Show mapping statistics
    etf_counts = {}
    for stock, etf in STOCK_TO_INDUSTRY_ETF.items():
        etf_counts[etf] = etf_counts.get(etf, 0) + 1
    
    print("\nETF usage statistics:")
    for etf, count in sorted(etf_counts.items()):
        print(f"  {etf}: {count} stocks")
    
    print(f"\nTotal stocks mapped: {len(STOCK_TO_INDUSTRY_ETF)}")
    print(f"Expected stocks: 132")
    print(f"Missing stocks: {132 - len(STOCK_TO_INDUSTRY_ETF)}")