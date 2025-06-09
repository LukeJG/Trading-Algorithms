import spacy
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
from datetime import datetime

@dataclass
class FinancialEvent:
    catalyst: str
    entities: List[str]
    sentiment: str
    confidence: float
    event_type: str
    timestamp: Optional[datetime] = None

class FinancialNER:
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Please install it with: python -m spacy download {model_name}")
            raise
        
        # Comprehensive financial entity patterns
        self.financial_entities = {
            "CENTRAL_BANK": [
                "Federal Reserve", "Fed", "ECB", "European Central Bank", "Bank of England", "BOE", 
                "Bank of Japan", "BOJ", "People's Bank of China", "PBOC", "Reserve Bank of Australia", 
                "RBA", "Bank of Canada", "BOC", "Swiss National Bank", "SNB", "Reserve Bank of India", 
                "RBI", "Bank of Russia", "Central Bank"
            ],
            "INVESTMENT_BANK": [
                "Goldman Sachs", "JPMorgan", "JP Morgan", "Morgan Stanley", "Bank of America", "BofA",
                "Citigroup", "Citi", "Deutsche Bank", "Credit Suisse", "UBS", "Barclays", "HSBC",
                "Wells Fargo", "Nomura", "Jefferies", "Lazard"
            ],
            "TECH_COMPANY": [
                "Apple", "Microsoft", "Google", "Alphabet", "Amazon", "Meta", "Facebook", "Tesla",
                "Netflix", "Nvidia", "Intel", "AMD", "Oracle", "Salesforce", "Adobe", "IBM",
                "Twitter", "Uber", "Airbnb", "Zoom", "PayPal", "Square", "Block"
            ],
            "ENERGY_COMPANY": [
                "ExxonMobil", "Chevron", "Shell", "BP", "Total", "ConocoPhillips", "ENI", "Equinor",
                "Suncor", "Canadian Natural Resources", "Kinder Morgan", "Enbridge", "TC Energy"
            ],
            "AUTOMOTIVE": [
                "Tesla", "Ford", "General Motors", "GM", "Toyota", "Volkswagen", "BMW", "Mercedes",
                "Nissan", "Honda", "Hyundai", "Stellantis", "Lucid Motors", "Rivian", "NIO"
            ],
            "AIRLINE": [
                "American Airlines", "Delta", "United Airlines", "Southwest", "JetBlue", "Alaska Airlines",
                "Lufthansa", "British Airways", "Air France", "Emirates", "Qatar Airways", "Cathay Pacific"
            ],
            "RETAIL": [
                "Walmart", "Amazon", "Target", "Costco", "Home Depot", "Lowe's", "Best Buy", 
                "Macy's", "Nordstrom", "TJX", "Ross Stores", "Dollar General", "CVS", "Walgreens"
            ],
            "PHARMACEUTICAL": [
                "Pfizer", "Johnson & Johnson", "J&J", "Merck", "AbbVie", "Bristol Myers Squibb",
                "Eli Lilly", "Novartis", "Roche", "AstraZeneca", "GlaxoSmithKline", "GSK", "Sanofi",
                "Gilead", "Amgen", "Biogen", "Regeneron", "Vertex", "Moderna", "BioNTech"
            ],
            "FINANCIAL_SERVICES": [
                "Berkshire Hathaway", "BlackRock", "Vanguard", "Fidelity", "State Street", "T. Rowe Price",
                "Charles Schwab", "E*TRADE", "Interactive Brokers", "Robinhood", "Coinbase"
            ],
            "COMMODITY": [
                "oil", "crude oil", "Brent crude", "WTI", "natural gas", "gold", "silver", "platinum", 
                "palladium", "copper", "aluminum", "zinc", "nickel", "iron ore", "coal", "uranium",
                "wheat", "corn", "soybeans", "rice", "sugar", "coffee", "cocoa", "cotton",
                "lumber", "steel", "lithium", "cobalt", "rare earth metals"
            ],
            "CURRENCY": [
                "USD", "EUR", "GBP", "JPY", "CNY", "AUD", "CAD", "CHF", "SEK", "NOK", "NZD",
                "dollar", "euro", "pound", "yen", "yuan", "renminbi", "franc", "krona", "peso",
                "rupee", "real", "rand", "lira", "ruble", "won"
            ],
            "CRYPTOCURRENCY": [
                "Bitcoin", "BTC", "Ethereum", "ETH", "Binance Coin", "BNB", "XRP", "Ripple",
                "Cardano", "ADA", "Solana", "SOL", "Dogecoin", "DOGE", "Polygon", "MATIC",
                "Avalanche", "AVAX", "Chainlink", "LINK", "Litecoin", "LTC", "crypto", "cryptocurrency"
            ],
            "MARKET_INDEX": [
                "S&P 500", "SPX", "Dow Jones", "DJIA", "NASDAQ", "Russell 2000", "VIX",
                "FTSE", "FTSE 100", "DAX", "CAC 40", "Nikkei", "Nikkei 225", "Hang Seng",
                "Shanghai Composite", "MSCI", "Euro Stoxx", "ASX", "TSX", "IBEX", "SMI"
            ],
            "ECONOMIC_INDICATOR": [
                "GDP", "inflation", "CPI", "PPI", "unemployment rate", "jobless claims", "NFP",
                "non-farm payrolls", "PCE", "PMI", "ISM", "consumer confidence", "retail sales",
                "housing starts", "existing home sales", "durable goods", "trade balance",
                "current account", "budget deficit", "yield curve", "FOMC", "Jackson Hole"
            ],
            "REGULATORY_BODY": [
                "SEC", "Securities and Exchange Commission", "CFTC", "FINRA", "FDIC", "OCC",
                "Federal Trade Commission", "FTC", "Department of Justice", "DOJ", "Treasury",
                "IRS", "Fed", "Basel Committee", "FSB", "IOSCO", "European Commission"
            ],
            "INTERNATIONAL_ORG": [
                "OPEC", "OPEC+", "IMF", "International Monetary Fund", "World Bank", "WTO",
                "World Trade Organization", "G7", "G20", "OECD", "BIS", "Bank for International Settlements",
                "European Union", "EU", "ASEAN", "NAFTA", "USMCA"
            ],
            "RATING_AGENCY": [
                "Moody's", "Standard & Poor's", "S&P", "Fitch", "DBRS", "Scope Ratings",
                "credit rating", "rating agency", "investment grade", "junk bond", "high yield"
            ],
            "EXCHANGE": [
                "NYSE", "New York Stock Exchange", "NASDAQ", "Chicago Mercantile Exchange", "CME",
                "CBOE", "LSE", "London Stock Exchange", "Tokyo Stock Exchange", "TSE",
                "Shanghai Stock Exchange", "Hong Kong Stock Exchange", "HKEX", "Euronext",
                "Deutsche Börse", "Toronto Stock Exchange", "ASX", "Australian Securities Exchange"
            ],
            "SECTOR": [
                "technology", "tech", "healthcare", "financials", "energy", "utilities", "materials",
                "industrials", "consumer discretionary", "consumer staples", "real estate", "REITs",
                "telecommunications", "defense", "aerospace", "biotechnology", "fintech", "clean energy",
                "renewable energy", "electric vehicles", "EVs", "artificial intelligence", "AI",
                "cloud computing", "cybersecurity", "semiconductors", "mining", "agriculture"
            ],
            "COUNTRY": [
                "United States", "USA", "US", "China", "Japan", "Germany", "United Kingdom", "UK", 
                "France", "Italy", "Canada", "Australia", "South Korea", "India", "Brazil", "Russia",
                "Mexico", "Spain", "Netherlands", "Switzerland", "Taiwan", "Belgium", "Ireland",
                "Israel", "Saudi Arabia", "UAE", "Singapore", "Hong Kong", "Sweden", "Norway"
            ]
        }
        
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        patterns = []
        for entity_type, entities in self.financial_entities.items():
            for entity in entities:
                patterns.append({"label": entity_type, "pattern": entity})
        ruler.add_patterns(patterns)
    
    def add_custom_entities(self, entity_type: str, entities: List[str]):
        if entity_type not in self.financial_entities:
            self.financial_entities[entity_type] = []
        self.financial_entities[entity_type].extend(entities)
        
        patterns = []
        for entity in entities:
            patterns.append({"label": entity_type, "pattern": entity})
        
        ruler = self.nlp.get_pipe("entity_ruler")
        ruler.add_patterns(patterns)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text)
        entities = {}
        
        irrelevant_patterns = [
            r'^\d+[%$]?$',  # Pure numbers or percentages
            r'^[A-Z]{1,2}$',  # Single or double letters
            r'^\w{1,2}$',  # Very short words
            r'^(the|and|or|but|in|on|at|to|for|of|with|by)$',  # Common stop words
            r'^\d{1,2}:\d{2}$',  # Time formats
            r'^\d{1,2}/\d{1,2}(/\d{2,4})?$'  # Date formats
        ]
        
        for ent in doc.ents:
            if any(re.match(pattern, ent.text, re.IGNORECASE) for pattern in irrelevant_patterns):
                continue
            
            if len(ent.text.strip()) < 2:
                continue
            
            entity_text = ent.text.strip()
            entity_label = ent.label_
            
            if entity_label in ["ORG", "PERSON", "GPE"]:
                entity_type = self._classify_entity(entity_text)
                if entity_type:
                    entity_label = entity_type
            
            if entity_label not in entities:
                entities[entity_label] = []
            
            if entity_text not in entities[entity_label]:
                entities[entity_label].append(entity_text)
        
        entities = self._filter_false_positives(entities)
        
        return entities
    
    def _classify_entity(self, entity_text: str) -> Optional[str]:
        entity_lower = entity_text.lower()
        
        for entity_type, entity_list in self.financial_entities.items():
            for known_entity in entity_list:
                if entity_lower == known_entity.lower() or entity_lower in known_entity.lower():
                    return entity_type
        return None
    
    def _filter_false_positives(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        filtered_entities = {}
        
        false_positives = {
            "today", "yesterday", "tomorrow", "morning", "afternoon", "evening",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "first", "second", "third", "fourth", "last", "next", "this", "that",
            "report", "news", "article", "story", "update", "analysis"
        }
        
        for entity_type, entity_list in entities.items():
            filtered_list = []
            for entity in entity_list:
                if entity.lower() not in false_positives and len(entity.strip()) > 1:
                    filtered_list.append(entity)
            
            if filtered_list: 
                filtered_entities[entity_type] = filtered_list
        
        return filtered_entities



class EventCatalystDetector:
    def __init__(self):
        self.event_patterns = {
            "ANALYST_ACTION": {
                "patterns": [
                    r"(?:analyst|research).*?(?:raised|increased|lifted|boosted|upgraded|lowered|cut|reduced|downgraded).*?(?:price target|target price|rating)",
                    r"(?:price target|target price).*?(?:raised|increased|lifted|boosted|upgraded|lowered|cut|reduced|downgraded)",
                    r"(?:buy|sell|hold|neutral|overweight|underweight|outperform|underperform|strong buy|strong sell).*?rating",
                    r"(?:analyst|research|coverage).*?(?:initiated|resumed|suspended|maintained|reiterated)",
                    r"(?:Morgan Stanley|Goldman Sachs|JP Morgan|Bank of America|Citi|Wells Fargo|Deutsche Bank|Barclays).*?(?:analyst|research)"
                ],
                "keywords": ["analyst", "price target", "rating", "buy", "sell", "hold", "upgrade", "downgrade", "research"]
            },
            "INVESTOR_DAY": {
                "patterns": [
                    r"(?:investor day|investor event|analyst day|capital markets day)",
                    r"(?:expected to|will|plans to).*?(?:present|announce|outline|discuss).*?(?:strategy|plan|roadmap|guidance)",
                    r"(?:strategic plan|growth plan|long-term plan|business plan|roadmap)",
                    r"(?:outlook|guidance|forecast|projections).*?(?:2025|2026|2027|2028|long-term|multi-year)"
                ],
                "keywords": ["investor day", "strategic plan", "guidance", "roadmap", "outlook", "long-term"]
            },
            "INTEREST_RATE": {
                "patterns": [
                    r"(?:Federal Reserve|Fed|ECB|central bank).*?(?:raises?|hikes?|increases?|cuts?|lowers?|reduces?).*?(?:interest rates?|rates?|fed funds rate)",
                    r"(?:interest rates?|rates?|fed funds rate).*?(?:raised|hiked|increased|cut|lowered|reduced).*?(?:Federal Reserve|Fed|ECB|central bank)",
                    r"(?:FOMC|Federal Open Market Committee).*?(?:decision|meeting|announcement)",
                    r"(?:rate hike|rate cut|rate increase|rate decrease|monetary policy)(?!.*target)"  # Exclude "target" to avoid price target confusion
                ],
                "keywords": ["Federal Reserve", "Fed", "FOMC", "interest rate", "rate hike", "rate cut", "monetary policy"]
            },
            "PRODUCTION_CHANGE": {
                "patterns": [
                    r"(?:cuts?|reduces?|increases?|boosts?|slashes?).*?(?:production|output|supply|manufacturing)",
                    r"(?:production|output|supply|manufacturing).*?(?:cut|reduced|increased|boosted|slashed)",
                    r"production (?:quota|target|limit|agreement|deal|capacity)"
                ],
                "keywords": ["production cut", "output reduction", "supply cut", "production increase", "quota", "manufacturing"]
            },
            "REGULATORY_POLICY": {
                "patterns": [
                    r"(?:SEC|FDA|FTC|DOJ|Treasury|regulatory body).*?(?:announces?|implements?|proposes?).*?(?:regulation|policy|rule|law|ban|restriction)",
                    r"(?:regulation|policy|rule|law|ban|restriction).*?(?:announced|implemented|proposed|changed|passed).*?(?:SEC|FDA|FTC|DOJ|Treasury)",
                    r"regulatory (?:approval|ban|restriction|oversight|framework|compliance|filing)"
                ],
                "keywords": ["SEC", "FDA", "regulation", "policy change", "regulatory approval", "ban", "restriction", "compliance"]
            },
            "GEOPOLITICAL": {
                "patterns": [
                    r"(?:war|conflict|tensions?|sanctions?|trade war|tariffs?|embargo)",
                    r"(?:military|invasion|attack|strike|missile|nuclear)",
                    r"(?:diplomatic|political).*?(?:crisis|tension|dispute|negotiation)",
                    r"(?:election|referendum|coup|protest|civil unrest)"
                ],
                "keywords": ["war", "conflict", "sanctions", "trade war", "military action", "tariffs", "election"]
            },
            "EARNINGS_REPORT": {
                "patterns": [
                    r"(?:earnings|revenue|profit|sales).*?(?:beat|miss|exceed|disappoint|surprise)",
                    r"(?:quarterly|annual|Q[1-4]).*?(?:results|earnings|report|revenue)",
                    r"(?:guidance|forecast|outlook).*?(?:raised|lowered|maintained|revised|updated)",
                    r"(?:EPS|earnings per share).*?(?:beat|miss|exceed)"
                ],
                "keywords": ["earnings", "revenue", "profit", "quarterly results", "guidance", "beat", "miss"]
            },
            "MERGER_ACQUISITION": {
                "patterns": [
                    r"(?:acquires?|acquisition|merger|merges?|buyout|takeover)",
                    r"(?:deal|transaction).*?(?:worth|valued|billion|million)",
                    r"(?:hostile|friendly).*?(?:takeover|bid|offer)",
                    r"(?:joint venture|partnership|strategic alliance)"
                ],
                "keywords": ["acquisition", "merger", "buyout", "takeover", "deal", "joint venture"]
            },
            "IPO_LISTING": {
                "patterns": [
                    r"(?:IPO|initial public offering|goes public|public listing)",
                    r"(?:files for|plans|announces).*?(?:IPO|public offering)",
                    r"(?:debut|listing|trades).*?(?:stock exchange|NYSE|NASDAQ)",
                    r"(?:SPAC|special purpose acquisition)"
                ],
                "keywords": ["IPO", "initial public offering", "goes public", "listing", "SPAC"]
            },
            "DIVIDEND_BUYBACK": {
                "patterns": [
                    r"(?:dividend|payout).*?(?:increase|decrease|cut|suspended|resumed)",
                    r"(?:share buyback|stock repurchase|repurchase program)",
                    r"(?:special dividend|extra dividend|quarterly dividend)",
                    r"(?:authorizes|announces).*?(?:buyback|repurchase)"
                ],
                "keywords": ["dividend", "buyback", "repurchase", "payout", "share repurchase"]
            },
            "CREDIT_RATING": {
                "patterns": [
                    r"(?:credit rating|rating).*?(?:upgraded|downgraded|affirmed|maintained)",
                    r"(?:Moody's|S&P|Fitch).*?(?:upgrade|downgrade|rating|outlook)",
                    r"(?:investment grade|junk|speculative grade|default risk)",
                    r"(?:outlook).*?(?:positive|negative|stable|revised)"
                ],
                "keywords": ["credit rating", "upgrade", "downgrade", "Moody's", "S&P", "Fitch", "outlook"]
            },
            "BANKRUPTCY_DISTRESS": {
                "patterns": [
                    r"(?:bankruptcy|Chapter 11|Chapter 7|insolvency|liquidation)",
                    r"(?:files for|declares|enters).*?(?:bankruptcy|administration|receivership)",
                    r"(?:financial distress|debt restructuring|default|missed payment)",
                    r"(?:going concern|wind down|cease operations)"
                ],
                "keywords": ["bankruptcy", "Chapter 11", "insolvency", "liquidation", "financial distress", "default"]
            },
            "CRYPTO_EVENTS": {
                "patterns": [
                    r"(?:Bitcoin|crypto|cryptocurrency|blockchain).*?(?:surge|crash|rally|plunge)",
                    r"(?:mining|staking|DeFi|NFT|wallet|exchange).*?(?:hack|breach|exploit)",
                    r"(?:regulatory|SEC|CFTC).*?(?:crypto|Bitcoin|Ethereum|digital assets)",
                    r"(?:halving|fork|upgrade|hard fork|soft fork)"
                ],
                "keywords": ["Bitcoin", "crypto", "blockchain", "DeFi", "NFT", "mining", "halving", "fork"]
            },
            "ESG_SUSTAINABILITY": {
                "patterns": [
                    r"(?:ESG|environmental|sustainability|carbon).*?(?:goals|targets|emissions|neutral)",
                    r"(?:climate change|renewable energy|clean energy|green).*?(?:investment|initiative|policy)",
                    r"(?:social responsibility|governance|diversity|inclusion)",
                    r"(?:net zero|carbon footprint|sustainable|green bonds)"
                ],
                "keywords": ["ESG", "sustainability", "carbon neutral", "renewable energy", "climate change", "green bonds"]
            }
        }
        
        self.sector_sentiment_map = {
            "INTEREST_RATE": {
                "hike": {
                    "banks": "positive", 
                    "real_estate": "negative", 
                    "growth_stocks": "negative",
                    "financials": "positive",
                    "utilities": "negative",
                    "technology": "negative",
                    "insurance": "positive"
                },
                "cut": {
                    "banks": "negative", 
                    "real_estate": "positive", 
                    "growth_stocks": "positive",
                    "financials": "negative",
                    "utilities": "positive",
                    "technology": "positive",
                    "insurance": "negative"
                }
            },
            "PRODUCTION_CHANGE": {
                "cut": {
                    "energy": "positive", 
                    "airlines": "negative", 
                    "manufacturing": "negative",
                    "oil_gas": "positive",
                    "commodities": "positive",
                    "materials": "positive"
                },
                "increase": {
                    "energy": "negative", 
                    "airlines": "positive", 
                    "manufacturing": "positive",
                    "oil_gas": "negative",
                    "commodities": "negative",
                    "materials": "negative"
                }
            },
            "ANALYST_ACTION": {
                "upgrade": {
                    "all_sectors": "positive"
                },
                "downgrade": {
                    "all_sectors": "negative"
                },
                "price_target_increase": {
                    "all_sectors": "positive"
                },
                "price_target_decrease": {
                    "all_sectors": "negative"
                }
            },
            "EARNINGS_REPORT": {
                "beat": {
                    "all_sectors": "positive"
                },
                "miss": {
                    "all_sectors": "negative"
                },
                "guidance_raise": {
                    "all_sectors": "positive"
                },
                "guidance_lower": {
                    "all_sectors": "negative"
                }
            },
            "MERGER_ACQUISITION": {
                "acquisition_target": {
                    "all_sectors": "positive"
                },
                "acquisition_buyer": {
                    "all_sectors": "neutral"  # Mixed sentiment, depends on premium paid
                },
                "merger": {
                    "all_sectors": "positive"
                }
            },
            "REGULATORY_POLICY": {
                "approval": {
                    "pharmaceuticals": "positive",
                    "biotechnology": "positive",
                    "healthcare": "positive",
                    "energy": "positive",
                    "technology": "neutral"
                },
                "ban": {
                    "all_sectors": "negative"
                },
                "restriction": {
                    "all_sectors": "negative"
                },
                "deregulation": {
                    "all_sectors": "positive"
                }
            },
            "DIVIDEND_BUYBACK": {
                "dividend_increase": {
                    "all_sectors": "positive"
                },
                "dividend_cut": {
                    "all_sectors": "negative"
                },
                "dividend_suspended": {
                    "all_sectors": "negative"
                },
                "buyback_announced": {
                    "all_sectors": "positive"
                }
            },
            "CREDIT_RATING": {
                "upgrade": {
                    "all_sectors": "positive"
                },
                "downgrade": {
                    "all_sectors": "negative"
                },
                "outlook_positive": {
                    "all_sectors": "positive"
                },
                "outlook_negative": {
                    "all_sectors": "negative"
                }
            },
            "GEOPOLITICAL": {
                "conflict": {
                    "defense": "positive",
                    "energy": "positive",
                    "gold_commodities": "positive",
                    "airlines": "negative",
                    "tourism": "negative",
                    "emerging_markets": "negative"
                },
                "sanctions": {
                    "defense": "neutral",
                    "energy": "mixed",
                    "technology": "negative",
                    "consumer_goods": "negative"
                },
                "trade_war": {
                    "domestic_focused": "positive",
                    "export_dependent": "negative",
                    "import_dependent": "negative"
                }
            },
            "BANKRUPTCY_DISTRESS": {
                "bankruptcy": {
                    "all_sectors": "negative"
                },
                "restructuring": {
                    "all_sectors": "mixed"
                },
                "default": {
                    "all_sectors": "negative"
                }
            },
            "IPO_LISTING": {
                "ipo_success": {
                    "all_sectors": "positive"
                },
                "ipo_postponed": {
                    "all_sectors": "negative"
                },
                "ipo_pricing": {
                    "all_sectors": "neutral"
                }
            },
            "CRYPTO_EVENTS": {
                "surge": {
                    "crypto_related": "positive",
                    "blockchain": "positive",
                    "fintech": "positive"
                },
                "crash": {
                    "crypto_related": "negative",
                    "blockchain": "negative",
                    "fintech": "negative"
                },
                "regulatory_clarity": {
                    "crypto_related": "positive",
                    "fintech": "positive"
                },
                "hack_exploit": {
                    "crypto_related": "negative",
                    "cybersecurity": "positive"
                }
            },
            "ESG_SUSTAINABILITY": {
                "positive_esg": {
                    "renewable_energy": "positive",
                    "clean_tech": "positive",
                    "sustainable_companies": "positive"
                },
                "negative_esg": {
                    "fossil_fuels": "negative",
                    "heavy_industry": "negative"
                },
                "carbon_neutral_goals": {
                    "all_sectors": "mixed"  # Depends on implementation costs
                }
            },
            "INVESTOR_DAY": {
                "positive_guidance": {
                    "all_sectors": "positive"
                },
                "strategy_announcement": {
                    "all_sectors": "neutral"
                },
                "restructuring_plan": {
                    "all_sectors": "mixed"
                }
            }
        }
    
    def detect_events(self, text: str) -> List[Dict]:
        events = []
        text_lower = text.lower()
        
        for event_type, config in self.event_patterns.items():
            for pattern in config["patterns"]:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    events.append({
                        "event_type": event_type,
                        "matched_text": match.group(),
                        "confidence": 0.8,  
                        "start": match.start(),
                        "end": match.end()
                    })
        
        return events
    
    def determine_sentiment(self, event_type: str, text: str, entities: List[str]) -> str:
        text_lower = text.lower()
        
        if event_type == "ANALYST_ACTION":
            return self._analyze_analyst_sentiment(text_lower)
        
        strong_positive_words = ["surge", "soar", "rocket", "boom", "breakthrough", "record", "exceed", "beat", "strong", "robust"]
        positive_words = ["increase", "rise", "boost", "growth", "positive", "improved", "gains", "up", "higher", "recovery"]
        neutral_words = ["maintain", "hold", "stable", "unchanged", "flat", "steady", "reiterate", "affirm"]
        negative_words = ["decrease", "fall", "cut", "decline", "negative", "weak", "miss", "disappoint", "concerns", "challenges"]
        strong_negative_words = ["crash", "plunge", "collapse", "plummet", "disaster", "crisis", "fail", "suspended", "bankrupt"]
        
        strong_positive_score = sum(2 for word in strong_positive_words if word in text_lower)
        positive_score = sum(1 for word in positive_words if word in text_lower)
        neutral_score = sum(0.5 for word in neutral_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        strong_negative_score = sum(2 for word in strong_negative_words if word in text_lower)
        
        total_positive = strong_positive_score + positive_score
        total_negative = strong_negative_score + negative_score
        
        if total_positive > total_negative + neutral_score:
            return "Positive"
        elif total_negative > total_positive + neutral_score:
            return "Negative"
        elif neutral_score > 0:
            return "Neutral"
        else:
            return "Mixed"
    
    def _analyze_analyst_sentiment(self, text_lower: str) -> str:
        bullish_ratings = ["buy", "strong buy", "overweight", "outperform", "positive"]
        neutral_ratings = ["hold", "neutral", "market perform", "equal weight"]
        bearish_ratings = ["sell", "strong sell", "underweight", "underperform", "negative"]
        
        positive_actions = ["upgrade", "raised", "increased", "lifted", "boosted", "initiated.*buy"]
        negative_actions = ["downgrade", "lowered", "cut", "reduced", "suspended", "initiated.*sell"]
        
        cautious_phrases = [
            "cautious", "limited upside", "concerns", "challenges", "daunting task", 
            "execution risk", "headwinds", "pressures", "already rallied", "priced in"
        ]
        
        bullish_score = sum(1 for rating in bullish_ratings if rating in text_lower)
        neutral_score = sum(1 for rating in neutral_ratings if rating in text_lower)
        bearish_score = sum(1 for rating in bearish_ratings if rating in text_lower)
        
        positive_action_score = sum(1 for action in positive_actions if re.search(action, text_lower))
        negative_action_score = sum(1 for action in negative_actions if re.search(action, text_lower))
        
        caution_score = sum(1 for phrase in cautious_phrases if phrase in text_lower)
        
        net_positive = bullish_score + positive_action_score
        net_negative = bearish_score + negative_action_score + (caution_score * 0.5)  # Weight caution as half-negative
        net_neutral = neutral_score
        
        if net_positive > net_negative and net_positive > net_neutral:
            if caution_score > 1:  # High caution reduces to cautiously positive
                return "Cautiously Positive"
            return "Positive"
        elif net_negative > net_positive and net_negative > net_neutral:
            return "Negative"
        elif net_neutral > 0 or caution_score > 2:  # High caution or explicit neutral ratings
            return "Neutral/Cautious"
        else:
            return "Mixed"

class FinancialNLPPipeline:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.ner = FinancialNER(spacy_model)
        self.event_detector = EventCatalystDetector()
        
        self.sentiment_analyzer = None
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="ProsusAI/finbert",
                                             tokenizer="ProsusAI/finbert")
        except Exception as e:
            print(f"Could not load FinBERT model: {e}")
            print("Using rule-based sentiment analysis instead")
    
    def process_article(self, article_text: str, title: str = "") -> FinancialEvent:
        full_text = f"{title} {article_text}".strip()
        
        entities = self.ner.extract_entities(full_text)
        
        detected_events = self.event_detector.detect_events(full_text)
        
        if not detected_events:
            return FinancialEvent(
                catalyst="No specific catalyst identified",
                entities=self._flatten_entities(entities),
                sentiment="Neutral",
                confidence=0.0,
                event_type="UNKNOWN"
            )
        
        primary_event = max(detected_events, key=lambda x: x["confidence"])
        
        if self.sentiment_analyzer:
            try:
                sentiment_result = self.sentiment_analyzer(full_text[:512])  
                sentiment = f"{sentiment_result[0]['label']} ({sentiment_result[0]['score']:.2f})"
            except:
                sentiment = self.event_detector.determine_sentiment(
                    primary_event["event_type"], full_text, self._flatten_entities(entities)
                )
        else:
            sentiment = self.event_detector.determine_sentiment(
                primary_event["event_type"], full_text, self._flatten_entities(entities)
            )
        
        catalyst = self._generate_catalyst_description(primary_event, entities)
        
        return FinancialEvent(
            catalyst=catalyst,
            entities=self._flatten_entities(entities),
            sentiment=sentiment,
            confidence=primary_event["confidence"],
            event_type=primary_event["event_type"],
            timestamp=datetime.now()
        )
    
    def _flatten_entities(self, entities: Dict[str, List[str]]) -> List[str]:
        flat_entities = []
        for entity_list in entities.values():
            flat_entities.extend(entity_list)
        return flat_entities
    
    def _generate_catalyst_description(self, event: Dict, entities: Dict[str, List[str]]) -> str:
        event_type = event["event_type"]
        matched_text = event["matched_text"]
        
        if event_type == "ANALYST_ACTION":
            return self._generate_analyst_catalyst(matched_text, entities)
        
        if event_type == "INVESTOR_DAY":
            company = self._get_primary_company(entities)
            if company:
                return f"{company} investor day and strategic outlook"
            return "Company investor day and strategic planning event"
        
        relevant_entities = self._get_relevant_entities_for_event(event_type, entities)
        
        if relevant_entities:
            main_entity = relevant_entities[0]
            return f"{main_entity}: {matched_text}"
        else:
            return f"{event_type.replace('_', ' ').title()}: {matched_text}"
    
    def _generate_analyst_catalyst(self, matched_text: str, entities: Dict[str, List[str]]) -> str:
        company = self._get_primary_company(entities)
        analyst_firm = self._get_analyst_firm(entities)
        
        if "price target" in matched_text.lower() or "target" in matched_text.lower():
            action = "price target update"
        elif any(word in matched_text.lower() for word in ["upgrade", "downgrade"]):
            action = "rating change"
        elif "rating" in matched_text.lower():
            action = "rating reiteration"
        else:
            action = "analyst action"
        
        # Build catalyst description
        if company and analyst_firm:
            return f"{analyst_firm} {action} on {company}"
        elif company:
            return f"Analyst {action} on {company}"
        elif analyst_firm:
            return f"{analyst_firm} analyst action"
        else:
            return f"Analyst {action}"
    
    def _get_primary_company(self, entities: Dict[str, List[str]]) -> str:
        company_types = ["TECH_COMPANY", "ENERGY_COMPANY", "PHARMACEUTICAL", "AUTOMOTIVE", 
                        "AIRLINE", "RETAIL", "ORG", "ORGANIZATION"]
        
        for entity_type in company_types:
            if entity_type in entities and entities[entity_type]:
                return entities[entity_type][0]
        
        if "PRODUCT" in entities:  # spaCy sometimes labels tickers as PRODUCT
            for entity in entities["PRODUCT"]:
                if re.match(r'^[A-Z]{2,5}]$', entity):
                    return entity
        
        return ""

    def _get_analyst_firm(self, entities: Dict[str, List[str]]) -> str:
        if "INVESTMENT_BANK" in entities and entities["INVESTMENT_BANK"]:
            return entities["INVESTMENT_BANK"][0]
        return ""

    def _get_relevant_entities_for_event(self, event_type: str, entities: Dict[str, List[str]]) -> List[str]:
        relevant_entities = []
        if event_type in ["INTEREST_RATE", "REGULATORY_POLICY"]:
            if "CENTRAL_BANK" in entities:
                relevant_entities.extend(entities["CENTRAL_BANK"])
            if "REGULATORY_BODY" in entities:
                relevant_entities.extend(entities["REGULATORY_BODY"])
            if "COUNTRY" in entities:
                 relevant_entities.extend(entities["COUNTRY"])
        elif event_type in ["PRODUCTION_CHANGE", "CRYPTO_EVENTS"]:
             if "COMMODITY" in entities:
                 relevant_entities.extend(entities["COMMODITY"])
             if "ORGANIZATION" in entities: # OPEC
                 relevant_entities.extend(entities["ORGANIZATION"])
             if "CURRENCY" in entities: # Crypto events might involve currencies
                 relevant_entities.extend(entities["CURRENCY"])
             if "CRYPTOCURRENCY" in entities:
                 relevant_entities.extend(entities["CRYPTOCURRENCY"])
             elif event_type in ["EARNINGS_REPORT", "MERGER_ACQUISITION", "IPO_LISTING", "DIVIDEND_BUYBACK", "BANKRUPTCY_DISTRESS", "INVESTOR_DAY", "ANALYST_ACTION", "ESG_SUSTAINABILITY"]:
                      # For company-specific events, prioritize company/organization entities
                company = self._get_primary_company(entities)
                if company:
                    relevant_entities.append(company)
                # Also include relevant financial organizations for M&A, IPOs, Analyst Actions
                if event_type in ["MERGER_ACQUISITION", "IPO_LISTING", "ANALYST_ACTION"]:
                    if "INVESTMENT_BANK" in entities:
                        relevant_entities.extend(entities["INVESTMENT_BANK"])
                if "ORG" in entities:
                    relevant_entities.extend(entities["ORG"])
                if "ORGANIZATION" in entities:
                    relevant_entities.extend(entities["ORGANIZATION"])
                if "SECTOR" in entities: # ESG might be sector specific
                    relevant_entities.extend(entities["SECTOR"])
             elif event_type == "CREDIT_RATING":
                if "RATING_AGENCY" in entities:
                    relevant_entities.extend(entities["RATING_AGENCY"])
                company = self._get_primary_company(entities)
                if company:
                    relevant_entities.append(company)
             elif event_type == "GEOPOLITICAL":
                if "COUNTRY" in entities:
                    relevant_entities.extend(entities["COUNTRY"])
                if "INTERNATIONAL_ORG" in entities:
                    relevant_entities.extend(entities["INTERNATIONAL_ORG"])
                if "COMMODITY" in entities: # Geopolitical events often impact commodities
                    relevant_entities.extend(entities["COMMODITY"])
        if not relevant_entities:
                for entity_type in ["ORGANIZATION", "PERSON", "GPE", "LOC", "PRODUCT", "EVENT", "NORP"]:
                    if entity_type in entities:
                        relevant_entities.extend(entities[entity_type])

        return relevant_entities


    def process_multiple_articles(self, articles: List[Dict[str, str]]) -> List[FinancialEvent]:
        results = []
        for article in articles:
            title = article.get("title", "")
            content = article.get("content", "")
            result = self.process_article(content, title)
            results.append(result)
        return results



# Example Testing
def main():
    pipeline = FinancialNLPPipeline()
    
    sample_articles = [
        {
            "title": "Federal Reserve Raises Interest Rates by 0.75%",
            "content": "The Federal Reserve announced a 0.75 percentage point increase in interest rates today, citing ongoing inflation concerns. The decision affects borrowing costs across the economy and is expected to impact real estate and growth stocks negatively while benefiting banks."
        },
        {
            "title": "OPEC Announces Major Oil Production Cut",
            "content": "OPEC and its allies agreed to cut oil production by 2 million barrels per day, the largest reduction since 2020. The decision is expected to boost crude oil prices but could negatively impact airlines and other transportation companies due to higher fuel costs."
        },
        {
            "title": "Tesla Reports Record Quarterly Earnings",
            "content": "Tesla exceeded Wall Street expectations with record quarterly earnings, reporting revenue growth of 35% year-over-year. The electric vehicle manufacturer raised its full-year production guidance and announced plans for new factory expansions."
        },
        {
            "title": " ",
            "content": """
            On May 23, Morgan Stanley analyst Meta Marshall raised the price target on Coherent Corp. (NYSE:COHR) to $74 from $70, but reiterated her cautious stance with a Hold rating. 
            The company is expected to hold its investor day on May 28, where Marshall expects it will present a long-term growth plan. This strategic plan is expected to include a roadmap 
            to achieve over 40% gross margin, more than $6 in EPS, and outline the revenue growth trajectory from 2025 to 2028.

            While a favorable outlook at the investor day should catalyze the share price, Marshall sees limited upside potential, as the stock has already rallied substantially over the past month. 
            Another reason for her cautious view is that she believes achieving gross margins above 40% will be a daunting task for the company. This is due to the impacts of tariffs and the lack of 
            operating leverage in some business segments, which, in her view, will require more time to improve. She also expressed concerns about execution challenges in certain areas, such as the SiC business, networking technologies, and the vertical integration of the datacom business. Therefore, she maintained her Hold rating
            
            Coherent Corp. (NYSE:COHR) specializes in the development and manufacturing of engineered materials, networking products, optoelectronic components, and optical and laser systems for various industries, including industrial, communications, electronics, and instrumentation. The company maintains a strong foothold in the optical communications market, particularly with its advanced solutions for data centers, such as datacom optical transceivers.
            While we acknowledge the potential of COHR as an investment, our conviction lies in the belief that some AI stocks hold greater promise for delivering higher returns and have limited downside risk. If you are looking for an AI stock that is more promising than COHR and that has 100x upside potential, check out our report about the cheapest AI stock.
            """
        }
    ]
    
    print("Financial News Analysis Results:")
    print("=" * 50)
    
    for i, article in enumerate(sample_articles, 1):
        result = pipeline.process_article(article["content"], article["title"])
        
        print(f"\nArticle {i}: {article['title']}")
        print("-" * 30)
        print(f"Catalyst: {result.catalyst}")
        print(f"Entities: {', '.join(result.entities)}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Event Type: {result.event_type}")
        print(f"Confidence: {result.confidence:.2f}")
    
    batch_results = pipeline.process_multiple_articles(sample_articles)
    
    df_data = []
    for result in batch_results:
        df_data.append({
            "catalyst": result.catalyst,
            "entities_count": len(result.entities),
            "sentiment": result.sentiment,
            "event_type": result.event_type,
            "confidence": result.confidence
        })
    
    df = pd.DataFrame(df_data)
    print(f"\n\nBatch Processing Summary:")
    print(df.groupby("event_type").size().to_string())

if __name__ == "__main__":
    main()