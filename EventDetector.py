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
                    })

        confidence_counts = {}
        for event in events:
            event_type = event["event_type"]
            confidence_counts[event_type] = confidence_counts.get(event_type, 0) + 1

        for event in events:
            event["confidence"] = confidence_counts[event["event_type"]]

        event_list = (set(event["event_type"] for event in events)) 

        combined_events = []
        for event in event_list:
            combined_events.append({
                "event_type": event,
                "matched_text": [e["matched_text"] for e in events if e["event_type"] == event],
                "confidence": ([e["confidence"] for e in events if e["event_type"] == event][0])
            })

        return combined_events
    
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
