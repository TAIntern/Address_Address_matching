"""
Optimized Address Matching System

Enhanced version with:
- Batch processing capabilities
- Better error handling and logging
- Connection pooling and retry logic
- Memory-efficient processing
- Configurable field mappings for different indices
- Progress tracking and statistics
- Export capabilities

Required packages:
    pip install usaddress unidecode rapidfuzz requests pydantic tqdm
"""

import logging
import re 
import json
import time
import csv
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass, field
from enum import Enum
import os
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import threading
from collections import defaultdict

import requests
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from unidecode import unidecode
from rapidfuzz import fuzz
import difflib
from tqdm import tqdm

# ------------------------------------------------------------------
#  Helper: split an embedded apartment/unit out of a street string
# ------------------------------------------------------------------
APT_PAT = re.compile(r"\b(?:apt|unit|ste|suite|#)\s*([a-z0-9-]+)\b", re.I)

def split_street_unit(text: str) -> tuple[str, str]:
    """Extract apartment/unit from street string."""
    if not text:
        return text, ""
    m = APT_PAT.search(text)
    if not m:
        return text, ""
    unit = m.group(1).upper()
    street = (text[: m.start()] + text[m.end():]).strip()
    return street, unit

# ──────────────────────────────────────────────────────────────
#  USADDRESS IMPORT
# ──────────────────────────────────────────────────────────────
USADDRESS_AVAILABLE = False
try:
    import usaddress
    USADDRESS_AVAILABLE = True
    print("✅ usaddress loaded successfully")
except ImportError:
    logging.warning("⚠️  usaddress not available — using fallback parsing.")

# ──────────────────────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────────────────────
def setup_logging(level=logging.INFO, log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%Y‑%m‑%d %H:%M:%S",
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ──────────────────────────────────────────────────────────────
#  CONFIGURATION AND DATA CLASSES
# ──────────────────────────────────────────────────────────────
class MatchStrategy(Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    HYBRID = "hybrid"

@dataclass
class IndexFieldMapping:
    """Field mappings for different OpenSearch indices"""
    # Source index fields (what we're matching against)
    house_field: str = "HOUSE"
    street_field: str = "STREET"
    city_field: str = "CITY"
    state_field: str = "STATE"
    zip_field: str = "ZIP_CODE"
    unit_field: str = "APTNBR"
    address_field: str = "ADDRESS"
    
    # Additional fields for other indices
    predir_field: str = "PREDIR"
    strtype_field: str = "STRTYPE"
    


@dataclass
class MatchConfig:
    # Component weights must sum to 1.0 - optimized for exact matching
    street_weight: float = 0.40  # Increased for better street matching
    house_weight: float = 0.25   # Increased for exact house number matching
    city_weight: float = 0.15
    zip_weight: float = 0.10     # Reduced since ZIP can vary
    state_weight: float = 0.05   # Reduced since state is usually correct
    unit_weight: float = 0.05

    min_similarity_threshold: float = 70.0
    exact_threshold: float = 98.0  # Raised threshold for exact matches

    max_results: int = 10
    timeout: int = 30
    max_retries: int = 5
    backoff_factor: float = 0.5

    require_house: bool = False
    require_zip: bool = False
    
    # Batch processing settings
    batch_size: int = 100
    max_workers: int = 4
    
    # Field mappings
    field_mapping: IndexFieldMapping = field(default_factory=IndexFieldMapping)

    def __post_init__(self):
        total_weight = (
            self.street_weight + self.house_weight + self.city_weight + 
            self.zip_weight + self.state_weight + self.unit_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    @classmethod
    def from_file(cls, path: str):
        """Load config from a JSON or YAML file."""
        ext = os.path.splitext(path)[-1].lower()
        with open(path, "r") as f:
            if ext in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(**data)

@dataclass
class ParsedAddress:
    house: str = ""
    street: str = ""
    unit: str = ""
    city: str = ""
    state: str = ""
    zip: str = ""
    raw: Dict[str, str] = field(default_factory=dict)

    def normalized(self) -> str:
        parts = [self.house, self.street, self.city, self.state, self.zip]
        if self.unit:
            parts.insert(2, f"#{self.unit}")
        return " ".join(filter(None, parts))

@dataclass
class MatchResult:
    query: str
    matched: Optional[Dict[str, Any]]
    confidence: float
    similarity: float
    es_score: float
    component_scores: Dict[str, float]
    ms: int
    error: Optional[str] = None
    index: str = ""

    @property
    def high_conf(self) -> bool:
        return self.confidence >= 85

    @property
    def exact(self) -> bool:
        # More strict exact match criteria
        return (self.similarity >= 95 and 
                self.confidence >= 95 and
                self._is_truly_exact())
    
    def _is_truly_exact(self) -> bool:
        """Check if this is truly an exact match based on critical components"""
        if not self.matched:
            return False
            
        # For exact match, house number and street must be very high scores
        house_score = self.component_scores.get('house', 0)
        street_score = self.component_scores.get('street', 0)
        
        # Critical components must be near perfect
        if house_score > 0 and house_score < 95:  # If house exists, it must be exact
            return False
        if street_score < 85:  # Street must be very close (slightly more lenient)
            return False
            
        # If ZIP exists in both, it should match reasonably well
        zip_score = self.component_scores.get('zip', 0)
        if zip_score > 0 and zip_score < 80:  # More lenient on ZIP (cross-borough issues)
            return False
            
        return True

@dataclass
class BatchStats:
    """Statistics for batch processing"""
    total_processed: int = 0
    successful_matches: int = 0
    high_confidence_matches: int = 0
    exact_matches: int = 0
    errors: int = 0
    avg_confidence: float = 0.0
    avg_processing_time: float = 0.0
    error_details: List[str] = field(default_factory=list)

# ──────────────────────────────────────────────────────────────
#  SOUNDEX AND NORMALIZER
# ──────────────────────────────────────────────────────────────
def soundex(name):
    """Optimized Soundex implementation."""
    if not name:
        return "0000"
    name = name.upper()
    soundex = name[0]
    mapping = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6"}
    
    for char in name[1:]:
        for key in mapping:
            if char in key:
                code = mapping[key]
                if code != soundex[-1]:
                    soundex += code
                break
    
    return soundex[:4].ljust(4, "0")

class AddressNormalizer:
    """Enhanced address normalizer with caching"""
    
    _cache = {}
    _cache_lock = threading.Lock()
    
    STREET_ABB = {
        "north": "n", "south": "s", "east": "e", "west": "w",
        "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
        "street": "st", "avenue": "ave", "boulevard": "blvd", "drive": "dr",
        "lane": "ln", "road": "rd", "court": "ct", "place": "pl",
    }
    
    DIR_ABB = {
        "north": "n", "south": "s", "east": "e", "west": "w",
        "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    }

    @classmethod
    def normalize(cls, txt: str, use_cache: bool = True) -> str:
        if not txt:
            return ""
            
        # Check cache first
        if use_cache:
            with cls._cache_lock:
                if txt in cls._cache:
                    return cls._cache[txt]
        
        original = txt
        txt = unidecode(txt).lower()
        txt = re.sub(r"[^\w\s\-#/]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        

        
        # Replace street type abbreviations
        for full, abbr in {**cls.STREET_ABB, **cls.DIR_ABB}.items():
            txt = re.sub(rf"\b{full}\b", abbr, txt)
        
        # Collapse ordinals (e.g., "92nd" → "92")
        txt = re.sub(r"\b(\d+)[-_]?(st|nd|rd|th|ht|tt|thh|ndd|rdd|h|t)\b", r"\1", txt)
        
        # Cache result
        if use_cache:
            with cls._cache_lock:
                cls._cache[original] = txt
                # Limit cache size
                if len(cls._cache) > 10000:
                    cls._cache.clear()
        
        return txt



    @staticmethod
    def normalize_unit(unit: str) -> str:
        if not unit:
            return ""
        unit = unit.lower().replace(" ", "")
        unit = re.sub(r"^(apt|unit|ste|suite|#)+", "", unit)
        return unit.upper()

# ──────────────────────────────────────────────────────────────
#  ENHANCED ADDRESS PARSER
# ──────────────────────────────────────────────────────────────
class AddressParser:
    @staticmethod
    def parse(addr: str) -> ParsedAddress:
        if USADDRESS_AVAILABLE:
            try:
                try:
                    parsed, address_type = usaddress.tag(addr)
                except usaddress.RepeatedLabelError as e:
                    parsed_list = usaddress.parse(addr)
                    parsed = {}
                    for component, label in parsed_list:
                        if label not in parsed:
                            parsed[label] = component

                # City alias normalization
                if (parsed.get("PlaceName", "").strip().upper() == "NY" and 
                    parsed.get("StateName", "").strip().upper() == "NY"):
                    parsed["PlaceName"] = "NEW YORK"

                return ParsedAddress(
                    house=parsed.get("AddressNumber", "").strip(),
                    street=" ".join(filter(None, [
                        parsed.get("StreetNamePreDirectional", ""),
                        parsed.get("StreetName", ""),
                        parsed.get("StreetNamePostType", "")
                    ])).strip(),
                    unit=parsed.get("OccupancyIdentifier", "").strip().upper(),
                    city=parsed.get("PlaceName", "").strip(),
                    state=parsed.get("StateName", "").strip(),
                    zip=parsed.get("ZipCode", "").strip(),
                    raw=parsed,
                )
            except Exception as e:
                logger.warning(f"usaddress parsing failed for '{addr}': {e}")

        # Fallback parsing with better error handling
        return AddressParser._fallback_parse(addr)

    @staticmethod
    def _fallback_parse(addr: str) -> ParsedAddress:
        """Enhanced fallback parser"""
        try:
            addr_norm = AddressNormalizer.normalize(addr)
            addr = addr_norm

            # Extract ZIP
            zip_match = re.search(r"\b(\d{5})(?:-\d{4})?\b", addr)
            zip_code = zip_match.group(1) if zip_match else ""
            if zip_code:
                addr = addr.replace(zip_code, "").strip()

            tokens = addr.split()
            if not tokens:
                return ParsedAddress(zip=zip_code)

            # Extract house number
            house = ""
            if tokens and tokens[0].isdigit():
                house = tokens.pop(0)

            # Extract state
            US_STATES = {
                "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in","ia","ks",
                "ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv","nh","nj","nm","ny",
                "nc","nd","oh","ok","or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa","wv",
                "wi","wy",
            }
            state = ""
            if tokens and tokens[-1] in US_STATES:
                state = tokens.pop()

            # Extract city
            city_tokens = []
            while tokens:
                last = tokens[-1]
                if last in {"n","s","e","w","ne","nw","se","sw","st","ave","blvd","rd","dr","ln"}:
                    break
                if re.search(r"\d", last):
                    break
                city_tokens.insert(0, tokens.pop())
                if len(city_tokens) == 2:
                    break
            city = " ".join(city_tokens)

            # City alias normalization
            if city.upper() == "NY" and state.upper() == "NY":
                city = "NEW YORK"

            # Extract unit
            unit = ""
            for i, tok in enumerate(tokens):
                if tok.startswith("#"):
                    unit = tok.lstrip("#")
                    tokens.pop(i)
                    break
                if tok in {"apt", "unit", "suite", "ste"} and i + 1 < len(tokens):
                    unit = tokens[i + 1].lstrip("#")
                    del tokens[i : i + 2]
                    break

            # Remaining tokens form the street
            street = " ".join(tokens)
            unit = unit.upper()

            return ParsedAddress(
                house=house,
                street=street,
                unit=unit,
                city=city,
                state=state,
                zip=zip_code,
                raw={
                    "AddressNumber": house,
                    "Street": street,
                    "OccupancyIdentifier": unit,
                    "PlaceName": city,
                    "StateName": state,
                    "ZipCode": zip_code,
                }
            )
        except Exception as e:
            logger.error(f"Fallback parsing failed for '{addr}': {e}")
            return ParsedAddress()

# ──────────────────────────────────────────────────────────────
#  ENHANCED ELASTICSEARCH CLIENT
# ──────────────────────────────────────────────────────────────
class ESClient:
    def __init__(self, url: str, auth: HTTPBasicAuth, cfg: MatchConfig):
        self.url = url.rstrip("/")
        self.auth = auth
        self.cfg = cfg
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create session with enhanced retry logic"""
        session = requests.Session()
        retry_strategy = Retry(
            total=self.cfg.max_retries,
            backoff_factor=self.cfg.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def search(self, index: str, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced search with better error handling"""
        try:
            resp = self.session.post(
                f"{self.url}/{index}/_search",
                auth=self.auth,
                json=body,
                timeout=self.cfg.timeout,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            
            if "error" in data:
                logger.error(f"OpenSearch error: {data['error']}")
                return []
                
            return data.get("hits", {}).get("hits", [])
        except requests.exceptions.Timeout:
            logger.error(f"Search timeout for index {index}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Search request failed for index {index}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise

    def msearch(self, index: str, queries: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Multi-search for batch processing"""
        if not queries:
            return []
            
        # Build msearch request body
        body_lines = []
        for query in queries:
            body_lines.append(json.dumps({"index": index}))
            body_lines.append(json.dumps(query))
        
        body = "\n".join(body_lines) + "\n"
        
        try:
            resp = self.session.post(
                f"{self.url}/_msearch",
                auth=self.auth,
                data=body,
                timeout=self.cfg.timeout * 2,  # Longer timeout for batch
                headers={"Content-Type": "application/x-ndjson"},
            )
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for response in data.get("responses", []):
                if "error" in response:
                    logger.warning(f"Multi-search error: {response['error']}")
                    results.append([])
                else:
                    results.append(response.get("hits", {}).get("hits", []))
            
            return results
        except Exception as e:
            logger.error(f"Multi-search failed: {e}")
            raise

# ──────────────────────────────────────────────────────────────
#  ENHANCED ADDRESS MATCHER
# ──────────────────────────────────────────────────────────────
class AddressMatcher:
    def __init__(self, es_url: str, auth: HTTPBasicAuth, cfg: MatchConfig = None):
        self.cfg = cfg or MatchConfig()
        self.es = ESClient(es_url, auth, self.cfg)
        self.stats = BatchStats()

    def _build_query(self, p: ParsedAddress) -> Dict[str, Any]:
        """Build OpenSearch query optimized for exact matching first"""
        mapping = self.cfg.field_mapping
        must, should, boost = [], [], []

        # Street matching - prioritize exact matches
        if p.street:
            must.append({
                "bool": {
                    "should": [
                        # Exact phrase match gets highest boost
                        {"match_phrase": {mapping.street_field: {"query": p.street, "boost": 5}}},
                        # Exact term match for street
                        {"term": {f"{mapping.street_field}.keyword": {"value": p.street.upper(), "boost": 4}}},
                        # Fuzzy match as fallback
                        {"match": {mapping.street_field: {"query": p.street, "fuzziness": "AUTO", "boost": 2}}},
                    ]
                }
            })
        
        # House number - must be exact for high confidence
        if p.house:
            must.append({
                "bool": {
                    "should": [
                        {"term": {mapping.house_field: {"value": p.house, "boost": 5}}},
                        {"term": {f"{mapping.house_field}.keyword": {"value": p.house, "boost": 4}}},
                    ]
                }
            })
        
        # ZIP code - exact match preferred
        if p.zip:
            boost.append({"term": {mapping.zip_field: {"value": p.zip, "boost": 6}}})
        
        # City - MUST match for geographic accuracy (moved from should to must)
        if p.city:
            must.append({
                "bool": {
                    "should": [
                        {"term": {f"{mapping.city_field}.keyword": {"value": p.city.upper(), "boost": 3}}},
                        {"match": {mapping.city_field: {"query": p.city, "fuzziness": "AUTO", "boost": 2}}},
                    ]
                }
            })
        
        # State - MUST be exact for geographic accuracy (moved from should to must)
        if p.state:
            must.append({"term": {mapping.state_field: {"value": p.state.upper(), "boost": 3}}})
        
        # Unit - prefer exact match
        if p.unit:
            should.append({
                "bool": {
                    "should": [
                        {"term": {mapping.unit_field: {"value": p.unit, "boost": 3}}},
                        {"match": {mapping.unit_field: {"query": p.unit, "fuzziness": 1, "boost": 1}}},
                    ]
                }
            })

        return {
            "size": self.cfg.max_results,
            "query": {
                "bool": {
                    "must": must,
                    "should": should + boost,
                    "minimum_should_match": 1 if should else 0,
                }
            },
            # Sort by score to get best matches first
            "sort": [
                {"_score": {"order": "desc"}},
            ]
        }

    def _calculate_component_scores(self, p: ParsedAddress, src: Dict[str, Any]) -> Dict[str, float]:
        """Calculate component-wise similarity scores"""
        mapping = self.cfg.field_mapping
        scores = {}

        # Street scoring with unit extraction
        if p.street:
            rec_full_street = " ".join(filter(None, [
                str(src.get(mapping.predir_field, "")).strip(),
                str(src.get(mapping.street_field, "")).strip(),
                str(src.get(mapping.strtype_field, "")).strip(),
            ])).upper()

            q_street, q_unit_in_street = split_street_unit(p.street)
            q_street_norm = AddressNormalizer.normalize(q_street)
            r_street_norm = AddressNormalizer.normalize(rec_full_street)
            
            # Check for exact match first
            if q_street_norm == r_street_norm:
                scores["street"] = 100.0
            else:
                # Use token_sort_ratio for fuzzy matching
                scores["street"] = fuzz.token_sort_ratio(q_street.upper(), rec_full_street)
            
            # Soundex similarity
            q_soundex = soundex(q_street)
            r_soundex = soundex(rec_full_street)
            scores["street_soundex"] = 100 if q_soundex == r_soundex else 0

            # Unit scoring
            r_unit_in_street = ""
            if src.get(mapping.street_field):
                _, r_unit_in_street = split_street_unit(str(src[mapping.street_field]).lower())

            unit_query = AddressNormalizer.normalize_unit(p.unit or q_unit_in_street)
            unit_rec = AddressNormalizer.normalize_unit(
                src.get(mapping.unit_field) or r_unit_in_street
            )
            if unit_query and unit_rec:
                scores["unit"] = fuzz.ratio(unit_query, unit_rec)

        # House number scoring - must be exact for high confidence
        if p.house and src.get(mapping.house_field):
            house_query = p.house.strip()
            house_record = str(src[mapping.house_field]).strip()
            
            # Exact match gets 100%
            if house_query == house_record:
                scores["house"] = 100.0
            # Fuzzy match for slight variations
            else:
                ratio = fuzz.ratio(house_query, house_record)
                # Be more strict with house numbers - small differences matter a lot
                if ratio < 90:
                    scores["house"] = max(0, ratio - 20)  # Penalize non-exact house matches
                else:
                    scores["house"] = ratio
        
        if p.city and src.get(mapping.city_field):
            scores["city"] = fuzz.partial_ratio(p.city.lower(), str(src[mapping.city_field]).lower())
        
        if p.zip and src.get(mapping.zip_field):
            scores["zip"] = fuzz.ratio(p.zip, str(src[mapping.zip_field]))
        
        if p.state and src.get(mapping.state_field):
            scores["state"] = fuzz.ratio(p.state.upper(), str(src[mapping.state_field]).upper())

        return scores

    def _calculate_weighted_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate weighted confidence score with strict exact match criteria"""
        weight_map = {
            "street": self.cfg.street_weight,
            "house": self.cfg.house_weight,
            "city": self.cfg.city_weight,
            "zip": self.cfg.zip_weight,
            "state": self.cfg.state_weight,
            "unit": self.cfg.unit_weight,
        }

        # For exact matching, be more strict about critical components
        house_score = scores.get("house", 0)
        street_score = scores.get("street", 0)
        
        # If house number exists but doesn't match well, heavily penalize
        if house_score > 0 and house_score < 80:
            return min(house_score, 70.0)  # Cap at 70% for poor house matches
        
        # Street is critical - if poor, penalize heavily
        if street_score < 70:
            return min(street_score, 60.0)  # Cap at 60% for poor street matches

        # Exclude unreliable ZIP scores and missing city data
        if scores.get("city", 0) == 0:
            weight_map["city"] = 0.0
        if scores.get("zip", 0) < 50:
            weight_map.pop("zip", None)

        total_score = 0.0
        total_weight = 0.0
        
        for component, score in scores.items():
            if component in weight_map:
                weight = weight_map[component]
                total_score += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0
            
        weighted_score = total_score / total_weight
        
        # Additional penalty for missing critical components in high-scoring matches
        if weighted_score > 85:
            # If claiming high confidence, house and street must be very good
            if house_score > 0 and house_score < 90:
                weighted_score = min(weighted_score, 75.0)
            if street_score < 85:
                weighted_score = min(weighted_score, 80.0)
        
        return weighted_score

    def _rank_results(self, query: str, hits: List[Dict[str, Any]], parsed: ParsedAddress) -> List[MatchResult]:
        """Rank and filter results with exact match prioritization"""
        normalized_query = AddressNormalizer.normalize(query)
        results = []
        exact_matches = []
        
        for hit in hits:
            src = hit["_source"]
            es_score = hit["_score"]
            
            component_scores = self._calculate_component_scores(parsed, src)
            confidence = self._calculate_weighted_confidence(component_scores)
            
            # Calculate overall similarity
            address_field = src.get(self.cfg.field_mapping.address_field, "")
            similarity = fuzz.ratio(normalized_query, AddressNormalizer.normalize(address_field))
            
            if confidence >= self.cfg.min_similarity_threshold:
                result = MatchResult(
                    query=query,
                    matched=src,
                    confidence=confidence,
                    similarity=similarity,
                    es_score=es_score,
                    component_scores=component_scores,
                    ms=0,
                )
                
                # Separate exact matches for priority ranking
                if result.exact:
                    exact_matches.append(result)
                else:
                    results.append(result)
        
        # Sort exact matches by confidence, then regular matches
        exact_matches.sort(key=lambda r: r.confidence, reverse=True)
        results.sort(key=lambda r: r.confidence, reverse=True)
        
        # Return exact matches first, then regular matches
        return exact_matches + results

    def match(self, query: str, index: str = "addresses") -> MatchResult:
        """Match a single address"""
        start_time = time.time()
        
        if not query.strip():
            return MatchResult(query, None, 0, 0, 0, {}, 0, "Empty query", index)

        try:
            parsed = AddressParser.parse(query)
            
            if self.cfg.require_house and not parsed.house:
                return MatchResult(query, None, 0, 0, 0, {}, 0, "House number required", index)
            if self.cfg.require_zip and not parsed.zip:
                return MatchResult(query, None, 0, 0, 0, {}, 0, "ZIP code required", index)

            es_query = self._build_query(parsed)
            hits = self.es.search(index, es_query)
            
            ranked_results = self._rank_results(query, hits, parsed)
            best_result = ranked_results[0] if ranked_results else MatchResult(
                query, None, 0, 0, 0, {}, 0, "No good match", index
            )
            
            best_result.ms = int((time.time() - start_time) * 1000)
            best_result.index = index
            
            return best_result
            
        except Exception as e:
            logger.error(f"Match failed for query '{query}': {e}")
            return MatchResult(query, None, 0, 0, 0, {}, 0, str(e), index)

    def batch_match(self, queries: List[str], index: str = "addresses") -> Iterator[MatchResult]:
        """Batch match multiple addresses with progress tracking"""
        if not queries:
            return

        logger.info(f"Starting batch match of {len(queries)} addresses against index '{index}'")
        
        with tqdm(total=len(queries), desc="Matching addresses") as pbar:
            # Process in batches
            for i in range(0, len(queries), self.cfg.batch_size):
                batch = queries[i:i + self.cfg.batch_size]
                
                # Parse all addresses in batch
                parsed_addresses = []
                for query in batch:
                    if query.strip():
                        parsed_addresses.append((query, AddressParser.parse(query)))
                    else:
                        yield MatchResult(query, None, 0, 0, 0, {}, 0, "Empty query", index)
                        pbar.update(1)
                        continue

                # Build queries for multi-search
                es_queries = []
                for _, parsed in parsed_addresses:
                    es_queries.append(self._build_query(parsed))

                try:
                    # Execute multi-search
                    if es_queries:
                        batch_results = self.es.msearch(index, es_queries)
                        
                        # Process results
                        for j, ((query, parsed), hits) in enumerate(zip(parsed_addresses, batch_results)):
                            start_time = time.time()
                            
                            ranked_results = self._rank_results(query, hits, parsed)
                            best_result = ranked_results[0] if ranked_results else MatchResult(
                                query, None, 0, 0, 0, {}, 0, "No good match", index
                            )
                            
                            best_result.ms = int((time.time() - start_time) * 1000)
                            best_result.index = index
                            
                            # Update statistics
                            self._update_stats(best_result)
                            
                            yield best_result
                            pbar.update(1)
                            
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Yield error results for the batch
                    for query, _ in parsed_addresses:
                        yield MatchResult(query, None, 0, 0, 0, {}, 0, str(e), index)
                        pbar.update(1)

    def _update_stats(self, result: MatchResult):
        """Update batch processing statistics"""
        self.stats.total_processed += 1
        
        if result.matched:
            self.stats.successful_matches += 1
            if result.high_conf:
                self.stats.high_confidence_matches += 1
            if result.exact:
                self.stats.exact_matches += 1
        
        if result.error:
            self.stats.errors += 1
            self.stats.error_details.append(result.error)
        
        # Update averages
        if self.stats.successful_matches > 0:
            total_conf = getattr(self.stats, '_total_confidence', 0) + result.confidence
            self.stats._total_confidence = total_conf
            self.stats.avg_confidence = total_conf / self.stats.successful_matches
        
        total_time = getattr(self.stats, '_total_time', 0) + result.ms
        self.stats._total_time = total_time
        self.stats.avg_processing_time = total_time / self.stats.total_processed

    def get_stats(self) -> BatchStats:
        """Get current batch processing statistics"""
        return self.stats

    def reset_stats(self):
        """Reset batch processing statistics"""
        self.stats = BatchStats()

# ──────────────────────────────────────────────────────────────
#  FACTORY AND UTILITIES
# ──────────────────────────────────────────────────────────────
def build_exact_match_config(index_type: str = "addresses") -> MatchConfig:
    """Create configuration optimized for exact matching"""
    cfg = MatchConfig()
    
    # Even more strict settings for exact matching
    cfg.min_similarity_threshold = 75.0
    cfg.exact_threshold = 98.0
    cfg.require_house = True  # Require house number for high confidence
    
    # Emphasize critical components even more
    cfg.street_weight = 0.45
    cfg.house_weight = 0.30
    cfg.city_weight = 0.10
    cfg.zip_weight = 0.10
    cfg.state_weight = 0.03
    cfg.unit_weight = 0.02
    

    
    return cfg

def build_matcher(
    es_url: str = None,
    user: str = None,
    pwd: str = None,
    cfg: MatchConfig = None,
    index_type: str = "addresses",
    exact_match_mode: bool = False
) -> AddressMatcher:
    """Build matcher with environment variable support"""
    es_url = es_url or os.environ.get("ES_URL", "https://search-addresses-f7voryraair5mrpw73ub2v3yg4.aos.us-east-1.on.aws")
    user = user or os.environ.get("ES_USER", "intern_3")
    pwd = pwd or os.environ.get("ES_PWD", "Trustscout1!")
    
    if cfg is None:
        if exact_match_mode:
            cfg = build_exact_match_config(index_type)
        else:
            cfg = MatchConfig()
    
    return AddressMatcher(es_url, HTTPBasicAuth(user, pwd), cfg)

def export_results_to_csv(results: List[MatchResult], filename: str):
    """Export match results to CSV"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'query', 'matched_address', 'confidence', 'similarity', 
            'es_score', 'processing_time_ms', 'error', 'index',
            'street_score', 'house_score', 'city_score', 'zip_score', 
            'state_score', 'unit_score'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            matched_addr = ""
            if result.matched:
                # Try to get the full address field
                addr_field = result.matched.get('ADDRESS') or result.matched.get('full_address')
                if addr_field:
                    matched_addr = addr_field
                else:
                    # Construct from components
                    parts = [
                        result.matched.get('HOUSE', result.matched.get('house_number', '')),
                        result.matched.get('STREET', result.matched.get('street_name', '')),
                        result.matched.get('CITY', result.matched.get('city', '')),
                        result.matched.get('STATE', result.matched.get('state', '')),
                        result.matched.get('ZIP', result.matched.get('zip_code', ''))
                    ]
                    matched_addr = ' '.join(filter(None, [str(p) for p in parts]))
            
            writer.writerow({
                'query': result.query,
                'matched_address': matched_addr,
                'confidence': f"{result.confidence:.2f}",
                'similarity': f"{result.similarity:.2f}",
                'es_score': f"{result.es_score:.2f}",
                'processing_time_ms': result.ms,
                'error': result.error or '',
                'index': result.index,
                'street_score': f"{result.component_scores.get('street', 0):.2f}",
                'house_score': f"{result.component_scores.get('house', 0):.2f}",
                'city_score': f"{result.component_scores.get('city', 0):.2f}",
                'zip_score': f"{result.component_scores.get('zip', 0):.2f}",
                'state_score': f"{result.component_scores.get('state', 0):.2f}",
                'unit_score': f"{result.component_scores.get('unit', 0):.2f}",
            })
    
    logger.info(f"Results exported to {filename}")

def main():
    """Enhanced CLI with batch processing support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Address Matcher for OpenSearch")
    parser.add_argument("address", nargs="*", help="Address(es) to match")
    parser.add_argument("--index", default="addresses", help="OpenSearch index name")
    parser.add_argument("--index-type", choices=["addresses"], 
                       default="addresses", help="Index type for field mapping")
    parser.add_argument("--batch-file", help="File containing addresses to match (one per line)")
    parser.add_argument("--output", help="Output CSV file for results")
    parser.add_argument("--config", help="Path to config file (JSON/YAML)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--logfile", help="Log file path")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--exact-match", action="store_true", help="Use exact match optimized configuration")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.logfile:
        setup_logging(logging.DEBUG if args.verbose else logging.INFO, args.logfile)
    
    # Load configuration
    cfg = None
    if args.config:
        cfg = MatchConfig.from_file(args.config)
    else:
        cfg = MatchConfig()
        cfg.batch_size = args.batch_size
        cfg.max_workers = args.workers
    
    # Build matcher
    matcher = build_matcher(cfg=cfg, index_type=args.index_type, exact_match_mode=args.exact_match)
    
    # Determine addresses to process
    addresses_to_match = []
    
    if args.batch_file:
        try:
            with open(args.batch_file, 'r', encoding='utf-8') as f:
                addresses_to_match = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(addresses_to_match)} addresses from {args.batch_file}")
        except Exception as e:
            logger.error(f"Failed to load batch file: {e}")
            return 1
    elif args.address:
        addresses_to_match = [" ".join(args.address)]
    else:
        try:
            addr = input("Enter address to match: ").strip()
            if addr:
                addresses_to_match = [addr]
        except (KeyboardInterrupt, EOFError):
            print()
            return 0
    
    if not addresses_to_match:
        print("No addresses provided")
        return 1
    
    # Process addresses
    results = []
    
    if len(addresses_to_match) == 1:
        # Single address
        result = matcher.match(addresses_to_match[0], args.index)
        results.append(result)
        
        if result.error:
            print(f"❌ {result.error}")
            return 1
        elif result.matched:
            addr_field = result.matched.get('ADDRESS') or result.matched.get('full_address', 'N/A')
            print(f"✅ {addr_field} ({result.confidence:.1f}%)")
            print(f"   Components: {result.component_scores}")
        else:
            print("❌ No match found")
    else:
        # Batch processing
        print(f"Processing {len(addresses_to_match)} addresses...")
        
        for result in matcher.batch_match(addresses_to_match, args.index):
            results.append(result)
        
        # Print summary statistics
        stats = matcher.get_stats()
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total Processed: {stats.total_processed}")
        print(f"   Successful Matches: {stats.successful_matches}")
        print(f"   High Confidence (≥85%): {stats.high_confidence_matches}")
        print(f"   Exact Matches (≥95%): {stats.exact_matches}")
        print(f"   Errors: {stats.errors}")
        print(f"   Average Confidence: {stats.avg_confidence:.1f}%")
        print(f"   Average Processing Time: {stats.avg_processing_time:.1f}ms")
        
        if stats.errors > 0:
            print(f"\n⚠️  Error Details:")
            for error in set(stats.error_details[:5]):  # Show unique errors, limit to 5
                print(f"   - {error}")
    
    # Export results if requested
    if args.output:
        export_results_to_csv(results, args.output)
    
    return 0

if __name__ == "__main__":
    exit(main())