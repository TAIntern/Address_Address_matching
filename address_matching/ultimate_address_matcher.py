#!/usr/bin/env python3
"""
Ultimate Address Matcher - Complete address processing and matching system

Features:
1. Advanced address parsing and normalization
2. Multiple address variant generation
3. 10-level matching system with quality scoring
4. Component-based scoring system
5. Reliability assessment
6. Support for multiple data source formats
7. Target 95%+ match rate
"""

import sys
import os
sys.path.append('.')

import re
import requests
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from optimized import MatchConfig, AddressMatcher, ESClient, MatchResult, IndexFieldMapping
from requests.auth import HTTPBasicAuth
import difflib

# Load usaddress library
try:
    import usaddress
    print("✅ usaddress loaded successfully")
except ImportError:
    print("❌ Failed to import usaddress. Install with: pip install usaddress")
    usaddress = None

@dataclass
class AddressComponents:
    """Standardized address components"""
    unit: str = ""
    house_number: str = ""
    street_name: str = ""
    street_type: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    country: str = "US"
    
    def to_string(self, format_type: str = "full") -> str:
        """Convert to string format"""
        if format_type == "full":
            parts = []
            if self.unit:
                parts.append(self.unit)
            if self.house_number:
                parts.append(self.house_number)
            if self.street_name:
                parts.append(self.street_name)
            if self.street_type:
                parts.append(self.street_type)
            if self.city:
                parts.append(self.city)
            if self.state:
                parts.append(self.state)
            if self.zip_code:
                parts.append(self.zip_code)
            return " ".join(parts)
        
        elif format_type == "street_only":
            parts = []
            if self.house_number:
                parts.append(self.house_number)
            if self.street_name:
                parts.append(self.street_name)
            if self.street_type:
                parts.append(self.street_type)
            return " ".join(parts)
        
        elif format_type == "likely_seller":
            parts = []
            if self.house_number:
                parts.append(self.house_number)
            if self.street_name:
                parts.append(self.street_name)
            if self.street_type:
                parts.append(self.street_type)
            if self.unit:
                parts.append(f"Apt {self.unit}")
            return " ".join(parts)
        
        return ""

@dataclass
class UltimateMatchResult:
    """Final match result with quality assessment"""
    original_result: MatchResult
    match_level: str
    confidence_adjusted: float
    quality_score: float
    reliability: str
    warning_flags: List[str]

    @property
    def is_reliable(self) -> bool:
        return self.reliability in ["high", "medium"] and self.quality_score >= 50

    @property
    def is_questionable(self) -> bool:
        return self.reliability in ["low"] and self.quality_score >= 25

    @property
    def is_speculative(self) -> bool:
        return self.reliability in ["very_low", "speculative"]

class UltimateAddressMatcher:
    """Ultimate address matcher with advanced processing and 10-level matching system"""

    def __init__(self, custom_field_mapping: Optional[IndexFieldMapping] = None):
        self.es_url = "https://search-addresses-f7voryraair5mrpw73ub2v3yg4.aos.us-east-1.on.aws"
        self.auth = HTTPBasicAuth("intern_3", "Trustscout1!")
        self.field_mapping_cache = {}
        self.custom_field_mapping = custom_field_mapping
        self.matchers = {}
        self.level_stats = {}
        self._initialized_for_index = None
        
        # Street type mapping
        self.street_types = {
            'street': 'st', 'avenue': 'ave', 'boulevard': 'blvd', 'drive': 'dr',
            'road': 'rd', 'lane': 'ln', 'court': 'ct', 'place': 'pl', 'way': 'way',
            'terrace': 'ter', 'circle': 'cir', 'parkway': 'pkwy', 'highway': 'hwy'
        }
        
        # State name mapping
        self.state_mapping = {
            'new york': 'ny', 'new jersey': 'nj', 'california': 'ca', 'texas': 'tx',
            'florida': 'fl', 'illinois': 'il', 'pennsylvania': 'pa', 'ohio': 'oh',
            'georgia': 'ga', 'north carolina': 'nc', 'michigan': 'mi', 'virginia': 'va'
        }
        
        # City name mapping
        self.city_mapping = {
            'brooklyn': 'brooklyn', 'manhattan': 'manhattan', 'queens': 'queens',
            'bronx': 'bronx', 'staten island': 'staten island', 'nyc': 'new york'
        }
        
        # Street name variants
        self.street_variants = {
            'broadway': ['broadway', 'broadway ave', 'broadway avenue'],
            'kent': ['kent', 'kent ave', 'kent avenue'],
            'berry': ['berry', 'berry st', 'berry street'],
            'jackson': ['jackson', 'jackson st', 'jackson street'],
            'north 3rd': ['north 3rd', 'n 3rd', 'north 3rd st', 'n 3rd st'],
            'north 11th': ['north 11th', 'n 11th', 'north 11th st', 'n 11th st'],
            'woodpoint': ['woodpoint', 'woodpoint rd', 'woodpoint road'],
            'metropolitan': ['metropolitan', 'metropolitan ave', 'metropolitan avenue'],
            'grand': ['grand', 'grand st', 'grand street'],
            'bedford': ['bedford', 'bedford ave', 'bedford avenue'],
            'marcy': ['marcy', 'marcy ave', 'marcy avenue'],
            'stagg': ['stagg', 'stagg st', 'stagg street'],
            'maspeth': ['maspeth', 'maspeth ave', 'maspeth avenue'],
            'union': ['union', 'union ave', 'union avenue'],
            'graham': ['graham', 'graham ave', 'graham avenue']
        }

    def _build_matcher(self, cfg: MatchConfig) -> AddressMatcher:
        return AddressMatcher(self.es_url, self.auth, cfg)

    def _detect_index_field_format(self, index: str) -> IndexFieldMapping:
        """Auto detect index field format (upper/lowercase)"""
        if index in self.field_mapping_cache:
            return self.field_mapping_cache[index]

        print(f"Detecting field format for index '{index}'...")

        try:
            query = {"query": {"match_all": {}}, "size": 1}
            response = requests.post(
                f"{self.es_url}/{index}/_search",
                auth=self.auth,
                headers={"Content-Type": "application/json"},
                data=json.dumps(query),
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                hits = result.get("hits", {}).get("hits", [])

                if hits:
                    sample_doc = hits[0]["_source"]
                    fields = set(sample_doc.keys())

                    has_uppercase = any(field in fields for field in ["ADDRESS", "STREET", "CITY", "STATE"])
                    has_lowercase = any(field in fields for field in ["address", "street", "city", "state"])

                    if has_uppercase and not has_lowercase:
                        print("Detected uppercase field format")
                        mapping = IndexFieldMapping()
                    elif has_lowercase and not has_uppercase:
                        print("Detected lowercase field format")
                        mapping = IndexFieldMapping(
                            house_field="house",
                            street_field="street",
                            city_field="city",
                            state_field="state",
                            zip_field="zip_code"
                        )
                    else:
                        print("Detected mixed field format, using default mapping")
                        mapping = IndexFieldMapping()

                    self.field_mapping_cache[index] = mapping
                    return mapping

        except Exception as e:
            print(f"Field format detection failed: {e}")

        print("Using default field mapping")
        return IndexFieldMapping()

    def _initialize_matchers_for_index(self, index: str):
        """Initialize all matchers for a specific index"""
        if self._initialized_for_index == index:
            return

        print(f"Initializing matchers for index '{index}'...")
        
        field_mapping = self._detect_index_field_format(index)
        
        # Reset matchers and stats
        self.matchers = {}
        self.level_stats = {
            "exact": 0, "relaxed": 0, "partial": 0, "geographic": 0,
            "fuzzy": 0, "ultra_fuzzy": 0, "semantic": 0, "phonetic": 0,
            "keyword": 0, "desperate": 0, "failed": 0
        }

        # Create all matchers
        self._create_exact_matcher(field_mapping)
        self._create_relaxed_matcher(field_mapping)
        self._create_partial_matcher(field_mapping)
        self._create_geographic_matcher(field_mapping)
        self._create_fuzzy_matcher(field_mapping)
        self._create_ultra_fuzzy_matcher(field_mapping)
        self._create_semantic_matcher(field_mapping)
        self._create_phonetic_matcher(field_mapping)
        self._create_keyword_matcher(field_mapping)
        self._create_desperate_matcher(field_mapping)

        self._initialized_for_index = index
        print(f"Matcher initialization complete ({len(self.matchers)} levels)")

    def _create_exact_matcher(self, field_mapping: IndexFieldMapping):
        """Create exact match matcher"""
        cfg = MatchConfig(
            min_similarity_threshold=95,
            max_results=1,
            field_mapping=field_mapping
        )
        self.matchers["exact"] = self._build_matcher(cfg)

    def _create_relaxed_matcher(self, field_mapping: IndexFieldMapping):
        """Create relaxed match matcher"""
        cfg = MatchConfig(
            min_similarity_threshold=85,
            max_results=3,
            field_mapping=field_mapping
        )
        self.matchers["relaxed"] = self._build_matcher(cfg)

    def _create_partial_matcher(self, field_mapping: IndexFieldMapping):
        """Create partial match matcher"""
        cfg = MatchConfig(
            min_similarity_threshold=75,
            max_results=5,
            field_mapping=field_mapping
        )
        self.matchers["partial"] = self._build_matcher(cfg)

    def _create_geographic_matcher(self, field_mapping: IndexFieldMapping):
        """Create geographic match matcher"""
        cfg = MatchConfig(
            min_similarity_threshold=65,
            max_results=10,
            field_mapping=field_mapping
        )
        self.matchers["geographic"] = self._build_matcher(cfg)

    def _create_fuzzy_matcher(self, field_mapping: IndexFieldMapping):
        """Create fuzzy match matcher"""
        cfg = MatchConfig(
            min_similarity_threshold=55,
            max_results=15,
            field_mapping=field_mapping
        )
        self.matchers["fuzzy"] = self._build_matcher(cfg)

    def _create_ultra_fuzzy_matcher(self, field_mapping: IndexFieldMapping):
        """Create ultra fuzzy match matcher"""
        cfg = MatchConfig(
            min_similarity_threshold=45,
            max_results=20,
            field_mapping=field_mapping
        )
        self.matchers["ultra_fuzzy"] = self._build_matcher(cfg)

    def _create_semantic_matcher(self, field_mapping: IndexFieldMapping):
        """Create semantic match matcher"""
        cfg = MatchConfig(
            min_similarity_threshold=35,
            max_results=25,
            field_mapping=field_mapping
        )
        self.matchers["semantic"] = self._build_matcher(cfg)

    def _create_phonetic_matcher(self, field_mapping: IndexFieldMapping):
        """Create phonetic match matcher"""
        cfg = MatchConfig(
            min_similarity_threshold=25,
            max_results=30,
            field_mapping=field_mapping
        )
        self.matchers["phonetic"] = self._build_matcher(cfg)

    def _create_keyword_matcher(self, field_mapping: IndexFieldMapping):
        """Create keyword match matcher"""
        cfg = MatchConfig(
            min_similarity_threshold=15,
            max_results=35,
            field_mapping=field_mapping
        )
        self.matchers["keyword"] = self._build_matcher(cfg)

    def _create_desperate_matcher(self, field_mapping: IndexFieldMapping):
        """Create desperate match matcher"""
        cfg = MatchConfig(
            min_similarity_threshold=5,
            max_results=50,
            field_mapping=field_mapping
        )
        self.matchers["desperate"] = self._build_matcher(cfg)

    def parse_address(self, address: str) -> AddressComponents:
        """Parse address into standardized components using usaddress first"""
        if not address or not address.strip():
            return AddressComponents()
        
        cleaned = address.strip()
        
                # First try usaddress for standard parsing (if available)
        if usaddress:
            try:
                parsed, addr_type = usaddress.tag(cleaned)
                
                # Extract components from usaddress result
                unit = parsed.get('OccupancyIdentifier', '') or parsed.get('SubaddressIdentifier', '')
                house_number = parsed.get('AddressNumber', '')
                street_pre_dir = parsed.get('StreetNamePreDirectional', '')
                street_name = parsed.get('StreetName', '')
                street_post_type = parsed.get('StreetNamePostType', '')
                city = parsed.get('PlaceName', '')
                state = parsed.get('StateName', '')
                zip_code = parsed.get('ZipCode', '')
                
                # Fix common usaddress parsing errors for unit+house combinations
                if not unit and house_number and street_name:
                    # Pattern 1: AddressNumber is unit (e.g., "1H") and StreetName starts with digits (e.g., "55 Berry")
                    if re.match(r'^[A-Z0-9]{1,3}$', house_number) and re.match(r'^\d+\s+', street_name):
                        # Extract the real house number from street name
                        match = re.match(r'^(\d+)\s+(.+)', street_name)
                        if match:
                            real_house = match.group(1)
                            real_street = match.group(2)
                            
                            # Only reassign if the pattern looks like unit+house+street
                            if len(house_number) <= 3 and house_number.isalnum():
                                unit = house_number
                                house_number = real_house  
                                street_name = real_street
                    
                    # Pattern 2: AddressNumber contains unit+house (e.g., "3D 130") and StreetName is just street
                    elif re.match(r'^[A-Z0-9]{1,3}\s+\d+', house_number):
                        # Extract unit and house from AddressNumber
                        match = re.match(r'^([A-Z0-9]{1,3})\s+(\d+)', house_number)
                        if match:
                            unit = match.group(1)
                            house_number = match.group(2)
                
                # Combine street components
                street_parts = [street_pre_dir, street_name]
                street_name_combined = ' '.join(filter(None, street_parts))
                
                # If usaddress didn't find a unit, try custom extraction ONLY for specific patterns
                if not unit:
                    # Only use custom extraction if address starts with a pattern like "1A", "2B", etc.
                    if re.match(r'^[A-Z0-9]{1,3}\s+\d', cleaned):
                        unit = self._extract_unit(cleaned)
                
                # If usaddress didn't find house number, try custom extraction
                if not house_number:
                    house_number = self._extract_house_number(cleaned)
                
                return AddressComponents(
                    unit=unit,
                    house_number=house_number,
                    street_name=street_name_combined,
                    street_type=street_post_type.lower() if street_post_type else '',
                    city=city,
                    state=state,
                    zip_code=zip_code
                )
                
            except Exception as e:
                # Fall back to original parsing if usaddress fails
                print(f"Warning: usaddress failed for '{cleaned}': {e}")
        
        # Fall back to original parsing if usaddress is not available or failed
        # Parse unit number
        unit = self._extract_unit(cleaned)
        if unit:
            cleaned = cleaned.replace(unit, '', 1).strip()
        
        # Parse house number
        house_number = self._extract_house_number(cleaned)
        if house_number:
            cleaned = cleaned.replace(house_number, '', 1).strip()
        
        # Separate street and location information
        street_part, location_part = self._separate_street_and_location(cleaned)
        
        # Parse street name and type
        street_name, street_type = self._parse_street(street_part)
        
        # Parse location information
        city, state, zip_code = self._parse_location(location_part)
        
        return AddressComponents(
            unit=unit,
            house_number=house_number,
            street_name=street_name,
            street_type=street_type,
            city=city,
            state=state,
            zip_code=zip_code
        )
    
    def _extract_unit(self, address: str) -> str:
        """Extract unit number with enhanced patterns"""
        # State abbreviations to exclude from unit matching
        state_abbrevs = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        }
        
        # Enhanced unit number patterns
        unit_patterns = [
            r'^([A-Z0-9]{1,3})\s+',  # 1A, 2B, 3C, 4R at start
            r'\b(apt|suite|unit|#)\s*([A-Z0-9]{1,3})\b',  # Apt 1A, Suite 2B
            r'\b([A-Z0-9]{1,3})\b(?=\s+\d)',  # 1A 123 Main St
            r'\b([A-Z0-9]{1,3})\s+(?=\d)',  # 1A 123 Main St (space before number)
            r'^([A-Z0-9]{1,3})[-\s]',  # 1A-, 2B- at start
            r'\b([A-Z0-9]{1,3})\b(?=\s+[A-Z])',  # 1A Main St
        ]
        
        for pattern in unit_patterns:
            match = re.search(pattern, address, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    unit = match.group(2).upper()
                else:
                    unit = match.group(1).upper()
                
                # Validate unit format (should be alphanumeric, 1-3 chars)
                # but exclude state abbreviations
                if re.match(r'^[A-Z0-9]{1,3}$', unit) and unit not in state_abbrevs:
                    return unit
        
        return ""
    
    def _extract_house_number(self, address: str) -> str:
        """Extract house number"""
        # Match house number patterns
        house_patterns = [
            r'^(\d+[A-Za-z]*)',  # 123, 123A
            r'^(\d+-\d+)',  # 123-125
            r'\b(\d+[A-Za-z]*)\s+[A-Za-z]',  # 123 Main
        ]
        
        for pattern in house_patterns:
            match = re.match(pattern, address)
            if match:
                return match.group(1)
        
        return ""
    
    def _separate_street_and_location(self, address: str) -> Tuple[str, str]:
        """Separate street and location information"""
        # Find separators
        separators = [',', ' - ', ' – ', ' — ']
        
        for sep in separators:
            if sep in address:
                parts = address.split(sep, 1)
                return parts[0].strip(), parts[1].strip()
        
        # If no separator, try intelligent separation
        words = address.split()
        
        # Find state name or zip code as separation point
        for i, word in enumerate(words):
            if (len(word) == 2 and word.upper() in ['NY', 'NJ', 'CA', 'TX', 'FL']) or \
               (len(word) == 5 and word.isdigit()):
                street_part = ' '.join(words[:i])
                location_part = ' '.join(words[i:])
                return street_part, location_part
        
        # Default return entire address as street part
        return address, ""
    
    def _parse_street(self, street_part: str) -> Tuple[str, str]:
        """Parse street name and type"""
        if not street_part:
            return "", ""
        
        words = street_part.split()
        street_name = ""
        street_type = ""
        
        # Search for street type from back to front
        for i in range(len(words) - 1, -1, -1):
            word = words[i].lower()
            if word in self.street_types:
                street_type = self.street_types[word]
                street_name = ' '.join(words[:i])
                break
            elif word in ['st', 'ave', 'rd', 'dr', 'blvd', 'ln', 'ct', 'pl']:
                street_type = word
                street_name = ' '.join(words[:i])
                break
        
        # If no street type found, try to infer
        if not street_type and words:
            last_word = words[-1].lower()
            if last_word in self.street_types:
                street_type = self.street_types[last_word]
                street_name = ' '.join(words[:-1])
            else:
                street_name = street_part
        
        return street_name.strip(), street_type
    
    def _parse_location(self, location_part: str) -> Tuple[str, str, str]:
        """Parse location information (city, state, zip code)"""
        if not location_part:
            return "Brooklyn", "NY", ""
        
        words = location_part.split()
        city = "Brooklyn"
        state = "NY"
        zip_code = ""
        
        # Find zip code
        for word in words:
            if len(word) == 5 and word.isdigit():
                zip_code = word
                break
        
        # Find state name
        for word in words:
            if len(word) == 2 and word.upper() in ['NY', 'NJ', 'CA', 'TX', 'FL']:
                state = word.upper()
                break
        
        # Find city name
        for word in words:
            word_lower = word.lower()
            if word_lower in self.city_mapping:
                city = self.city_mapping[word_lower]
                break
        
        return city, state, zip_code

    def _create_street_with_location(self, components: AddressComponents) -> str:
        """Create street variant with geographic location for accuracy"""
        parts = []
        if components.house_number:
            parts.append(components.house_number)
        if components.street_name:
            parts.append(components.street_name)
        if components.street_type:
            parts.append(components.street_type)
        
        # Always include location for geographic accuracy
        if components.city:
            parts.append(components.city)
        if components.state:
            parts.append(components.state)
        if components.zip_code:
            parts.append(components.zip_code)
            
        return " ".join(parts)

    def _create_likely_seller_with_location(self, components: AddressComponents) -> str:
        """Create likely seller variant with geographic location for accuracy"""
        parts = []
        if components.house_number:
            parts.append(components.house_number)
        if components.street_name:
            parts.append(components.street_name)
        if components.street_type:
            parts.append(components.street_type)
        if components.unit:
            parts.append(f"Apt {components.unit}")
            
        # Always include location for geographic accuracy
        if components.city:
            parts.append(components.city)
        if components.state:
            parts.append(components.state)
        if components.zip_code:
            parts.append(components.zip_code)
            
        return " ".join(parts)

    def generate_variants(self, address: str, max_variants: int = 15) -> List[str]:
        """Generate address variants with enhanced unit handling"""
        components = self.parse_address(address)
        variants = []
        
        # Base variants - ensure all variants include geographic location
        base_variants = [
            components.to_string("full"),  # Full address with all components
            # Modified to include geographic location for accuracy
            self._create_street_with_location(components),  # Street + City, State
            self._create_likely_seller_with_location(components)  # Street + Unit + City, State
        ]
        variants.extend([v for v in base_variants if v])
        
        # Enhanced unit variants - CRITICAL for unit matching
        if components.unit:
            # Create variants with unit in different formats
            unit_variants = [
                f"{components.house_number} {components.street_name} {components.street_type} Apt {components.unit}",
                f"{components.house_number} {components.street_name} {components.street_type} Unit {components.unit}",
                f"{components.house_number} {components.street_name} {components.street_type} #{components.unit}",
                f"{components.house_number} {components.street_name} {components.street_type} {components.unit}",
                f"{components.house_number} {components.street_name} Apt {components.unit}",
                f"{components.house_number} {components.street_name} Unit {components.unit}",
                f"{components.house_number} {components.street_name} #{components.unit}",
                f"{components.house_number} {components.street_name} {components.unit}",
            ]
            variants.extend([v for v in unit_variants if v])
        
        # Street name variants
        if components.street_name:
            street_lower = components.street_name.lower()
            for base_name, variant_list in self.street_variants.items():
                if base_name in street_lower:
                    for variant in variant_list[:3]:  # Take first 3 variants
                        new_street = street_lower.replace(base_name, variant)
                        if components.house_number:
                            new_variant = f"{components.house_number} {new_street}"
                            if components.unit:
                                new_variant += f" Apt {components.unit}"
                            # Add location for geographic accuracy
                            if components.city:
                                new_variant += f" {components.city}"
                            if components.state:
                                new_variant += f" {components.state}"
                        else:
                            new_variant = new_street
                            # Add location for geographic accuracy
                            if components.city:
                                new_variant += f" {components.city}"
                            if components.state:
                                new_variant += f" {components.state}"
                        variants.append(new_variant)
                    break
        
        # Simplified variants - include location for geographic accuracy
        if components.house_number and components.street_name:
            simplified = f"{components.house_number} {components.street_name}"
            if components.city:
                simplified += f" {components.city}"
            if components.state:
                simplified += f" {components.state}"
            variants.append(simplified)
        
        # Remove unit number variants (for fallback matching)
        if components.unit:
            no_unit_components = AddressComponents(
                house_number=components.house_number,
                street_name=components.street_name,
                street_type=components.street_type,
                city=components.city,
                state=components.state,
                zip_code=components.zip_code
            )
            variants.append(no_unit_components.to_string("full"))
        
        # Remove duplicates and limit quantity
        unique_variants = []
        seen = set()
        
        for variant in variants:
            if variant and variant.strip() and variant not in seen:
                unique_variants.append(variant.strip())
                seen.add(variant)
                if len(unique_variants) >= max_variants:
                    break
        
        return unique_variants

    def _extract_keywords(self, address: str) -> List[str]:
        """Extract meaningful keywords from address"""
        # Remove common words and punctuation
        stop_words = {"the", "and", "or", "of", "in", "at", "to", "for", "with", "by"}
        
        # Clean address
        cleaned = re.sub(r'[^\w\s]', ' ', address.lower())
        words = cleaned.split()
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add street type variations
        street_types = ["street", "st", "avenue", "ave", "boulevard", "blvd", "road", "rd", "lane", "ln"]
        for word in words:
            if word in street_types:
                keywords.append(word)
        
        return keywords

    def _calculate_enhanced_quality_score(self, result: MatchResult, match_level: str,
                                        preprocessing_issues: Dict[str, str]) -> float:
        """Calculate enhanced quality score with improved unit matching"""
        base_score = 0
        
        if result.matched:
            # Base score from confidence
            base_score = result.confidence
            
            # Component scoring
            matched_data = result.matched
            component_scores = result.component_scores or {}
            
            # Street score
            if 'street' in matched_data and matched_data['street']:
                base_score += 20
            
            # House number score
            if 'house' in matched_data and matched_data['house']:
                base_score += 15
            
            # Enhanced Unit score - CRITICAL for unit matching
            unit_score = 0
            if 'unit' in matched_data and matched_data['unit']:
                unit_score += 30  # High weight for unit matching
            elif 'unit' in component_scores and component_scores['unit'] > 0:
                unit_score += 25  # Component score for unit
            
            # Check for APTNBR field specifically
            if 'APTNBR' in matched_data and matched_data['APTNBR']:
                unit_score += 35  # Even higher weight for APTNBR
            elif 'aptnbr' in matched_data and matched_data['aptnbr']:
                unit_score += 35  # Case insensitive
            
            # Check for APTTYPE field
            if 'APTTYPE' in matched_data and matched_data['APTTYPE']:
                unit_score += 5  # Bonus for having apartment type
            
            base_score += unit_score
            
            # City score
            if 'city' in matched_data and matched_data['city']:
                base_score += 10
            
            # State score
            if 'state' in matched_data and matched_data['state']:
                base_score += 5
            
            # Zip score
            if 'zip_code' in matched_data and matched_data['zip_code']:
                base_score += 5
            
            # ES score bonus
            if result.es_score > 0:
                base_score += min(result.es_score / 10, 20)
            
            # Similarity bonus
            if result.similarity > 0:
                base_score += min(result.similarity / 5, 15)
        
        # Level adjustments
        level_adjustments = {
            "exact": 0, "relaxed": -5, "partial": -10, "geographic": -15,
            "fuzzy": -25, "ultra_fuzzy": -35, "semantic": -45,
            "phonetic": -55, "keyword": -65, "desperate": -75
        }
        
        base_score += level_adjustments.get(match_level, -80)
        
        # Issue penalties
        for issue in preprocessing_issues.values():
            if "empty" in issue.lower():
                base_score -= 50
            elif "too_short" in issue.lower():
                base_score -= 30
            elif "numeric_only" in issue.lower():
                base_score -= 20
        
        return max(0, min(100, base_score))

    def _assess_reliability(self, result: MatchResult, match_level: str,
                          quality_score: float) -> Tuple[str, List[str]]:
        """Assess match reliability"""
        warnings = []
        
        if quality_score >= 80:
            reliability = "high"
        elif quality_score >= 60:
            reliability = "medium"
        elif quality_score >= 40:
            reliability = "low"
        elif quality_score >= 20:
            reliability = "very_low"
        else:
            reliability = "speculative"
        
        # Add warnings based on match level
        if match_level in ["phonetic", "keyword", "desperate"]:
            warnings.append("Low confidence match level")
        
        if result.similarity < 50:
            warnings.append("Low similarity score")
        
        if result.es_score < 5:
            warnings.append("Low Elasticsearch score")
        
        return reliability, warnings

    def match_address(self, address: str, index: str = "tci_base_staging_data") -> UltimateMatchResult:
        """Enhanced multilevel adaptive address matching pipeline with unit prioritization"""
        self._initialize_matchers_for_index(index)

        # Parse address to get unit information
        components = self.parse_address(address)
        has_unit = bool(components.unit)
        
        # CRITICAL FIX: If we have a unit, try direct unit matching first
        if has_unit and components.house_number and components.street_name:
            unit_result = self._try_direct_unit_match(components, index)
            if unit_result:
                return unit_result
        
        # Generate address variants
        variants = self.generate_variants(address)
        
        if not variants:
            empty_result = MatchResult(address, None, 0, 0, 0, {}, 0, "Empty address", index)
            return UltimateMatchResult(empty_result, "failed", 0.0, 0.0, "speculative", ["Empty address"])

        # Try all matchers with enhanced variants
        for level_name, matcher in self.matchers.items():
            try:
                # Try each variant for this level
                for variant in variants[:5]:  # Limit to first 5 variants per level
                    if level_name in ["semantic", "phonetic", "keyword"]:
                        keywords = self._extract_keywords(variant)
                        if keywords:
                            for i in range(min(3, len(keywords))):
                                test_address = " ".join(keywords[:i+1])
                                result = matcher.match(test_address, index)
                                if result.matched and result.confidence > 0:
                                    break
                            else:
                                result = matcher.match(variant, index)
                        else:
                            result = matcher.match(variant, index)
                    else:
                        result = matcher.match(variant, index)

                    if result.matched and result.confidence >= 0:
                        # Check unit matching if original address has unit
                        unit_match_bonus = 0
                        if has_unit and components.unit:
                            matched_unit = result.matched.get('APTNBR', '')
                            if matched_unit and components.unit.upper() == matched_unit.upper():
                                unit_match_bonus = 50  # Big bonus for exact unit match
                            elif matched_unit:  # Has unit but doesn't match
                                unit_match_bonus = -20  # Penalty for wrong unit
                        
                        self.level_stats[level_name] += 1
                        quality = self._calculate_enhanced_quality_score(result, level_name, {})
                        quality += unit_match_bonus  # Apply unit matching bonus/penalty

                        level_adjustments = {
                            "exact": 0, "relaxed": -5, "partial": -10, "geographic": -15,
                            "fuzzy": -25, "ultra_fuzzy": -35, "semantic": -45,
                            "phonetic": -55, "keyword": -65, "desperate": -75
                        }

                        adjusted_conf = max(0, result.confidence + level_adjustments.get(level_name, -80))
                        reliability, warnings = self._assess_reliability(result, level_name, quality)

                        # If we have a unit match, prefer this result even if it's not the first level
                        if unit_match_bonus > 0:
                            return UltimateMatchResult(result, level_name, adjusted_conf, quality, reliability, warnings)
                        
                        # For non-unit matches, continue searching if we expect a unit match
                        if has_unit and unit_match_bonus <= 0 and level_name in ["exact", "relaxed"]:
                            continue  # Keep searching for better unit match
                        
                        return UltimateMatchResult(result, level_name, adjusted_conf, quality, reliability, warnings)

            except Exception:
                continue

        self.level_stats["failed"] += 1
        failed_result = MatchResult(address, None, 0, 0, 0, {}, 0, "No match at any level", index)
        return UltimateMatchResult(failed_result, "failed", 0.0, 0.0, "speculative", ["Unmatched address"])
    
    def _generate_unit_variants(self, unit: str) -> List[str]:
        """生成单元号的多种变体"""
        variants = [unit.upper().strip()]  # 原始形式
        
        # 去除前缀和后缀
        clean_unit = unit.upper().strip()
        
        # 移除常见前缀
        for prefix in ['#', 'APT', 'UNIT', 'SUITE', 'STE', 'NO']:
            if clean_unit.startswith(prefix):
                clean_unit = clean_unit[len(prefix):].strip()
        
        variants.append(clean_unit)
        
        # 数字+字母组合的变体
        if re.match(r'^\d+[A-Z]$', clean_unit):  # 如 "19D"
            # 尝试只用数字部分
            number_part = re.match(r'^(\d+)', clean_unit).group(1)
            variants.append(number_part)
            
            # 尝试添加常见后缀
            variants.append(f"{number_part}A")  # 如 19A
            variants.append(f"{number_part}B")  # 如 19B
            variants.append(f"{number_part}C")  # 如 19C
            
        # 字母+数字组合的变体
        elif re.match(r'^[A-Z]\d+$', clean_unit):  # 如 "A19"
            # 转换为数字+字母格式
            letter = clean_unit[0]
            number = clean_unit[1:]
            variants.append(f"{number}{letter}")
            
        # 纯数字的变体
        elif clean_unit.isdigit():
            # 添加常见字母后缀
            for suffix in ['A', 'B', 'C', 'D']:
                variants.append(f"{clean_unit}{suffix}")
        
        # 去重并返回
        return list(dict.fromkeys(variants))

    def _try_direct_unit_match(self, components: AddressComponents, index: str) -> Optional[UltimateMatchResult]:
        """Enhanced unit matching with exact/approximate distinction"""
        if not components.unit or not components.house_number or not components.street_name:
            return None
        
        import requests
        import json
        
        print(f"🔍 智能单元匹配: {components.unit} {components.house_number} {components.street_name}")
        
        # 步骤1: 尝试精确单元匹配
        print(f"🔍 步骤1: 尝试精确单元匹配 '{components.unit}'")
        exact_matches = self._search_unit_with_variant(components, components.unit, index, "exact_priority")
        
        if exact_matches:
            print(f"✅ 找到精确单元匹配: {len(exact_matches)} 条")
            best_match = self._select_best_unit_match(exact_matches, components)
            if best_match:
                return self._create_enhanced_unit_result(best_match, components, index, is_exact=True)
        
        print(f"❌ 精确单元匹配失败，尝试变体匹配...")
        
        # 步骤2: 尝试单元变体匹配
        all_matches = []
        unit_variants = self._generate_unit_variants(components.unit)[1:]  # 排除原始
        
        for variant in unit_variants[:5]:  # 限制变体数量
            print(f"🔍 尝试单元变体: '{variant}'")
            matches = self._search_unit_with_variant(components, variant, index, "variant")
            all_matches.extend(matches)
        
        # 步骤3: 广泛搜索（最后手段）
        if not all_matches:
            print(f"🔍 尝试广泛搜索...")
            all_matches = self._search_unit_broadly(components, index)
        
        if all_matches:
            best_match = self._select_best_unit_match(all_matches, components)
            if best_match:
                matched_unit = best_match['source'].get('APTNBR', '').upper()
                original_unit = components.unit.upper()
                
                if matched_unit == original_unit:
                    print(f"✅ 找到精确单元匹配: {components.unit}")
                    return self._create_enhanced_unit_result(best_match, components, index, is_exact=True)
                else:
                    print(f"⚠️  找到近似单元匹配: {components.unit} → {matched_unit}")
                    print(f"   📝 数据库中不存在精确单元'{components.unit}'，返回最接近匹配")
                    return self._create_enhanced_unit_result(best_match, components, index, is_exact=False)
        
        print(f"❌ 未找到任何单元匹配: {components.unit}")
        return None

    def _search_unit_with_variant(self, components: AddressComponents, unit_variant: str, 
                                 index: str, strategy: str) -> List[Dict[str, Any]]:
        """使用单元变体进行搜索 - 加入地理位置约束"""
        import requests
        import json
        
        try:
            query = {
                "size": 100,  # 增大搜索范围
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"HOUSE": components.house_number}},
                            {"match": {"STREET": components.street_name}},
                            {"term": {"APTNBR": unit_variant}}
                        ]
                    }
                },
                "_source": ["HOUSE", "STREET", "APTNBR", "ADDRESS", "CITY", "STATE", "ZIP"],
                "sort": [{"_score": {"order": "desc"}}]
            }
            
            # 添加地理位置约束
            if components.city:
                query["query"]["bool"]["must"].append({"match": {"CITY": components.city}})
            if components.state:
                query["query"]["bool"]["must"].append({"term": {"STATE": components.state.upper()}})
            
            response = requests.post(
                f"{self.es_url}/{index}/_search",
                auth=self.auth,
                headers={"Content-Type": "application/json"},
                data=json.dumps(query),
                timeout=10
            )
            
            if response.status_code == 200:
                result_data = response.json()
                hits = result_data.get("hits", {}).get("hits", [])
                return [{"hit": hit, "variant_used": unit_variant, "strategy": strategy} for hit in hits]
                
        except Exception as e:
            print(f"搜索错误 (变体: {unit_variant}): {e}")
            
        return []

    def _search_unit_broadly(self, components: AddressComponents, index: str) -> List[Dict[str, Any]]:
        """广泛搜索单元 - 加入地理位置约束"""
        import requests
        import json
        
        try:
            query = {
                "size": 200,
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"STREET": components.street_name}},
                            {"term": {"HOUSE": components.house_number}}
                        ],
                        "minimum_should_match": 1,
                        "must": []  # 添加must数组用于地理位置约束
                    }
                },
                "_source": ["HOUSE", "STREET", "APTNBR", "ADDRESS", "CITY", "STATE", "ZIP"],
                "sort": [{"_score": {"order": "desc"}}]
            }
            
            # 添加地理位置约束
            if components.city:
                query["query"]["bool"]["must"].append({"match": {"CITY": components.city}})
            if components.state:
                query["query"]["bool"]["must"].append({"term": {"STATE": components.state.upper()}})
            
            response = requests.post(
                f"{self.es_url}/{index}/_search",
                auth=self.auth,
                headers={"Content-Type": "application/json"},
                data=json.dumps(query),
                timeout=15
            )
            
            if response.status_code == 200:
                result_data = response.json()
                hits = result_data.get("hits", {}).get("hits", [])
                
                # 客户端过滤
                filtered_hits = []
                unit_variants = self._generate_unit_variants(components.unit)
                
                for hit in hits:
                    source = hit["_source"]
                    hit_unit = source.get('APTNBR', '').upper()
                    
                    # 检查是否匹配任何变体
                    for variant in unit_variants:
                        if hit_unit == variant.upper():
                            filtered_hits.append({"hit": hit, "variant_used": variant, "strategy": "broad_scope"})
                            break
                
                return filtered_hits
                
        except Exception as e:
            print(f"广泛搜索错误: {e}")
            
        return []

    def _select_best_unit_match(self, matches: List[Dict[str, Any]], 
                               components: AddressComponents) -> Optional[Dict[str, Any]]:
        """选择最佳单元匹配"""
        if not matches:
            return None
        
        # 评分系统
        scored_matches = []
        
        for match in matches:
            hit = match["hit"]
            source = hit["_source"]
            score = hit.get("_score", 0)
            
            # 基础评分
            total_score = score
            
            # 策略加分 - 精确匹配获得巨大优势
            if match["strategy"] == "exact_priority":
                total_score += 1000  # 精确匹配巨大加分
            elif match["strategy"] == "exact":
                total_score += 100   # 其他精确策略
            elif match["strategy"] == "variant":
                total_score += 20    # 变体匹配较少加分
            elif match["strategy"] == "fuzzy_address":
                total_score += 10
            elif match["strategy"] == "broad_scope":
                total_score += 5
            
            # 地址匹配度加分
            if source.get("HOUSE", "").upper() == components.house_number.upper():
                total_score += 30
            
            if components.street_name.upper() in source.get("STREET", "").upper():
                total_score += 20
            
            # 单元匹配度加分 - 完全匹配优先级极高
            hit_unit = source.get('APTNBR', '').upper()
            original_unit = components.unit.upper()
            
            if hit_unit == original_unit:
                total_score += 500  # 完全匹配获得巨大加分
            elif hit_unit in [v.upper() for v in self._generate_unit_variants(components.unit)]:
                total_score += 30   # 变体匹配较少加分
            
            scored_matches.append({
                "match": match,
                "score": total_score,
                "hit": hit,
                "source": source
            })
        
        # 返回最高分的匹配 - 优先完全匹配
        if scored_matches:
            # 首先检查是否有完全匹配
            exact_matches = [m for m in scored_matches if m['source'].get('APTNBR', '').upper() == components.unit.upper()]
            
            if exact_matches:
                # 如果有完全匹配，选择其中评分最高的
                best = max(exact_matches, key=lambda x: x["score"])
                print(f"🎯 选择完全匹配: {best['source'].get('APTNBR', '')} (评分: {best['score']:.1f})")
                return best
            else:
                # 否则选择总体评分最高的
                best = max(scored_matches, key=lambda x: x["score"])
                print(f"📝 选择最佳变体匹配: {best['source'].get('APTNBR', '')} (评分: {best['score']:.1f})")
                return best
        
        return None

    def _create_enhanced_unit_result(self, best_match: Dict[str, Any], 
                                   components: AddressComponents, index: str, is_exact: bool = True) -> UltimateMatchResult:
        """创建增强的单元匹配结果"""
        hit = best_match["hit"]
        source = best_match["source"]
        score = best_match["score"]
        matched_unit = source.get('APTNBR', '')
        
        # 根据是否精确匹配调整参数
        if is_exact:
            # 精确匹配
            match_level = "exact_unit"
            confidence = min(95.0, score)
            quality_score = 120.0 + (score * 0.3)
            reliability = "high"
            warning_flags = []
            unit_component_score = 100.0
        else:
            # 近似匹配
            match_level = "approximate_unit"
            confidence = min(75.0, score * 0.8)  # 降低置信度
            quality_score = 80.0 + (score * 0.2)  # 降低质量分数
            reliability = "medium"
            warning_flags = [f"Unit mismatch: requested '{components.unit}' but found '{matched_unit}'"]
            unit_component_score = 60.0  # 降低单元组件分数
        
        # 创建MatchResult
        from optimized import MatchResult
        match_result = MatchResult(
            query=f"{components.unit} {components.house_number} {components.street_name}",
            matched=source,
            confidence=confidence,
            similarity=min(95.0, score * 0.9),
            es_score=hit.get("_score", 10.0),
            component_scores={
                "unit": unit_component_score,
                "house": 95.0 if source.get("HOUSE") == components.house_number else 80.0,
                "street": 90.0
            },
            ms=1,
            index=index
        )
        
        return UltimateMatchResult(
            original_result=match_result,
            match_level=match_level,
            confidence_adjusted=confidence,
            quality_score=min(200.0, quality_score),
            reliability=reliability,
            warning_flags=warning_flags
        )

    def get_statistics(self) -> Dict[str, any]:
        """Return distribution statistics after matching"""
        total = sum(self.level_stats.values())
        if total == 0:
            return {}

        stats = {}
        for level, count in self.level_stats.items():
            stats[f"{level}_count"] = count
            stats[f"{level}_percentage"] = count / total * 100

        stats["total_processed"] = total
        stats["total_matched"] = total - self.level_stats["failed"]
        stats["match_rate"] = (total - self.level_stats["failed"]) / total * 100 if total > 0 else 0

        return stats

def test_ultimate_address_matcher():
    """Test the Ultimate Address Matcher"""
    print("Testing Ultimate Address Matcher")
    print("=" * 60)
    
    matcher = UltimateAddressMatcher()
    
    # Test addresses
    test_addresses = [
        "85-101 North 3rd Street Brooklyn, NY 11249",
        "3R 112 Bedford Avenue Brooklyn, NY 11211",
        "1H 55 Berry Street Brooklyn, NY 11211",
        "21F 22 NORTH 6 STREET Brooklyn, NY 11249",
        "3Q 60 Broadway Brooklyn, NY 11249"
    ]
    
    print("Address Parsing Test:")
    for i, address in enumerate(test_addresses, 1):
        print(f"\n{i}. Original Address: {address}")
        
        # Parse address
        components = matcher.parse_address(address)
        print(f"   Unit: {components.unit}")
        print(f"   House Number: {components.house_number}")
        print(f"   Street Name: {components.street_name}")
        print(f"   Street Type: {components.street_type}")
        print(f"   City: {components.city}")
        print(f"   State: {components.state}")
        print(f"   Zip Code: {components.zip_code}")
        
        # Generate variants
        variants = matcher.generate_variants(address)
        print(f"   Address Variants ({len(variants)}):")
        for j, variant in enumerate(variants[:5], 1):
            print(f"    {j}. {variant}")
    
    print(f"\nMatching Test:")
    for i, address in enumerate(test_addresses[:3], 1):
        print(f"\n{i}. Matching Address: {address}")
        
        result = matcher.match_address(address, "likely_seller")
        
        print(f"   Success: {'Yes' if result.original_result.matched else 'No'}")
        print(f"   Match Level: {result.match_level}")
        print(f"   Quality Score: {result.quality_score:.1f}")
        print(f"   Reliability: {result.reliability}")
        print(f"   Confidence: {result.confidence_adjusted:.1f}")
        
        if result.original_result.matched:
            matched = result.original_result.matched
            print(f"   Matched Address: {matched.get('address', '')}")
            print(f"   PID: {matched.get('pid', '')}")
            print(f"   HHID: {matched.get('hhid', '')}")
            print(f"   ADDRID: {matched.get('addrid', '')}")
    
    # Show statistics
    stats = matcher.get_statistics()
    print(f"\nMatching Statistics:")
    print(f"   Total Processed: {stats.get('total_processed', 0)}")
    print(f"   Total Matched: {stats.get('total_matched', 0)}")
    print(f"   Match Rate: {stats.get('match_rate', 0):.1f}%")
    
    print(f"\nLevel Distribution:")
    for level in ["exact", "relaxed", "partial", "geographic", "fuzzy", "ultra_fuzzy", "semantic", "phonetic", "keyword", "desperate"]:
        count = stats.get(f"{level}_count", 0)
        percentage = stats.get(f"{level}_percentage", 0)
        if count > 0:
            print(f"   {level.title()}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    test_ultimate_address_matcher() 