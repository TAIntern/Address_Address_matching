# Ultimate Address Matcher

A comprehensive address processing and matching system with advanced parsing, normalization, and 10-level matching capabilities. Designed for high-accuracy address matching with special focus on unit/apartment matching and NYC addresses.

## Features

- **Advanced Address Parsing**: Uses `usaddress` library with intelligent fallback parsing
- **10-Level Matching System**: Progressive matching from exact to desperate levels
- **Enhanced Unit Matching**: Specialized handling for apartment/unit numbers
- **Multiple Address Variants**: Generates up to 15 address variants per input
- **Quality Scoring**: Component-based scoring system with reliability assessment
- **Elasticsearch Integration**: Connects to AWS OpenSearch for address data
- **Batch Processing**: Efficient processing of multiple addresses
- **95%+ Target Match Rate**: Optimized for maximum address resolution

## System Architecture

### Matching Levels

1. **Exact**: Perfect matches (≥95% similarity)
2. **Relaxed**: High-quality matches (≥85% similarity)  
3. **Partial**: Good matches (≥75% similarity)
4. **Geographic**: Location-based matches (≥65% similarity)
5. **Fuzzy**: Fuzzy string matches (≥55% similarity)
6. **Ultra Fuzzy**: More lenient fuzzy matches (≥45% similarity)
7. **Semantic**: Meaning-based matches (≥35% similarity)
8. **Phonetic**: Sound-based matches (≥25% similarity)
9. **Keyword**: Keyword-based matches (≥15% similarity)
10. **Desperate**: Last-resort matches (≥5% similarity)

### Address Components

- **Unit Number**: Apartment, suite, or unit identifiers
- **House Number**: Street address number
- **Street Name**: Street name and directionals
- **Street Type**: Street, Avenue, Boulevard, etc.
- **City**: City name with alias normalization
- **State**: State abbreviation
- **ZIP Code**: 5-digit postal code

## Installation

### Requirements

```bash
pip install usaddress unidecode rapidfuzz requests pydantic tqdm
```

### Required Files

- `ultimate_address_matcher.py` - Main matcher system
- `optimized.py` - Core matching engine and utilities

### Environment Setup

The system connects to an AWS OpenSearch instance. Default credentials are embedded, but you can override with environment variables:

```bash
export ES_URL="your-elasticsearch-url"
export ES_USER="your-username"
export ES_PWD="your-password"
```

## Usage

### Basic Usage

```python
from ultimate_address_matcher import UltimateAddressMatcher

# Initialize matcher
matcher = UltimateAddressMatcher()

# Match a single address
address = "85-101 North 3rd Street Brooklyn, NY 11249"
result = matcher.match_address(address, "your_index_name")

# Check results
if result.is_reliable:
    print(f"Match found: {result.original_result.matched}")
    print(f"Quality Score: {result.quality_score}")
    print(f"Confidence: {result.confidence_adjusted}")
    print(f"Reliability: {result.reliability}")
else:
    print("No reliable match found")
```

### Address Parsing

```python
# Parse address components
components = matcher.parse_address("1H 55 Berry Street Brooklyn, NY 11211")

print(f"Unit: {components.unit}")              # "1H"
print(f"House: {components.house_number}")     # "55"
print(f"Street: {components.street_name}")     # "Berry"
print(f"Type: {components.street_type}")       # "Street"
print(f"City: {components.city}")              # "Brooklyn"
print(f"State: {components.state}")            # "NY"
print(f"ZIP: {components.zip_code}")           # "11211"
```

### Generate Address Variants

```python
# Generate multiple address variants
variants = matcher.generate_variants("3R 112 Bedford Avenue Brooklyn, NY 11211")

for i, variant in enumerate(variants, 1):
    print(f"{i}. {variant}")
```

### Batch Processing

```python
# Process multiple addresses
addresses = [
    "85-101 North 3rd Street Brooklyn, NY 11249",
    "3R 112 Bedford Avenue Brooklyn, NY 11211",
    "1H 55 Berry Street Brooklyn, NY 11211"
]

results = []
for address in addresses:
    result = matcher.match_address(address, "your_index")
    results.append(result)

# Get statistics
stats = matcher.get_statistics()
print(f"Match Rate: {stats['match_rate']:.1f}%")
```

## Advanced Features

### Custom Field Mapping

```python
from optimized import IndexFieldMapping

# Define custom field mapping for your index
custom_mapping = IndexFieldMapping(
    house_field="house_number",
    street_field="street_name", 
    city_field="city_name",
    state_field="state_abbr",
    zip_field="postal_code",
    unit_field="apt_number"
)

matcher = UltimateAddressMatcher(custom_field_mapping=custom_mapping)
```

### Quality Assessment

```python
result = matcher.match_address(address, index)

# Check match quality
if result.is_reliable:
    print("High quality match")
elif result.is_questionable:
    print("Questionable match - review recommended")
elif result.is_speculative:
    print("Speculative match - low confidence")

# Check for warnings
if result.warning_flags:
    for warning in result.warning_flags:
        print(f"⚠️ {warning}")
```

## Command Line Interface

Run the system from command line using the underlying optimized matcher:

```bash
# Single address
python optimized.py "123 Main Street Brooklyn NY 11201"

# Batch processing
python optimized.py --batch-file addresses.txt --output results.csv

# Advanced options
python optimized.py \
    --batch-file input.txt \
    --output results.csv \
    --index your_index_name \
    --exact-match \
    --verbose
```

## Testing

Run the built-in test suite:

```python
python ultimate_address_matcher.py
```

This will test:
- Address parsing accuracy
- Variant generation
- Matching performance
- Statistics collection

## Configuration

### Match Configuration

```python
from optimized import MatchConfig

config = MatchConfig(
    street_weight=0.40,      # Street name importance
    house_weight=0.25,       # House number importance  
    city_weight=0.15,        # City name importance
    zip_weight=0.10,         # ZIP code importance
    state_weight=0.05,       # State importance
    unit_weight=0.05,        # Unit number importance
    min_similarity_threshold=70.0,
    max_results=10,
    batch_size=100
)
```

### Index Field Mapping

```python
mapping = IndexFieldMapping(
    house_field="HOUSE",         # House number field
    street_field="STREET",       # Street name field
    city_field="CITY",          # City field
    state_field="STATE",        # State field
    zip_field="ZIP_CODE",       # ZIP code field
    unit_field="APTNBR",        # Unit/apartment field
    address_field="ADDRESS"     # Full address field
)
```

## Special Features

### Enhanced Unit Matching

The system excels at matching complex unit/apartment addresses:

- Handles patterns like "1H 55 Berry Street" (unit + house + street)
- Recognizes "3R 112 Bedford Avenue" (unit prefix patterns)
- Processes "Apt 2B", "Unit 1A", "#3C" variations
- Includes unit variant generation and fuzzy matching

### NYC Address Optimization

Special handling for New York City addresses:

- Brooklyn, Manhattan, Queens, Bronx, Staten Island normalization
- Street name variants (Broadway, Kent Ave, North 3rd St, etc.)
- NYC-specific parsing patterns
- Geographic constraint validation

### Multi-Language Support

- Includes Chinese language comments and text
- Unicode-aware text processing
- International character normalization

## Error Handling

The system provides comprehensive error handling:

```python
result = matcher.match_address(address, index)

if result.original_result.error:
    print(f"Error: {result.original_result.error}")
    
# Check warning flags
for warning in result.warning_flags:
    print(f"Warning: {warning}")
```

## Performance

- **Target Match Rate**: 95%+
- **Processing Speed**: ~100-1000 addresses/minute (depending on complexity)
- **Memory Efficient**: Streaming batch processing
- **Connection Pooling**: Optimized Elasticsearch connections
- **Caching**: Built-in normalization caching

## Output Format

### Match Result Structure

```python
class UltimateMatchResult:
    original_result: MatchResult      # Original matching result
    match_level: str                  # Which level matched (exact, fuzzy, etc.)
    confidence_adjusted: float        # Adjusted confidence score (0-100)
    quality_score: float             # Quality assessment score (0-200)
    reliability: str                 # high, medium, low, very_low, speculative
    warning_flags: List[str]         # Any warning messages
```

### Statistics Output

```python
stats = matcher.get_statistics()
# Returns:
{
    'total_processed': 100,
    'total_matched': 95, 
    'match_rate': 95.0,
    'exact_count': 60,
    'exact_percentage': 60.0,
    'relaxed_count': 20,
    'relaxed_percentage': 20.0,
    # ... other levels
}
```

## Troubleshooting

### Common Issues

1. **usaddress Import Error**: Install with `pip install usaddress`
2. **Connection Timeout**: Check Elasticsearch URL and credentials
3. **Low Match Rates**: Verify index field mappings match your data schema
4. **Unit Matching Issues**: Check if your index has APTNBR field

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When contributing to this project:

1. Maintain the 10-level matching hierarchy
2. Test with NYC address patterns
3. Ensure unit matching accuracy
4. Validate against the 95% match rate target
5. Include appropriate error handling

## License

This project is designed for address matching applications with focus on real estate and delivery services. 
