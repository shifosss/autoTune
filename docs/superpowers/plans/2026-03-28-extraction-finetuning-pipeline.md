# Extraction Finetuning Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete finetuning project for 35-field structured extraction from messy text, integrated with the autoTune framework, running LoRA training via mlx-lm on Apple M5 Pro 64GB.

**Architecture:** Standalone git repo (`extraction-finetune/`) that autoTune overlays onto. Training script reads autoTune env vars, runs mlx-lm LoRA, evaluates field-level metrics, writes `result.json`. autoTune manages the DAG, manifold navigation across LoRA hyperparams, and checkpoint lifecycle.

**Tech Stack:** Python 3.11+, mlx-lm (LoRA/QLoRA on Apple Silicon), Qwen 2.5 7B Instruct (4-bit MLX), autoTune CLI

**Research reference:** `docs/research/2026-03-28-extraction-finetuning-research.md`

---

## File Structure

```
extraction-finetune/
├── train.py                    # Main training script (autoTune contract)
├── evaluate.py                 # Evaluation logic (field-level metrics)
├── generate_data.py            # Synthetic data generation script
├── schema.py                   # Pydantic schema for 35 extraction fields
├── data/
│   ├── schema_template.json    # Empty JSON template included in every prompt
│   ├── train.jsonl             # Training data (ChatML format)
│   ├── valid.jsonl             # Validation data (ChatML format)
│   └── test.jsonl              # Test data (held-out, ChatML format)
├── requirements.txt            # Project dependencies
└── .gitignore
```

After `autotune init`, the overlay is added:
```
├── autotune/
│   ├── config.yaml             # Customized for extraction task
│   ├── program.md
│   ├── learnings.md
│   ├── .gitignore
│   └── orchestrator/
│       ├── queue/
│       ├── running/
│       ├── completed/
│       ├── archive/
│       └── checkpoints/
```

---

## Phase 1: Project Scaffolding & Dependencies

### Task 1: Create Project Repository

**Files:**
- Create: `extraction-finetune/.gitignore`
- Create: `extraction-finetune/requirements.txt`

- [ ] **Step 1: Create project directory and initialize git**

```bash
mkdir -p extraction-finetune/data
cd extraction-finetune
git init
```

- [ ] **Step 2: Create .gitignore**

```gitignore
__pycache__/
*.pyc
.venv/
*.egg-info/
autotune/
.autotune_worktrees/
.parent_checkpoint/
.checkpoint_output/
*.safetensors
*.bin
*.pt
mlx_models/
```

- [ ] **Step 3: Create requirements.txt**

```
mlx-lm>=0.22.0
pydantic>=2.0
```

- [ ] **Step 4: Install dependencies**

```bash
cd extraction-finetune
pip install -r requirements.txt
```

Verify: `python -c "import mlx_lm; print(mlx_lm.__version__)"`

- [ ] **Step 5: Commit scaffolding**

```bash
git add .gitignore requirements.txt
git commit -m "chore: initialize extraction finetuning project"
```

---

### Task 2: Download Base Model

- [ ] **Step 1: Download Qwen 2.5 7B Instruct 4-bit MLX**

```bash
python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-7B-Instruct-4bit')"
```

This downloads to the HuggingFace cache (~4GB). Verify with:
```bash
python -c "from mlx_lm import load; model, tokenizer = load('mlx-community/Qwen2.5-7B-Instruct-4bit'); print('OK:', type(model))"
```

Expected: `OK: <class 'mlx_lm.models.qwen2.Model'>` (or similar)

If this fails with import errors, try: `pip install mlx mlx-lm --upgrade`

---

## Phase 2: Data Preparation

### Task 3: Define Extraction Schema

**Files:**
- Create: `schema.py`
- Create: `data/schema_template.json`

- [ ] **Step 1: Create schema.py with Pydantic model (35 fields)**

Domain: **commercial property listings** — naturally messy broker descriptions with inconsistent formatting, abbreviations, and missing info.

```python
"""Pydantic schema for commercial property listing extraction (35 fields)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class ContactInfo(BaseModel):
    agent_name: Optional[str] = None
    agent_phone: Optional[str] = None
    agent_email: Optional[str] = None
    brokerage: Optional[str] = None


class LocationInfo(BaseModel):
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    county: Optional[str] = None
    neighborhood: Optional[str] = None


class FinancialInfo(BaseModel):
    asking_price: Optional[float] = None
    price_per_sqft: Optional[float] = None
    cap_rate: Optional[float] = None
    noi: Optional[float] = None
    annual_taxes: Optional[float] = None
    hoa_monthly: Optional[float] = None


class PropertyListing(BaseModel):
    listing_id: Optional[str] = None
    property_type: Optional[str] = None
    listing_status: Optional[str] = None
    location: LocationInfo = LocationInfo()
    financial: FinancialInfo = FinancialInfo()
    total_sqft: Optional[int] = None
    lot_size_acres: Optional[float] = None
    year_built: Optional[int] = None
    num_floors: Optional[int] = None
    num_units: Optional[int] = None
    zoning: Optional[str] = None
    occupancy_pct: Optional[float] = None
    parking_spaces: Optional[int] = None
    parking_type: Optional[str] = None
    construction_type: Optional[str] = None
    roof_type: Optional[str] = None
    hvac_type: Optional[str] = None
    ceiling_height_ft: Optional[float] = None
    loading_docks: Optional[int] = None
    power_capacity_amps: Optional[int] = None
    fiber_internet: Optional[bool] = None
    sprinkler_system: Optional[bool] = None
    ada_compliant: Optional[bool] = None
    environmental_issues: Optional[str] = None
    property_condition: Optional[str] = None
    last_renovated: Optional[int] = None
    utilities_included: Optional[list[str]] = None
    amenities: Optional[list[str]] = None
    contact: ContactInfo = ContactInfo()


def empty_template() -> dict:
    """Return the empty JSON template for inclusion in prompts."""
    return PropertyListing().model_dump()


def schema_json() -> str:
    """Return the JSON schema string."""
    return PropertyListing.model_json_schema()


if __name__ == "__main__":
    template_path = Path("data/schema_template.json")
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(json.dumps(empty_template(), indent=2))
    print(f"Wrote template to {template_path}")
    print(f"Total fields: 35")
```

- [ ] **Step 2: Generate schema_template.json**

```bash
cd extraction-finetune
python schema.py
```

Expected output: `Wrote template to data/schema_template.json`

Verify the template has all 35 fields (including nested): `python -c "import json; d=json.load(open('data/schema_template.json')); print(json.dumps(d, indent=2))" | head -40`

- [ ] **Step 3: Commit**

```bash
git add schema.py data/schema_template.json
git commit -m "feat: define 35-field property listing extraction schema"
```

---

### Task 4: Create Synthetic Data Generator

**Files:**
- Create: `generate_data.py`

- [ ] **Step 1: Create generate_data.py**

This script generates synthetic training examples using the **reverse generation** approach — it creates realistic messy listing text from random structured data.

```python
"""Generate synthetic training data for property listing extraction.

Uses reverse generation: creates structured data first, then generates
realistic messy text containing that information.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path

from schema import PropertyListing, empty_template

SYSTEM_PROMPT = (
    "You are an expert at structured data extraction. "
    "You will be given unstructured text from a commercial property listing "
    "and must extract information into the provided JSON schema. "
    "Output valid JSON only. Use null for fields not found in the text."
)

# ── building blocks for synthetic data ──────────────────────────────────────

PROPERTY_TYPES = [
    "office", "retail", "industrial", "warehouse", "mixed-use",
    "medical office", "restaurant", "flex space", "self-storage",
    "strip mall", "shopping center",
]
STATUSES = ["active", "pending", "reduced", "new listing", "back on market"]
CITIES = [
    ("Austin", "TX", "78701"), ("Denver", "CO", "80202"),
    ("Portland", "OR", "97201"), ("Nashville", "TN", "37201"),
    ("Charlotte", "NC", "28202"), ("Tampa", "FL", "33602"),
    ("Phoenix", "AZ", "85004"), ("San Diego", "CA", "92101"),
    ("Raleigh", "NC", "27601"), ("Salt Lake City", "UT", "84101"),
    ("Boise", "ID", "83702"), ("Minneapolis", "MN", "55401"),
]
STREETS = [
    "Main St", "Commerce Blvd", "Industrial Pkwy", "Oak Ave",
    "Business Center Dr", "Warehouse Row", "Market St", "Tech Park Ln",
    "Enterprise Way", "Airport Blvd", "Highway 9 Frontage",
]
COUNTIES = ["Travis", "Denver", "Multnomah", "Davidson", "Mecklenburg", "Hillsborough"]
NEIGHBORHOODS = ["Downtown", "Midtown", "Southside", "East End", "Tech District", "Arts District"]
ZONING = ["C-1", "C-2", "M-1", "M-2", "I-1", "MU-1", "PUD", "B-2", "LI"]
CONSTRUCTION = ["steel frame", "concrete block", "tilt-up concrete", "wood frame", "masonry", "pre-engineered metal"]
ROOFS = ["flat membrane", "TPO", "metal standing seam", "built-up", "EPDM"]
HVAC = ["central forced air", "rooftop units", "split system", "VRF", "radiant", "none"]
PARKING = ["surface lot", "covered", "underground garage", "street only", "shared lot"]
CONDITIONS = ["excellent", "good", "fair", "needs renovation", "shell condition", "recently renovated"]
UTILITIES = ["water", "sewer", "electric", "gas", "trash", "internet"]
AMENITIES = [
    "conference room", "break room", "elevator", "lobby", "security system",
    "loading bay", "drive-in door", "fenced yard", "signage", "kitchenette",
    "shower facilities", "on-site management", "CCTV", "card access",
]
BROKERAGES = [
    "CBRE", "JLL", "Cushman & Wakefield", "Marcus & Millichap",
    "Keller Williams Commercial", "RE/MAX Commercial", "NAI Global",
]
FIRST_NAMES = ["Mike", "Sarah", "David", "Jennifer", "Tom", "Lisa", "James", "Amanda", "Robert", "Karen"]
LAST_NAMES = ["Johnson", "Smith", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson"]
ENV_ISSUES = [None, None, None, None, "Phase I ESA clear", "asbestos abated 2019", "underground tank removed", None]


def random_phone() -> str:
    return f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"


def random_email(first: str, last: str) -> str:
    domains = ["realty.com", "cre-group.com", "commercial.net", "gmail.com", "properties.co"]
    return f"{first.lower()}.{last.lower()}@{random.choice(domains)}"


def random_listing_id() -> str:
    prefix = random.choice(["CRE", "MLS", "LX", "P", "COM"])
    return f"{prefix}-{random.randint(100000, 999999)}"


def generate_structured() -> dict:
    """Generate a random structured property listing."""
    city, state, zip_code = random.choice(CITIES)
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    prop_type = random.choice(PROPERTY_TYPES)
    sqft = random.choice([None, random.randint(800, 50000)])
    year = random.choice([None, random.randint(1955, 2025)])
    price = random.choice([None, random.randint(150000, 12000000)])
    floors = random.choice([None, 1, 1, 2, 2, 3, 4])
    units = random.choice([None, 1, 2, 4, 6, 8, 12])

    listing = PropertyListing(
        listing_id=random_listing_id() if random.random() > 0.1 else None,
        property_type=prop_type,
        listing_status=random.choice(STATUSES) if random.random() > 0.15 else None,
        location={
            "street_address": f"{random.randint(100,9999)} {random.choice(STREETS)}" if random.random() > 0.05 else None,
            "city": city,
            "state": state,
            "zip_code": zip_code if random.random() > 0.1 else None,
            "county": random.choice(COUNTIES) if random.random() > 0.6 else None,
            "neighborhood": random.choice(NEIGHBORHOODS) if random.random() > 0.5 else None,
        },
        financial={
            "asking_price": price,
            "price_per_sqft": round(price / sqft, 2) if price and sqft else None,
            "cap_rate": round(random.uniform(4.0, 9.5), 1) if random.random() > 0.4 else None,
            "noi": random.randint(20000, 800000) if random.random() > 0.5 else None,
            "annual_taxes": random.randint(3000, 120000) if random.random() > 0.4 else None,
            "hoa_monthly": random.choice([None, None, 250, 500, 750, 1200]),
        },
        total_sqft=sqft,
        lot_size_acres=round(random.uniform(0.1, 5.0), 2) if random.random() > 0.3 else None,
        year_built=year,
        num_floors=floors,
        num_units=units,
        zoning=random.choice(ZONING) if random.random() > 0.2 else None,
        occupancy_pct=random.choice([None, 100, 95, 85, 75, 60, 0]),
        parking_spaces=random.randint(2, 100) if random.random() > 0.3 else None,
        parking_type=random.choice(PARKING) if random.random() > 0.3 else None,
        construction_type=random.choice(CONSTRUCTION) if random.random() > 0.4 else None,
        roof_type=random.choice(ROOFS) if random.random() > 0.5 else None,
        hvac_type=random.choice(HVAC) if random.random() > 0.35 else None,
        ceiling_height_ft=random.choice([None, 9, 10, 12, 14, 16, 20, 24, 28]) if random.random() > 0.4 else None,
        loading_docks=random.choice([None, 0, 1, 2, 3, 4]) if "warehouse" in prop_type or "industrial" in prop_type else None,
        power_capacity_amps=random.choice([None, 200, 400, 600, 800, 1200]) if random.random() > 0.6 else None,
        fiber_internet=random.choice([None, True, True, False]) if random.random() > 0.4 else None,
        sprinkler_system=random.choice([None, True, True, False]),
        ada_compliant=random.choice([None, True, True, False]) if random.random() > 0.3 else None,
        environmental_issues=random.choice(ENV_ISSUES),
        property_condition=random.choice(CONDITIONS) if random.random() > 0.2 else None,
        last_renovated=random.randint(2005, 2025) if random.random() > 0.5 else None,
        utilities_included=random.sample(UTILITIES, random.randint(2, 5)) if random.random() > 0.3 else None,
        amenities=random.sample(AMENITIES, random.randint(1, 6)) if random.random() > 0.2 else None,
        contact={
            "agent_name": f"{first} {last}",
            "agent_phone": random_phone(),
            "agent_email": random_email(first, last) if random.random() > 0.2 else None,
            "brokerage": random.choice(BROKERAGES) if random.random() > 0.3 else None,
        },
    )
    return listing.model_dump()


# ── messy text generation templates ─────────────────────────────────────────

TEMPLATES = [
    # Style 1: Terse broker shorthand
    lambda d: _terse_style(d),
    # Style 2: Conversational marketing
    lambda d: _marketing_style(d),
    # Style 3: Data dump / MLS-like
    lambda d: _mls_style(d),
    # Style 4: Email forwarded listing
    lambda d: _email_style(d),
    # Style 5: Mixed / sloppy notes
    lambda d: _notes_style(d),
]


def _fmt_price(p: float | None) -> str:
    if p is None:
        return ""
    if p >= 1_000_000:
        return random.choice([f"${p/1e6:.1f}M", f"${p:,.0f}", f"{p/1e6:.2f} mil"])
    return random.choice([f"${p:,.0f}", f"${p/1000:.0f}K", f"${p:,}"])


def _maybe(val, templates: list[str], prob: float = 0.85) -> str:
    if val is None or random.random() > prob:
        return ""
    tmpl = random.choice(templates)
    return tmpl.format(val=val)


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict for template access."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{key}_"))
        else:
            flat[key] = v
    return flat


def _terse_style(d: dict) -> str:
    f = _flatten(d)
    parts = []
    if f.get("listing_id"):
        parts.append(f"#{f['listing_id']}")
    if f.get("property_type"):
        parts.append(f["property_type"].upper())
    if f.get("listing_status"):
        parts.append(f"— {f['listing_status']}")
    parts.append("")  # newline
    if f.get("location_street_address"):
        parts.append(f"{f['location_street_address']}")
    loc_parts = [x for x in [f.get("location_city"), f.get("location_state"), f.get("location_zip_code")] if x]
    if loc_parts:
        parts.append(", ".join(str(x) for x in loc_parts))
    parts.append("")
    if f.get("financial_asking_price"):
        parts.append(f"Asking: {_fmt_price(f['financial_asking_price'])}")
    specs = []
    if f.get("total_sqft"):
        specs.append(f"{f['total_sqft']:,} sf")
    if f.get("lot_size_acres"):
        specs.append(f"{f['lot_size_acres']} ac lot")
    if f.get("year_built"):
        specs.append(f"built {f['year_built']}")
    if f.get("num_floors"):
        specs.append(f"{f['num_floors']} flrs")
    if specs:
        parts.append(" | ".join(specs))
    if f.get("zoning"):
        parts.append(f"Zoning: {f['zoning']}")
    if f.get("financial_cap_rate"):
        parts.append(f"Cap: {f['financial_cap_rate']}%")
    if f.get("financial_noi"):
        parts.append(f"NOI: ${f['financial_noi']:,}")
    details = []
    if f.get("construction_type"):
        details.append(f["construction_type"])
    if f.get("hvac_type"):
        details.append(f"HVAC: {f['hvac_type']}")
    if f.get("parking_spaces"):
        details.append(f"{f['parking_spaces']} parking")
    if f.get("ceiling_height_ft"):
        details.append(f"{f['ceiling_height_ft']}' clear")
    if details:
        parts.append(", ".join(details))
    if f.get("contact_agent_name"):
        contact = f["contact_agent_name"]
        if f.get("contact_agent_phone"):
            contact += f" {f['contact_agent_phone']}"
        parts.append(contact)
    return "\n".join(parts)


def _marketing_style(d: dict) -> str:
    f = _flatten(d)
    parts = []
    openers = [
        "Don't miss this incredible opportunity!",
        "Prime commercial property now available.",
        "Rare find in a hot market!",
        "Investor's dream property.",
        "Turnkey commercial space ready for business.",
        "Exceptional value — priced to sell!",
    ]
    parts.append(random.choice(openers))
    if f.get("property_type"):
        adj = random.choice(["fantastic", "spacious", "well-maintained", "stunning", "versatile"])
        parts.append(f"This {adj} {f['property_type']} property")
        if f.get("location_neighborhood"):
            parts[-1] += f" in the heart of {f['location_neighborhood']}"
        if f.get("location_city"):
            parts[-1] += f", {f['location_city']}"
        parts[-1] += " is waiting for you."
    if f.get("total_sqft"):
        parts.append(f"Boasting {f['total_sqft']:,} square feet")
        if f.get("lot_size_acres"):
            parts[-1] += f" on a generous {f['lot_size_acres']}-acre lot"
        parts[-1] += "."
    if f.get("year_built"):
        parts.append(f"Originally built in {f['year_built']}")
        if f.get("last_renovated"):
            parts[-1] += f" and beautifully renovated in {f['last_renovated']}"
        parts[-1] += "."
    if f.get("financial_asking_price"):
        parts.append(f"Offered at {_fmt_price(f['financial_asking_price'])}.")
    if f.get("financial_cap_rate"):
        parts.append(f"Currently showing a solid {f['financial_cap_rate']}% cap rate.")
    features = []
    if f.get("parking_spaces"):
        features.append(f"{f['parking_spaces']} dedicated parking spaces ({f.get('parking_type', 'surface lot')})")
    if f.get("fiber_internet"):
        features.append("fiber internet connectivity")
    if f.get("sprinkler_system"):
        features.append("full sprinkler system")
    if f.get("ada_compliant"):
        features.append("ADA compliant")
    if features:
        parts.append("Features include " + ", ".join(features) + ".")
    if f.get("amenities"):
        parts.append("Amenities: " + ", ".join(f["amenities"]) + ".")
    if f.get("contact_agent_name"):
        parts.append(f"Contact {f['contact_agent_name']}")
        if f.get("contact_brokerage"):
            parts[-1] += f" at {f['contact_brokerage']}"
        if f.get("contact_agent_phone"):
            parts[-1] += f" — {f['contact_agent_phone']}"
        parts[-1] += " for a showing."
    return " ".join(parts)


def _mls_style(d: dict) -> str:
    f = _flatten(d)
    lines = []
    if f.get("listing_id"):
        lines.append(f"MLS# {f['listing_id']}")
    if f.get("location_street_address"):
        addr = f["location_street_address"]
        if f.get("location_city"):
            addr += f", {f['location_city']}"
        if f.get("location_state"):
            addr += f", {f['location_state']}"
        if f.get("location_zip_code"):
            addr += f" {f['location_zip_code']}"
        lines.append(f"Address: {addr}")
    kv_pairs = [
        ("Type", f.get("property_type")),
        ("Status", f.get("listing_status")),
        ("Price", _fmt_price(f.get("financial_asking_price")) if f.get("financial_asking_price") else None),
        ("Sq Ft", f"{f['total_sqft']:,}" if f.get("total_sqft") else None),
        ("$/SF", f"${f['financial_price_per_sqft']:,.2f}" if f.get("financial_price_per_sqft") else None),
        ("Lot", f"{f['lot_size_acres']} acres" if f.get("lot_size_acres") else None),
        ("Year Built", f.get("year_built")),
        ("Floors", f.get("num_floors")),
        ("Units", f.get("num_units")),
        ("Zoning", f.get("zoning")),
        ("Occupancy", f"{f['occupancy_pct']}%" if f.get("occupancy_pct") is not None else None),
        ("Cap Rate", f"{f['financial_cap_rate']}%" if f.get("financial_cap_rate") else None),
        ("NOI", f"${f['financial_noi']:,}" if f.get("financial_noi") else None),
        ("Taxes", f"${f['financial_annual_taxes']:,}/yr" if f.get("financial_annual_taxes") else None),
        ("HOA", f"${f['financial_hoa_monthly']}/mo" if f.get("financial_hoa_monthly") else None),
        ("Parking", f"{f['parking_spaces']} ({f.get('parking_type', '')})" if f.get("parking_spaces") else None),
        ("Construction", f.get("construction_type")),
        ("Roof", f.get("roof_type")),
        ("HVAC", f.get("hvac_type")),
        ("Ceiling Ht", f"{f['ceiling_height_ft']}'" if f.get("ceiling_height_ft") else None),
        ("Loading Docks", f.get("loading_docks")),
        ("Power", f"{f['power_capacity_amps']}A" if f.get("power_capacity_amps") else None),
        ("Fiber", "Yes" if f.get("fiber_internet") else ("No" if f.get("fiber_internet") is False else None)),
        ("Sprinkler", "Yes" if f.get("sprinkler_system") else ("No" if f.get("sprinkler_system") is False else None)),
        ("ADA", "Yes" if f.get("ada_compliant") else ("No" if f.get("ada_compliant") is False else None)),
        ("Condition", f.get("property_condition")),
        ("Renovated", f.get("last_renovated")),
        ("Environmental", f.get("environmental_issues")),
    ]
    for label, val in kv_pairs:
        if val is not None:
            lines.append(f"{label}: {val}")
    if f.get("utilities_included"):
        lines.append(f"Utilities: {', '.join(f['utilities_included'])}")
    if f.get("amenities"):
        lines.append(f"Amenities: {', '.join(f['amenities'])}")
    if f.get("location_county"):
        lines.append(f"County: {f['location_county']}")
    if f.get("contact_agent_name"):
        lines.append(f"Agent: {f['contact_agent_name']}")
        if f.get("contact_agent_phone"):
            lines.append(f"Phone: {f['contact_agent_phone']}")
        if f.get("contact_agent_email"):
            lines.append(f"Email: {f['contact_agent_email']}")
        if f.get("contact_brokerage"):
            lines.append(f"Brokerage: {f['contact_brokerage']}")
    return "\n".join(lines)


def _email_style(d: dict) -> str:
    f = _flatten(d)
    first = random.choice(["Hey", "Hi team", "FYI", "FW:", "Forwarding this"])
    lines = [f"{first} — just saw this come across, thought you'd want to look.\n"]
    if f.get("property_type") and f.get("location_city"):
        lines.append(f"There's a {f['property_type']} in {f['location_city']} that just hit the market.")
    if f.get("financial_asking_price"):
        lines.append(f"They're asking {_fmt_price(f['financial_asking_price'])}.")
    if f.get("total_sqft"):
        lines.append(f"About {f['total_sqft']:,} sqft give or take.")
    details = []
    if f.get("year_built"):
        details.append(f"built in {f['year_built']}")
    if f.get("zoning"):
        details.append(f"zoned {f['zoning']}")
    if f.get("financial_cap_rate"):
        details.append(f"cap rate around {f['financial_cap_rate']}%")
    if details:
        lines.append("Quick details: " + ", ".join(details) + ".")
    if f.get("property_condition"):
        lines.append(f"Condition looks {f['property_condition']}.")
    if f.get("parking_spaces"):
        lines.append(f"Has {f['parking_spaces']} parking spots.")
    if f.get("location_street_address"):
        addr = f["location_street_address"]
        if f.get("location_city"):
            addr += f", {f['location_city']}"
        if f.get("location_state"):
            addr += f" {f['location_state']}"
        if f.get("location_zip_code"):
            addr += f" {f['location_zip_code']}"
        lines.append(f"\nAddress: {addr}")
    if f.get("contact_agent_name"):
        lines.append(f"\nListing agent is {f['contact_agent_name']}")
        if f.get("contact_agent_phone"):
            lines[-1] += f", reach them at {f['contact_agent_phone']}"
        if f.get("contact_brokerage"):
            lines[-1] += f" ({f['contact_brokerage']})"
    lines.append("\nLet me know if you want me to schedule a walkthrough.")
    return "\n".join(lines)


def _notes_style(d: dict) -> str:
    f = _flatten(d)
    parts = []
    if f.get("location_street_address"):
        parts.append(f"property @ {f['location_street_address']}")
    if f.get("location_city"):
        parts[-1] += f" {f['location_city']}" if parts else f"in {f['location_city']}"
    if f.get("property_type"):
        parts.append(f"  - type: {f['property_type']}")
    if f.get("total_sqft"):
        parts.append(f"  - {f['total_sqft']} sqft (approx)")
    if f.get("financial_asking_price"):
        parts.append(f"  - price {_fmt_price(f['financial_asking_price'])} (?? check this)")
    if f.get("year_built"):
        parts.append(f"  - yr built {f['year_built']}")
    if f.get("zoning"):
        parts.append(f"  - zoning {f['zoning']}")
    if f.get("financial_cap_rate"):
        parts.append(f"  - cap {f['financial_cap_rate']}%")
    if f.get("occupancy_pct") is not None:
        parts.append(f"  - {f['occupancy_pct']}% occupied")
    misc = []
    if f.get("hvac_type"):
        misc.append(f"hvac={f['hvac_type']}")
    if f.get("construction_type"):
        misc.append(f["construction_type"])
    if f.get("ceiling_height_ft"):
        misc.append(f"{f['ceiling_height_ft']}ft ceilings")
    if f.get("loading_docks"):
        misc.append(f"{f['loading_docks']} docks")
    if misc:
        parts.append(f"  - {' / '.join(misc)}")
    bools = []
    if f.get("fiber_internet"):
        bools.append("fiber=Y")
    if f.get("sprinkler_system"):
        bools.append("sprinkler=Y")
    if f.get("ada_compliant"):
        bools.append("ADA=Y")
    if bools:
        parts.append(f"  - {', '.join(bools)}")
    if f.get("contact_agent_name"):
        parts.append(f"\nagent: {f['contact_agent_name']} ph:{f.get('contact_agent_phone', '?')}")
    return "\n".join(parts)


def build_example(structured: dict) -> dict:
    """Build a ChatML training example from structured data."""
    template = empty_template()
    text = random.choice(TEMPLATES)(structured)
    schema_str = json.dumps(template, indent=2)
    output_str = json.dumps(structured, indent=2)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{text}\n\nExtract the following fields:\n{schema_str}",
            },
            {"role": "assistant", "content": output_str},
        ]
    }


def generate_dataset(
    n_train: int = 200,
    n_valid: int = 30,
    n_test: int = 30,
    seed: int = 42,
) -> None:
    """Generate train/valid/test splits and write to data/."""
    random.seed(seed)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    for split, count in [("train", n_train), ("valid", n_valid), ("test", n_test)]:
        examples = []
        for _ in range(count):
            structured = generate_structured()
            example = build_example(structured)
            examples.append(example)
        path = data_dir / f"{split}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {count} examples to {path}")

    print(f"\nTotal: {n_train + n_valid + n_test} examples")


if __name__ == "__main__":
    generate_dataset()
```

- [ ] **Step 2: Generate training data**

```bash
cd extraction-finetune
python generate_data.py
```

Expected:
```
Wrote 200 examples to data/train.jsonl
Wrote 30 examples to data/valid.jsonl
Wrote 30 examples to data/test.jsonl

Total: 260 examples
```

- [ ] **Step 3: Validate data format**

```bash
python -c "
import json
with open('data/train.jsonl') as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        assert 'messages' in d
        assert len(d['messages']) == 3
        assert d['messages'][0]['role'] == 'system'
        assert d['messages'][1]['role'] == 'user'
        assert d['messages'][2]['role'] == 'assistant'
        out = json.loads(d['messages'][2]['content'])
        assert 'listing_id' in out  # top-level field
        assert 'location' in out     # nested group
        if i >= 4: break
print('All validation passed')
"
```

- [ ] **Step 4: Commit**

```bash
git add generate_data.py data/
git commit -m "feat: add synthetic data generator and 260 training examples"
```

---

## Phase 3: Training & Evaluation Scripts

### Task 5: Create Evaluation Module

**Files:**
- Create: `evaluate.py`

- [ ] **Step 1: Create evaluate.py with field-level metrics**

```python
"""Field-level evaluation metrics for structured extraction.

Implements the metric stack from the research report:
  Level 0: JSON validity, schema compliance
  Level 1: Per-field exact match, fuzzy match
  Level 2: Macro-avg field accuracy, aggregate exact match
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FieldResult:
    field_name: str
    predicted: Any
    expected: Any
    exact_match: bool
    fuzzy_score: float  # 0.0 - 1.0


@dataclass(frozen=True)
class SampleResult:
    json_valid: bool
    schema_compliant: bool
    field_results: tuple[FieldResult, ...]
    all_fields_correct: bool


@dataclass(frozen=True)
class EvalReport:
    n_samples: int
    json_validity_rate: float
    schema_compliance_rate: float
    aggregate_exact_match: float
    macro_field_accuracy: float
    per_field_accuracy: dict[str, float]
    per_field_fuzzy: dict[str, float]
    hallucination_rate: float


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict to dot-separated keys."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key))
        else:
            flat[key] = v
    return flat


def _fuzzy_score(a: Any, b: Any) -> float:
    """Compute fuzzy similarity between two values."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        a_set = set(str(x).lower().strip() for x in a)
        b_set = set(str(x).lower().strip() for x in b)
        if not a_set and not b_set:
            return 1.0
        intersection = a_set & b_set
        union = a_set | b_set
        return len(intersection) / len(union) if union else 1.0
    if isinstance(a, bool) or isinstance(b, bool):
        return 1.0 if a == b else 0.0
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return 1.0
        max_val = max(abs(a), abs(b), 1e-9)
        return max(0.0, 1.0 - abs(a - b) / max_val)
    sa, sb = str(a).lower().strip(), str(b).lower().strip()
    return SequenceMatcher(None, sa, sb).ratio()


def evaluate_sample(predicted_str: str, expected: dict, required_keys: set[str] | None = None) -> SampleResult:
    """Evaluate a single prediction against ground truth."""
    # Level 0: JSON validity
    try:
        predicted = json.loads(predicted_str)
        json_valid = True
    except (json.JSONDecodeError, TypeError):
        n_fields = len(_flatten_dict(expected))
        field_results = tuple(
            FieldResult(k, None, v, False, 0.0)
            for k, v in _flatten_dict(expected).items()
        )
        return SampleResult(
            json_valid=False,
            schema_compliant=False,
            field_results=field_results,
            all_fields_correct=False,
        )

    # Level 0: Schema compliance (check required keys present)
    flat_pred = _flatten_dict(predicted)
    flat_exp = _flatten_dict(expected)
    if required_keys is None:
        required_keys = set(flat_exp.keys())
    schema_compliant = required_keys.issubset(set(flat_pred.keys()))

    # Level 1: Per-field comparison
    all_keys = set(flat_exp.keys())
    field_results = []
    all_correct = True
    for key in sorted(all_keys):
        exp_val = flat_exp.get(key)
        pred_val = flat_pred.get(key)
        exact = _values_match(pred_val, exp_val)
        fuzzy = _fuzzy_score(pred_val, exp_val)
        if not exact:
            all_correct = False
        field_results.append(FieldResult(key, pred_val, exp_val, exact, fuzzy))

    return SampleResult(
        json_valid=json_valid,
        schema_compliant=schema_compliant,
        field_results=tuple(field_results),
        all_fields_correct=all_correct,
    )


def _values_match(a: Any, b: Any) -> bool:
    """Check if two values match (with type coercion)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return set(str(x).lower().strip() for x in a) == set(str(x).lower().strip() for x in b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-6 or a == b
    return str(a).lower().strip() == str(b).lower().strip()


def evaluate_dataset(predictions: list[str], ground_truths: list[dict]) -> EvalReport:
    """Evaluate a full dataset of predictions against ground truths."""
    results = [
        evaluate_sample(pred, gt)
        for pred, gt in zip(predictions, ground_truths)
    ]
    n = len(results)
    if n == 0:
        return EvalReport(0, 0.0, 0.0, 0.0, 0.0, {}, {}, 0.0)

    json_valid = sum(1 for r in results if r.json_valid) / n
    schema_ok = sum(1 for r in results if r.schema_compliant) / n
    agg_em = sum(1 for r in results if r.all_fields_correct) / n

    # Per-field accuracy
    field_correct: dict[str, list[bool]] = defaultdict(list)
    field_fuzzy: dict[str, list[float]] = defaultdict(list)
    hallucination_count = 0
    total_fields = 0

    for r in results:
        for fr in r.field_results:
            field_correct[fr.field_name].append(fr.exact_match)
            field_fuzzy[fr.field_name].append(fr.fuzzy_score)
            total_fields += 1
            # Hallucination: predicted non-null when expected null
            if fr.expected is None and fr.predicted is not None:
                hallucination_count += 1

    per_field_acc = {k: sum(v) / len(v) for k, v in field_correct.items()}
    per_field_fuz = {k: sum(v) / len(v) for k, v in field_fuzzy.items()}
    macro_acc = sum(per_field_acc.values()) / len(per_field_acc) if per_field_acc else 0.0
    hallucination_rate = hallucination_count / total_fields if total_fields else 0.0

    return EvalReport(
        n_samples=n,
        json_validity_rate=round(json_valid, 4),
        schema_compliance_rate=round(schema_ok, 4),
        aggregate_exact_match=round(agg_em, 4),
        macro_field_accuracy=round(macro_acc, 4),
        per_field_accuracy={k: round(v, 4) for k, v in sorted(per_field_acc.items())},
        per_field_fuzzy={k: round(v, 4) for k, v in sorted(per_field_fuz.items())},
        hallucination_rate=round(hallucination_rate, 4),
    )


def run_eval_on_file(predictions_path: str, ground_truth_path: str) -> EvalReport:
    """Run evaluation from file paths (used by train.py)."""
    preds = Path(predictions_path).read_text().strip().splitlines()
    gts = []
    with open(ground_truth_path) as f:
        for line in f:
            msg = json.loads(line)
            gt = json.loads(msg["messages"][2]["content"])
            gts.append(gt)
    return evaluate_dataset(preds, gts)
```

- [ ] **Step 2: Commit**

```bash
git add evaluate.py
git commit -m "feat: add field-level extraction evaluation metrics"
```

---

### Task 6: Create Training Script

**Files:**
- Create: `train.py`

This is the core script that autoTune's orchestrator will launch. It reads env vars, runs mlx-lm LoRA training, evaluates, and writes `result.json`.

- [ ] **Step 1: Verify mlx-lm LoRA CLI interface**

```bash
python -m mlx_lm.lora --help
```

Note the exact flag names for: model, data, adapter-path, learning-rate, batch-size, lora-layers, lora-rank, num-layers, iters, save-every, resume-adapter-file. These vary by mlx-lm version.

- [ ] **Step 2: Create train.py**

```python
"""Training script for structured extraction LoRA finetuning.

Conforms to the autoTune result contract:
- Reads AUTOTUNE_* env vars for checkpoint management
- Writes result.json to working directory on completion
- Saves LoRA adapter to AUTOTUNE_CHECKPOINT_DIR
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── autoTune env vars ───────────────────────────────────────────────────────

NODE_ID = os.environ.get("AUTOTUNE_NODE_ID", "manual_run")
CHECKPOINT_DIR = os.environ.get("AUTOTUNE_CHECKPOINT_DIR", "./checkpoint_output")
PARENT_CHECKPOINT = os.environ.get("AUTOTUNE_PARENT_CHECKPOINT", "")
MOVE_TYPE = os.environ.get("AUTOTUNE_MOVE_TYPE", "seed")

# ── defaults (overridden by spec.json theta if present) ────────────────────

BASE_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DATA_DIR = "data"
DEFAULT_THETA = {
    "lr": 2e-4,
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "warmup_ratio": 0.06,
}


def load_theta() -> dict:
    """Load hyperparameters from spec.json or env vars, falling back to defaults."""
    theta = dict(DEFAULT_THETA)

    # Try spec.json first (written by orchestrator to archive, or manually placed)
    spec_path = Path("spec.json")
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        spec_theta = spec.get("theta", {})
        for k, v in spec_theta.items():
            if k in theta:
                theta[k] = type(theta[k])(v)
        logger.info("Loaded theta from spec.json: %s", spec_theta)
    else:
        # Fall back to AUTOTUNE_THETA_* env vars
        for key in DEFAULT_THETA:
            env_key = f"AUTOTUNE_THETA_{key.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                theta[key] = type(DEFAULT_THETA[key])(env_val)
                logger.info("Loaded %s=%s from env", key, theta[key])

    return theta


def find_parent_adapter() -> str | None:
    """Find a parent LoRA adapter in the parent checkpoint directory."""
    if not PARENT_CHECKPOINT:
        return None
    parent = Path(PARENT_CHECKPOINT)
    if not parent.exists():
        return None
    # mlx-lm saves adapters as adapters.safetensors (+ adapter_config.json)
    for name in ["adapters.safetensors", "adapter_model.safetensors"]:
        if (parent / name).exists():
            logger.info("Found parent adapter: %s", parent / name)
            return str(parent)
    # Check subdirectories (autoTune may nest under node_id)
    for subdir in parent.iterdir():
        if subdir.is_dir():
            for name in ["adapters.safetensors", "adapter_model.safetensors"]:
                if (subdir / name).exists():
                    logger.info("Found parent adapter in subdir: %s", subdir)
                    return str(subdir)
    logger.info("No parent adapter found in %s — training from base model", parent)
    return None


def run_training(theta: dict) -> tuple[float, float]:
    """Run mlx-lm LoRA training. Returns (train_loss, val_loss)."""
    ckpt = Path(CHECKPOINT_DIR)
    ckpt.mkdir(parents=True, exist_ok=True)

    parent_adapter = find_parent_adapter()

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", BASE_MODEL,
        "--train",
        "--data", DATA_DIR,
        "--adapter-path", str(ckpt),
        "--batch-size", "2",
        "--lora-layers", "16",
        "--iters", "200",
        "--learning-rate", str(theta["lr"]),
        "--save-every", "50",
        "--val-batches", "10",
    ]

    if parent_adapter:
        cmd.extend(["--resume-adapter-file", os.path.join(parent_adapter, "adapters.safetensors")])

    logger.info("Training command: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
    )

    logger.info("STDOUT:\n%s", result.stdout[-3000:] if result.stdout else "(empty)")
    if result.stderr:
        logger.warning("STDERR:\n%s", result.stderr[-2000:])

    if result.returncode != 0:
        raise RuntimeError(f"Training failed (exit {result.returncode}):\n{result.stderr[-1000:]}")

    # Parse final losses from mlx-lm output
    train_loss = _parse_final_loss(result.stdout, "Train")
    val_loss = _parse_final_loss(result.stdout, "Val")
    return train_loss, val_loss


def _parse_final_loss(output: str, prefix: str) -> float:
    """Parse the last reported loss from mlx-lm output."""
    last_loss = float("inf")
    for line in output.splitlines():
        if prefix in line and "loss" in line.lower():
            # mlx-lm format: "Iter 100: Train loss 1.234, ..."
            # or "Val loss 1.234 ..."
            parts = line.split()
            for i, part in enumerate(parts):
                if part.lower() == "loss" and i + 1 < len(parts):
                    try:
                        val = parts[i + 1].rstrip(",").rstrip(".")
                        last_loss = float(val)
                    except ValueError:
                        continue
    return last_loss


def run_evaluation(theta: dict) -> dict:
    """Run model on validation set and compute metrics."""
    from evaluate import evaluate_dataset

    ckpt = Path(CHECKPOINT_DIR)
    predictions = []
    ground_truths = []

    valid_path = Path(DATA_DIR) / "valid.jsonl"
    if not valid_path.exists():
        logger.warning("No validation file found at %s", valid_path)
        return {"macro_field_accuracy": 0.0, "aggregate_exact_match": 0.0}

    with open(valid_path) as f:
        examples = [json.loads(line) for line in f]

    # Generate predictions using the finetuned model
    from mlx_lm import generate, load

    model, tokenizer = load(BASE_MODEL, adapter_path=str(ckpt))

    for ex in examples:
        messages = ex["messages"][:2]  # system + user only
        gt = json.loads(ex["messages"][2]["content"])
        ground_truths.append(gt)

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(
            model, tokenizer, prompt=prompt, max_tokens=2048, temp=0.0
        )
        predictions.append(response)

    report = evaluate_dataset(predictions, ground_truths)
    logger.info(
        "Eval: json_valid=%.2f schema=%.2f macro_acc=%.4f agg_em=%.4f halluc=%.4f",
        report.json_validity_rate,
        report.schema_compliance_rate,
        report.macro_field_accuracy,
        report.aggregate_exact_match,
        report.hallucination_rate,
    )
    return {
        "json_validity_rate": report.json_validity_rate,
        "schema_compliance_rate": report.schema_compliance_rate,
        "macro_field_accuracy": report.macro_field_accuracy,
        "aggregate_exact_match": report.aggregate_exact_match,
        "hallucination_rate": report.hallucination_rate,
        "per_field_accuracy": report.per_field_accuracy,
    }


def compute_composite(metrics: dict) -> float:
    """Compute composite score for autoTune DAG ranking.

    Formula from research report:
    composite = 0.6 * macro_field_accuracy + 0.3 * aggregate_exact_match
                + 0.1 * (1 - hallucination_rate)
    """
    return round(
        0.6 * metrics.get("macro_field_accuracy", 0.0)
        + 0.3 * metrics.get("aggregate_exact_match", 0.0)
        + 0.1 * (1.0 - metrics.get("hallucination_rate", 1.0)),
        4,
    )


def main() -> None:
    logger.info("=== autoTune Extraction Training ===")
    logger.info("Node: %s | Move: %s", NODE_ID, MOVE_TYPE)

    theta = load_theta()
    logger.info("Theta: %s", theta)

    start = time.time()
    status = "completed"
    metrics: dict = {}
    error_msg = None

    try:
        train_loss, val_loss = run_training(theta)
        metrics["task_loss"] = round(val_loss, 4)
        metrics["train_loss"] = round(train_loss, 4)
        logger.info("Training complete: train_loss=%.4f val_loss=%.4f", train_loss, val_loss)

        eval_metrics = run_evaluation(theta)
        metrics.update(eval_metrics)
    except Exception as e:
        status = "crash"
        error_msg = str(e)
        logger.error("Training failed: %s", e)

    elapsed = time.time() - start
    composite = compute_composite(metrics) if status == "completed" else 0.0

    # Determine task_score for autoTune (maps to DAG node.task_score)
    task_score = metrics.get("macro_field_accuracy", 0.0)

    result = {
        "status": status,
        "metrics": {
            "task_score": task_score,
            "task_loss": metrics.get("task_loss", float("inf")),
            **{k: v for k, v in metrics.items() if k not in ("task_score", "task_loss", "per_field_accuracy")},
        },
        "composite": composite,
        "elapsed_seconds": round(elapsed, 1),
        "peak_vram_mb": None,  # Not available on Apple Silicon via this method
        "checkpoint_saved": Path(CHECKPOINT_DIR).exists()
        and any(Path(CHECKPOINT_DIR).iterdir()),
    }

    if error_msg:
        result["error"] = error_msg

    # Per-field accuracy as supplemental data (not in main metrics)
    if "per_field_accuracy" in metrics:
        result["per_field_accuracy"] = metrics["per_field_accuracy"]

    Path("result.json").write_text(json.dumps(result, indent=2))
    logger.info("Wrote result.json: composite=%.4f status=%s", composite, status)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add training script with autoTune contract compliance"
```

---

## Phase 4: autoTune Integration

### Task 7: Wire Theta Passing from Orchestrator to Training Script

**Files:**
- Modify: `src/autotune/orchestrator.py` (add spec.json to worktree)

The orchestrator saves `spec.json` to the archive directory but does not copy it to the worktree where `train.py` runs. Add a single line to copy the spec into the worktree so `train.py` can read theta values.

- [ ] **Step 1: Add spec copy to _launch in orchestrator.py**

After the line `(archive_node / "spec.json").write_text(json.dumps(clean_spec, indent=2))` (around line 300), and after the worktree is created (after line 314), add:

```python
        # Copy spec to worktree so training script can read theta
        shutil.copy2(archive_node / "spec.json", wt_path / "spec.json")
```

This goes right after the worktree creation try/except block, before the checkpoint staging section.

- [ ] **Step 2: Run autoTune tests to verify no regression**

```bash
cd autoTune
uv run pytest tests/test_orchestrator.py -v
```

Expected: All tests pass.

- [ ] **Step 3: Commit (in autoTune repo)**

```bash
cd autoTune
git add src/autotune/orchestrator.py
git commit -m "feat: copy spec.json to worktree for training script theta access"
```

---

### Task 8: Initialize autoTune Overlay & Configure

**Files:**
- Modified by `autotune init`: `autotune/config.yaml`, `autotune/program.md`, etc.

- [ ] **Step 1: Initialize autoTune overlay**

```bash
cd extraction-finetune
autotune init --task instruction_tuning --base-model "mlx-community/Qwen2.5-7B-Instruct-4bit"
```

Expected: Creates `autotune/` directory with config.yaml, program.md, learnings.md, .gitignore, and orchestrator subdirs.

- [ ] **Step 2: Customize config.yaml for extraction task**

Edit `autotune/config.yaml`:

```yaml
project:
  name: "extraction-finetune"
  base_model: "mlx-community/Qwen2.5-7B-Instruct-4bit"

task:
  type: "instruction_tuning"

run:
  script: "train.py"
  checkpoint_format: "safetensors"

retention:
  enabled: false
  script: null
  probe_set: null

parameters:
  lr:
    type: float
    range: [1.0e-5, 1.0e-3]
    log_scale: true
  rank:
    type: int
    range: [4, 64]
  alpha:
    type: int
    range: [8, 128]
  dropout:
    type: float
    range: [0.0, 0.3]
  warmup_ratio:
    type: float
    range: [0.0, 0.15]

merge:
  default_strategy: "weighted_avg"
  custom_hook: null

orchestrator:
  poll_interval: 15
  max_per_gpu: 1
  safety_margin_gb: 2.0
  default_timeout_hours: 2
  checkpoint_retention_layers: 2

scoring:
  exploration_weight: 0.005
  diversity_weight: 0.01
  retention_cliff_threshold: -0.10

files:
  editable:
    - "train.py"
    - "data/"
  readonly: []
```

- [ ] **Step 3: Verify with autotune check**

```bash
autotune check
```

Expected: All checks pass (valid config, directories exist, no orphaned nodes).

- [ ] **Step 4: Commit overlay config**

```bash
git add autotune/config.yaml autotune/program.md autotune/learnings.md
git commit -m "feat: configure autoTune overlay for extraction finetuning"
```

---

### Task 9: Submit Seed Experiment

- [ ] **Step 1: Submit seed node with default theta**

```bash
cd extraction-finetune
autotune submit \
  --desc "Seed: Qwen 2.5 7B 4-bit, LoRA r=16 a=32, lr=2e-4, 200 iters" \
  --theta '{"lr": 2e-4, "rank": 16, "alpha": 32, "dropout": 0.05, "warmup_ratio": 0.06}' \
  --techniques seed_baseline
```

Expected: Creates node_0001 in the DAG and queues it in `autotune/orchestrator/queue/`.

- [ ] **Step 2: Verify DAG status**

```bash
autotune status
```

Expected: Shows 1 proposed node, 0 executed, queue depth 1.

---

## Phase 5: Pipeline Execution

### Task 10: Run Seed Training via Orchestrator

- [ ] **Step 1: Start the orchestrator**

```bash
cd extraction-finetune
autotune orchestrator start
```

This launches the daemon which will pick up the queued seed experiment, create a worktree, and run `train.py`.

- [ ] **Step 2: Monitor training**

```bash
# Check orchestrator status
autotune orchestrator status

# Watch the run log (path from status output)
tail -f autotune/orchestrator/archive/node_0001/run.log
```

- [ ] **Step 3: Wait for completion and reconcile**

After training completes (~20-30 min):

```bash
autotune reconcile
autotune status
```

Expected: node_0001 promoted to "keep" with metrics. DAG shows 1 executed node with composite score.

- [ ] **Step 4: Check results**

```bash
cat autotune/orchestrator/archive/node_0001/result.json
```

Verify: `status: "completed"`, metrics include `macro_field_accuracy`, `aggregate_exact_match`, `composite > 0`.

---

### Task 11: Agent-Driven Iteration (propose → train → reconcile)

This is the autoTune experiment loop. The agent (Claude) uses manifold suggestions to propose next experiments.

- [ ] **Step 1: Ask autoTune for suggestions**

```bash
autotune suggest --n 3
```

Note: With <10 executed nodes, the manifold is unreliable. Suggestions will be based on DAG-UCB scoring and random sampling within parameter bounds.

- [ ] **Step 2: Propose a branch experiment (higher learning rate)**

```bash
autotune propose \
  --parents node_0001 \
  --move branch \
  --desc "Branch: increase lr to 5e-4 to test faster convergence" \
  --theta '{"lr": 5e-4, "rank": 16, "alpha": 32, "dropout": 0.05, "warmup_ratio": 0.06}' \
  --techniques lr_increase

autotune submit --node node_0002
```

- [ ] **Step 3: Propose a branch experiment (higher rank)**

```bash
autotune propose \
  --parents node_0001 \
  --move branch \
  --desc "Branch: increase LoRA rank to 32 for more capacity" \
  --theta '{"lr": 2e-4, "rank": 32, "alpha": 64, "dropout": 0.05, "warmup_ratio": 0.06}' \
  --techniques rank_increase

autotune submit --node node_0003
```

- [ ] **Step 4: Let orchestrator run both experiments**

The orchestrator processes queue sequentially (max_per_gpu=1). Monitor:

```bash
autotune orchestrator status
autotune status
```

- [ ] **Step 5: Reconcile after completion**

```bash
autotune reconcile
autotune status
autotune frontier
```

- [ ] **Step 6: Check Pareto ranking**

```bash
autotune rank --n 5
```

This shows which experiments are Pareto-optimal (rank 0) and which are dominated.

---

### Task 12: Debug and Fix Pipeline Issues

This task is reactive — execute these steps when issues arise during the pipeline.

- [ ] **Step 1: Check for common failure modes**

```bash
# Check orchestrator logs
autotune orchestrator status

# Check individual experiment logs
ls autotune/orchestrator/archive/*/run.log
cat autotune/orchestrator/archive/node_0001/run.log | tail -50

# Check for crashed nodes
autotune status  # look for "crash", "oom", "timeout" status
```

- [ ] **Step 2: Common fixes**

**If mlx-lm import fails**: `pip install mlx-lm --upgrade`

**If model download fails**: Pre-download with `python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-7B-Instruct-4bit')"`

**If training OOMs**: Reduce batch-size to 1 in train.py, or reduce --lora-layers to 8

**If JSON parsing fails in evaluation**: Check that the model generates valid JSON. May need to add a JSON extraction step (find first `{` to last `}`)

**If worktree creation fails**: Check for stale worktrees: `git worktree list` and clean up: `git worktree prune`

**If result.json missing**: Training script crashed before writing it. Check run.log for the error.

- [ ] **Step 3: Restart after fix**

```bash
# If orchestrator died, restart it
autotune orchestrator start

# If a node needs re-running, cancel and re-submit
autotune reconcile  # sync state first
```

---

## Agent Team Definition

Create the following agent team for managing the pipeline:

### Team: `extraction-finetune`

| Agent | Role | Responsibilities |
|-------|------|-----------------|
| `data-prep` | Data Engineer | Schema definition, synthetic data generation, data validation, data augmentation |
| `ml-engineer` | ML Engineer | Training script, evaluation script, mlx-lm integration, model selection |
| `pipeline-ops` | Pipeline Operator | autoTune init/config, experiment submission, orchestrator management, reconciliation |
| `debugger` | Debug Specialist | Log analysis, error diagnosis, fix implementation, pipeline health checks |

**Dispatch rules:**
- Phase 2 (Tasks 3-4): `data-prep` agent
- Phase 3 (Tasks 5-6): `ml-engineer` agent
- Phase 4 (Tasks 7-8): `pipeline-ops` agent (Task 7 modifies autoTune code, needs careful review)
- Phase 5 (Tasks 9-12): `pipeline-ops` for submission/monitoring, `debugger` for issue resolution
- Tasks 3-4 and 5-6 can run in **parallel** (no file overlap)
- Tasks 7-8 are **sequential** (8 depends on autoTune being configured)

---

## Self-Review

**Spec coverage**: All requirements from the research report are addressed — dataset format (ChatML JSONL), 35-field schema (max 1-level nesting, null for missing), evaluation metrics (4-level stack), model choice (Qwen 2.5 7B 4-bit), LoRA config (rank=16, alpha=32), composite formula, result.json contract.

**Placeholder scan**: No TBD/TODO items. All code is complete. All commands include expected output.

**Type consistency**: `PropertyListing` in schema.py produces the same structure consumed by evaluate.py. `result.json` fields match the autoTune contract from `orchestrator.py:_handle_completion`. `theta` keys (lr, rank, alpha, dropout, warmup_ratio) match `config.yaml` parameters section.
