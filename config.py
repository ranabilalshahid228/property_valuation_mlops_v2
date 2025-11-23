from dataclasses import dataclass, field

@dataclass
class AppConfig:
    random_state: int = 42
    target: str = "price"
    id_col: str = "property_id"
    categorical_cols: list = field(default_factory=lambda: ["location","property_type","garage","age_category","size_category"])
    numeric_cols: list = field(default_factory=lambda: ["bedrooms","bathrooms","sqft","lot_size","age",
                                                        "price_per_sqft","bed_bath_ratio","sqft_per_bedroom","lot_sqft_ratio"])
    locations: list = field(default_factory=lambda: ["Downtown","Suburbs","Waterfront","Historic","New Development"])
    property_types: list = field(default_factory=lambda: ["Single Family","Condo","Townhouse","Multi Family"])
    garages: list = field(default_factory=lambda: ["None","Carport","Attached","Detached"])

    # Monitoring thresholds
    psi_drift_threshold: float = 0.2
    perf_min_r2: float = 0.80

CONFIG = AppConfig()
