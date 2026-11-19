"""Illustrative RCDS-to-MetaMap mechanistic simulation.

This is a synthetic, non-clinical model for exploring pathway coupling. It does
not diagnose carcinogen exposure or predict an individual's disease risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SensorReading:
    """Arduino-style optical/electrochemical measurement.

    RGB values are 8-bit readings from the assay channel and a blank/reference
    channel; absorbance is computed from their red-channel intensity ratio.
    A serial port may be supplied to read one CSV line: R,G,B,pH,temp_C.
    """
    rgb: tuple[int, int, int]
    blank_rgb: tuple[int, int, int]
    pH: float
    temperature_C: float

    @classmethod
    def from_serial(cls, port: str, baudrate: int = 9600,
                    timeout: float = 5.0) -> "SensorReading":
        try:
            import serial  # optional dependency for physical Arduino use
        except ImportError as exc:
            raise RuntimeError("Install pyserial to read an Arduino serial port") from exc
        with serial.Serial(port, baudrate=baudrate, timeout=timeout) as connection:
            fields = connection.readline().decode("ascii").strip().split(",")
        if len(fields) != 8:
            raise ValueError("Expected CSV: R,G,B,blank_R,blank_G,blank_B,pH,temp_C")
        values = [float(value) for value in fields]
        return cls(tuple(map(int, values[:3])), tuple(map(int, values[3:6])),
                   values[6], values[7])

    def estimate_concentration_uM(self, epsilon_M_cm: float = 3.8e3,
                                  path_length_cm: float = 1.0,
                                  dilution_factor: float = 1.0) -> tuple[float, float]:
        """Beer-Lambert estimate using red-channel optical density.

        A = -log10(I/I0), c = A/(epsilon*l), with M converted to uM.
        The extinction coefficient is an illustrative assay calibration and
        should be replaced with a validated standard curve for real hardware.
        """
        intensity = max(float(self.rgb[0]), 1.0)
        reference = max(float(self.blank_rgb[0]), 1.0)
        absorbance = max(0.0, -np.log10(intensity / reference))
        concentration = absorbance / (epsilon_M_cm * path_length_cm) * 1e6
        return absorbance, concentration * dilution_factor


def michaelis_menten(substrate: float, vmax: float, km: float,
                      inhibitor: float = 0.0, ki: Optional[float] = None) -> float:
    """Saturable flux with optional competitive inhibition of apparent Km."""
    apparent_km = km * (1.0 + inhibitor / ki) if ki else km
    return vmax * max(substrate, 0.0) / (apparent_km + max(substrate, 0.0))


def pathway_engine(galt_activity: float, carcinogen_uM: float,
                   sensor_pH: float, temperature_C: float,
                   dietary_glucose: float = 5.0,
                   dietary_fructose: float = 2.0,
                   dietary_galactose: float = 1.0,
                   enzyme_efficiencies: Optional[dict[str, float]] = None) -> dict[str, float]:
    """Compute normalized pathway fluxes and stress features (arbitrary units)."""
    galt_activity = float(np.clip(galt_activity, 0.0, 1.0))
    efficiencies = {"GALT": galt_activity, "ALDOB": 1.0, "GALK1": 1.0,
                    "UGP2": 1.0, "PDH": 1.0}
    if enzyme_efficiencies:
        efficiencies.update({key: float(np.clip(value, 0.0, 1.0))
                             for key, value in enzyme_efficiencies.items()
                             if key in efficiencies})
    efficiencies["GALT"] = float(np.clip(galt_activity, 0.0, 1.0))
    # Temperature and pH provide modest assay-context modifiers, not physiology.
    assay_factor = float(np.clip(1.0 - 0.012 * abs(sensor_pH - 7.4)
                                 - 0.004 * abs(temperature_C - 37.0), 0.75, 1.0))

    # Pathway pools are normalized concentrations; enzyme Vmax values are
    # normalized flux capacities. Sequential fluxes expose named enzymes.
    glucose = max(0.0, dietary_glucose)
    fructose = max(0.0, dietary_fructose)
    galactose = max(0.0, dietary_galactose)
    f1p = michaelis_menten(fructose, 1.3, 0.5)  # KHK
    triose_from_fructose = michaelis_menten(
        f1p, efficiencies["ALDOB"], 0.4)  # ALDOB/TKFC
    hk1 = michaelis_menten(glucose, 1.2, 0.7)
    pfk1 = michaelis_menten(hk1, 1.0, 0.5)
    pk = michaelis_menten(pfk1 + triose_from_fructose, 1.1, 0.6)

    galk1 = michaelis_menten(galactose, 0.9 * efficiencies["GALK1"], 0.4)
    galt = michaelis_menten(galk1, 0.9 * galt_activity, 0.3)
    # GALE operates in series with GALT; residual capacity follows the modeled
    # GALT activity for the deficiency scenario.
    gale = michaelis_menten(galt, 0.9 * galt_activity, 0.3)
    gal1p = max(0.0, galk1 - galt)
    galactitol = michaelis_menten(max(0.0, galactose - galk1) + gal1p, 0.65, 0.5)  # AKR1B1
    sord = michaelis_menten(galactitol, 0.55, 0.35)

    pdh = michaelis_menten(pk, 0.95 * efficiencies["PDH"], 0.4)
    tca_entry = michaelis_menten(pdh, 0.9, 0.35)
    acc = michaelis_menten(triose_from_fructose + hk1, 0.5, 0.8)  # ACC
    fasn = michaelis_menten(acc, 0.45, 0.3)  # FASN

    # Gal1P competes at UGP2. UDP-glucose supply supports UDP-glucuronate
    # generation; glucuronidation capacity is then limited by UGT1A1 and donor.
    ugp2 = michaelis_menten(gale, 0.9 * efficiencies["UGP2"], 0.35, inhibitor=gal1p, ki=0.35)  # UGP2
    udp_glucose = ugp2
    udp_glucuronate = michaelis_menten(udp_glucose, 0.85, 0.3)
    ugt1a1 = michaelis_menten(udp_glucuronate, 0.9, 0.3)  # UGT1A1

    # Demand scales with measured nitrosamine-equivalent concentration; clearance
    # capacity is normalized to a reference substrate load.
    carcinogen_demand = max(0.0, carcinogen_uM) / 10.0
    clearance = ugt1a1 * assay_factor
    unmet_fraction = (carcinogen_demand / (carcinogen_demand + clearance + 1e-9)
                      if carcinogen_demand > 0 else 0.0)
    ugt_demand = carcinogen_demand
    dna_adduct_index = carcinogen_demand * unmet_fraction

    # Relative ATP stress responds to fructose phosphorylation and reduced TCA entry.
    atp_depletion = float(np.clip(0.18 * f1p + 0.55 * (1.0 - tca_entry), 0.0, 1.0))
    lipogenic_load = float(np.clip(fasn, 0.0, 1.0))
    # Dimensionless features are clipped for a stable comparative sigma-sum.
    features = {
        "F1P buildup": float(np.clip(f1p / 1.3, 0.0, 1.0)),
        "Gal1P buildup": float(np.clip(gal1p / 0.9, 0.0, 1.0)),
        "Galactitol osmotic load": float(np.clip(galactitol / 0.65, 0.0, 1.0)),
        "ATP depletion": atp_depletion,
        "Lipogenic load": lipogenic_load,
        "Uncleared carcinogen adduct index": float(np.clip(dna_adduct_index / 2.0, 0.0, 1.0)),
    }
    weights = {
        "F1P buildup": 1.0,
        "Gal1P buildup": 1.2,
        "Galactitol osmotic load": 0.8,
        "ATP depletion": 1.0,
        "Lipogenic load": 0.7,
        "Uncleared carcinogen adduct index": 1.5,
    }
    sigma_sum = sum(weights[name] * value for name, value in features.items())
    return {
        **features,
        "HK1 flux": hk1, "PFK1 flux": pfk1, "PK flux": pk,
        "KHK flux": f1p, "ALDOB-TKFC flux": triose_from_fructose,
        "GALK1 flux": galk1, "GALT flux": galt, "GALE flux": gale,
        "AKR1B1 flux": galactitol, "SORD flux": sord,
        "PDH flux": pdh, "TCA entry flux": tca_entry,
        "ACC flux": acc, "FASN flux": fasn,
        "UGP2 flux": ugp2, "UDP-glucose pool": udp_glucose,
        "UDP-glucuronate pool": udp_glucuronate, "UGT1A1 clearance capacity": clearance,
        "UGT detoxification demand (scaled)": ugt_demand,
        "Uncleared DNA adduct risk (scaled)": dna_adduct_index,
        "Sigma-sum stress score": sigma_sum,
    }


def run_scenarios() -> pd.DataFrame:
    scenarios = [
        ("Healthy + Clean Sample", 1.00, 20),
        ("Healthy + High Nitrosamine Sample", 1.00, 180),
        ("GALT 2% Deficiency + Clean Sample", 0.02, 20),
        ("GALT 2% Deficiency + High Nitrosamine Sample", 0.02, 180),
    ]
    rows = []
    for name, galt_activity, target_uM in scenarios:
        # Construct an RGB reading whose absorbance corresponds to the target
        # concentration under the default Beer-Lambert calibration.
        absorbance = target_uM / 1e6 * 3.8e3
        red = int(round(255 * 10 ** (-absorbance)))
        reading = SensorReading((red, red, red), (255, 255, 255), 7.4, 37.0)
        measured_absorbance, concentration = reading.estimate_concentration_uM()
        result = pathway_engine(galt_activity, concentration, reading.pH,
                                reading.temperature_C)
        rows.append({
            "Scenario": name,
            "Assay absorbance": measured_absorbance,
            "Estimated carcinogen (uM)": concentration,
            "GALT activity (% normal)": galt_activity * 100,
            **result,
        })
    return pd.DataFrame(rows).set_index("Scenario").round(4)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("RCDS → MetaMap-ML mechanistic scenario summary (illustrative units)")
    print(run_scenarios().to_string())
