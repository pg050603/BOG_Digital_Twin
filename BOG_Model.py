"""
LH2 Carrier BOG Prediction Module
Author: (your name here)
Date: 2025-11-26

Description
-----------
Thermodynamic simulation of Liquid Hydrogen storage tanks during marine transport.

Features:
- Para-hydrogen EOS via CoolProp
- Lumped-parameter energy balance on tank contents
- Static heat leak via U·A·ΔT
- Sloshing Scaling Factor (SSF) as a function of sea state
- Self-pressurization and venting (BOG) with latent heat approximation
- Excel/CSV voyage profile ingestion and template generator

Assumptions:
- Homogeneous tank contents (no vertical stratification)
- Saturated liquid at initial state (Q = 0.0)
- BOG venting happens as saturated vapor at MAWP
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------------------------------------------------------
# GLOBAL CONSTANTS
# -----------------------------------------------------------------------------
FLUID_NAME = "ParaHydrogen"  # Critical for cryogenic accuracy


# -----------------------------------------------------------------------------
# DATA CLASSES
# -----------------------------------------------------------------------------
@dataclass
class TankSpecs:
    """
    Physical parameters of the Cargo Containment System (CCS).

    Defaults approximate a Suiso Frontier scale pilot tank.
    """
    volume_m3: float = 1250.0          # Geometric volume of tank
    surface_area_m2: float = 560.0     # Wetted / insulated surface area
    insulation_u_value: float = 0.009  # W/m²K – calibrated for ~0.3 %/day BOR
    mawp_pa: float = 5.0e5             # Maximum Allowable Working Pressure (≈5 bar)
    min_pressure_pa: float = 1.05e5    # Lower bound (slightly above atm)


@dataclass
class SimulationConfig:
    """
    High-level simulation configuration.
    """
    initial_fill_ratio: float = 0.98         # 98% full at departure
    initial_pressure_pa: float = 1.10e5      # ~1.1 bar
    time_step_hours: float = 1.0             # Δt in hours


@dataclass
class VoyageProfile:
    """
    Wrapper for the voyage time series (ambient conditions + sea state).

    Required columns:
    - Time_hr          : elapsed time [h]
    - Ambient_Temp_K   : ambient air temperature [K]
    - Sea_State        : Beaufort scale [0–12]
    """
    data: pd.DataFrame

    REQUIRED_COLUMNS = ("Time_hr", "Ambient_Temp_K", "Sea_State")

    @classmethod
    def from_excel(cls, path: str) -> "VoyageProfile":
        df = pd.read_excel(path)
        cls._validate(df)
        return cls(df)

    @classmethod
    def from_csv(cls, path: str) -> "VoyageProfile":
        df = pd.read_csv(path)
        cls._validate(df)
        return cls(df)

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in VoyageProfile.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Voyage profile is missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class LH2State:
    """
    Thermodynamic bulk state of the tank contents (lumped).

    We track:
    - Total mass (kg)
    - Specific internal energy (J/kg)

    Pressure and temperature are derived from (D, U) via CoolProp.
    """
    mass_kg: float
    specific_internal_energy_J_per_kg: float

    def density(self, tank_volume_m3: float) -> float:
        return self.mass_kg / tank_volume_m3

    def pressure(self, tank_volume_m3: float, fluid_name: str = FLUID_NAME) -> float:
        d = self.density(tank_volume_m3)
        return CP.PropsSI("P", "D", d, "U", self.specific_internal_energy_J_per_kg, fluid_name)

    def temperature(self, tank_volume_m3: float, fluid_name: str = FLUID_NAME) -> float:
        d = self.density(tank_volume_m3)
        return CP.PropsSI("T", "D", d, "U", self.specific_internal_energy_J_per_kg, fluid_name)


# -----------------------------------------------------------------------------
# BOG SIMULATOR
# -----------------------------------------------------------------------------
class BOGSimulator:
    """
    Core physics engine: evolves LH2State over a voyage profile and logs BOG.

    Workflow per time step:
    1. Compute static heat leak: Q_static = U * A * (T_amb - T_fluid)
    2. Apply Sloshing Scaling Factor (SSF) based on sea state
    3. Update internal energy (isochoric): U_total += Q_total * dt
    4. Flash with (D, U) -> P, T via EOS
    5. If P > MAWP: compute BOG mass from Q / h_latent, remove energy with h_vap
    """

    def __init__(self, tank: TankSpecs, voyage: VoyageProfile, config: SimulationConfig):
        self.tank = tank
        self.voyage = voyage
        self.config = config

        self.initial_state: Optional[LH2State] = None
        self.results: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------
    def _init_state(self) -> LH2State:
        """
        Initialize LH2 state as saturated liquid at the initial pressure.
        """
        P0 = self.config.initial_pressure_pa
        fill = self.config.initial_fill_ratio

        rho_liq = CP.PropsSI("D", "P", P0, "Q", 0.0, FLUID_NAME)  # kg/m³
        u_liq = CP.PropsSI("U", "P", P0, "Q", 0.0, FLUID_NAME)    # J/kg

        mass0 = rho_liq * self.tank.volume_m3 * fill
        state = LH2State(mass_kg=mass0, specific_internal_energy_J_per_kg=u_liq)

        logging.info(
            f"Initial LH2 mass: {mass0:,.1f} kg at P = {P0 / 1e5:.3f} bar, "
            f"fill ratio = {fill * 100:.1f}%"
        )
        self.initial_state = state
        return state

    # -------------------------------------------------------------------------
    # SLOSHING SCALING FACTOR
    # -------------------------------------------------------------------------
    @staticmethod
    def sloshing_factor(sea_state: int) -> float:
        """
        Map Beaufort sea state to an empirical Sloshing Scaling Factor (SSF).

        Values roughly follow literature trends:
        - Calm to smooth: SSF ≈ 1.0
        - Moderate:       SSF ≈ 1.15–1.4
        - Rough to high:  SSF ≈ 1.8–2.2
        """
        if sea_state <= 2:
            return 1.0
        if sea_state <= 4:
            return 1.15
        if sea_state <= 6:
            return 1.40
        if sea_state <= 8:
            return 1.80
        return 2.20  # Extreme seas

    # -------------------------------------------------------------------------
    # MAIN SOLVER
    # -------------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        Run the BOG simulation over the voyage profile.

        Returns
        -------
        pandas.DataFrame
            Time series with tank conditions and BOG statistics.
        """
        if self.initial_state is None:
            state = self._init_state()
        else:
            state = self.initial_state

        dt_s = self.config.time_step_hours * 3600.0  # step [s]

        # Total internal energy [J]
        U_total = state.specific_internal_energy_J_per_kg * state.mass_kg

        initial_mass = state.mass_kg
        cumulative_bog_kg = 0.0

        records: List[dict] = []

        for step, row in enumerate(self.voyage.data.itertuples(index=False)):
            time_hr = float(row.Time_hr)
            t_amb_K = float(row.Ambient_Temp_K)
            sea_state = int(row.Sea_State)

            # --- Current P, T from EOS (D, U) ---
            try:
                P_curr = state.pressure(self.tank.volume_m3)
                T_curr = state.temperature(self.tank.volume_m3)
            except ValueError as e:
                logging.warning(
                    f"CoolProp state evaluation failed at step {step}: {e}. "
                    "Falling back to saturated liquid at initial pressure."
                )
                P_curr = self.config.initial_pressure_pa
                T_curr = CP.PropsSI("T", "P", P_curr, "Q", 0.0, FLUID_NAME)

            # --- Static heat leak through insulation (no sloshing yet) ---
            q_static_W = (
                self.tank.insulation_u_value
                * self.tank.surface_area_m2
                * (t_amb_K - T_curr)
            )

            # --- Sloshing heat load multiplier ---
            ssf = self.sloshing_factor(sea_state)
            q_total_W = q_static_W * ssf

            # --- Integrate energy (isochoric) ---
            energy_input_J = q_total_W * dt_s
            U_total += energy_input_J

            # Trial state (before venting)
            u_trial = U_total / state.mass_kg
            d_trial = state.mass_kg / self.tank.volume_m3

            try:
                P_trial = CP.PropsSI("P", "D", d_trial, "U", u_trial, FLUID_NAME)
                T_trial = CP.PropsSI("T", "D", d_trial, "U", u_trial, FLUID_NAME)
            except ValueError as e:
                logging.warning(
                    f"CoolProp DU flash failed at step {step}: {e}. "
                    "Forcing pressure above MAWP to trigger venting."
                )
                P_trial = self.tank.mawp_pa * 2.0
                T_trial = T_curr

            # --- Venting (BOG) logic ---
            bog_step_kg = 0.0
            mode = "Closed"

            if P_trial > self.tank.mawp_pa:
                mode = "Venting"

                # Only vent if there is net heating; if q_total_W < 0, skip BOG
                if q_total_W > 0.0:
                    # Latent heat at MAWP (liquid → vapor)
                    h_liq = CP.PropsSI("H", "P", self.tank.mawp_pa, "Q", 0.0, FLUID_NAME)
                    h_vap = CP.PropsSI("H", "P", self.tank.mawp_pa, "Q", 1.0, FLUID_NAME)
                    h_lat = h_vap - h_liq  # J/kg

                    if h_lat <= 0.0:
                        logging.warning(
                            "Non-positive latent heat encountered; "
                            "skipping BOG mass calculation for this step."
                        )
                    else:
                        # Approximate constant-pressure BOG generation:
                        # m_dot ≈ Q_total / h_latent
                        bog_rate_kg_per_s = q_total_W / h_lat
                        bog_step_kg = bog_rate_kg_per_s * dt_s

                        # Safety clamp: don't remove more than 20% of current mass per step
                        bog_step_kg = max(0.0, min(bog_step_kg, 0.2 * state.mass_kg))

                        # Energy carried out by vented vapor (enthalpy transport)
                        energy_removed_J = bog_step_kg * h_vap

                        U_total -= energy_removed_J
                        state.mass_kg -= bog_step_kg
                        cumulative_bog_kg += bog_step_kg

                        # Update specific internal energy after venting
                        state.specific_internal_energy_J_per_kg = U_total / state.mass_kg

                        # Re-flash new state at MAWP (approximately)
                        d_new = state.mass_kg / self.tank.volume_m3
                        try:
                            P_trial = CP.PropsSI(
                                "P", "D", d_new, "U",
                                state.specific_internal_energy_J_per_kg,
                                FLUID_NAME,
                            )
                            T_trial = CP.PropsSI(
                                "T", "D", d_new, "U",
                                state.specific_internal_energy_J_per_kg,
                                FLUID_NAME,
                            )
                        except ValueError as e:
                            logging.warning(
                                f"Post-vent flash failed at step {step}: {e}. "
                                "Clamping pressure to MAWP."
                            )
                            P_trial = self.tank.mawp_pa
                            # Keep previous T_trial as a rough approximation
                else:
                    # No positive heat input; treat as closed for this step
                    mode = "Closed"
                    state.specific_internal_energy_J_per_kg = u_trial
            else:
                # Pressure within allowed range; accept trial state
                state.specific_internal_energy_J_per_kg = u_trial

            # Ensure pressure does not drop below minimum limit
            if P_trial < self.tank.min_pressure_pa:
                P_trial = self.tank.min_pressure_pa

            # Update derived metrics
            bor_step_pct = (bog_step_kg / initial_mass) * 100.0
            bor_cum_pct = (cumulative_bog_kg / initial_mass) * 100.0

            records.append(
                {
                    "Step": step,
                    "Time_hr": time_hr,
                    "Ambient_Temp_K": t_amb_K,
                    "Sea_State": sea_state,
                    "SSF": ssf,
                    "Mode": mode,
                    "Heat_Leak_W": q_total_W,
                    "Tank_Temperature_K": T_trial,
                    "Tank_Pressure_bar": P_trial / 1.0e5,
                    "Mass_LH2_kg": state.mass_kg,
                    "BOG_kg_step": bog_step_kg,
                    "Cum_BOG_kg": cumulative_bog_kg,
                    "BOR_step_pct_of_initial": bor_step_pct,
                    "BOR_cum_pct_of_initial": bor_cum_pct,
                }
            )

        self.results = pd.DataFrame.from_records(records)
        logging.info(
            "Simulation complete. Final cumulative BOR = "
            f"{self.results['BOR_cum_pct_of_initial'].iloc[-1]:.3f}% of initial mass."
        )
        return self.results


# -----------------------------------------------------------------------------
# VOYAGE TEMPLATE GENERATOR
# -----------------------------------------------------------------------------
def generate_voyage_template(
    filepath: str = "LH2_Voyage_Input.xlsx",
    days: float = 16.0,
    time_step_hours: float = 1.0,
) -> pd.DataFrame:
    """
    Generate a simple Australia → Equator → Japan voyage profile.

    - Time_hr: 0 → days*24
    - Ambient_Temp_K: 25°C → 30°C → 5°C (piecewise linear)
    - Sea_State: random 2–6 (light breeze to strong breeze)

    Returns
    -------
    pandas.DataFrame
        Generated profile (also written to Excel).
    """
    n_steps = int(days * 24 / time_step_hours) + 1
    time_hr = np.arange(0.0, n_steps * time_step_hours, time_step_hours)

    # Ensure length consistency
    if len(time_hr) > n_steps:
        time_hr = time_hr[:n_steps]

    # Temperature profile (piecewise)
    T_aus_K = 298.15  # 25°C
    T_eq_K = 303.15   # 30°C
    T_jp_K = 278.15   # 5°C

    n1 = n_steps // 3
    n2 = n_steps // 3
    n3 = n_steps - n1 - n2

    temps1 = np.linspace(T_aus_K, T_eq_K, n1, endpoint=False)
    temps2 = np.linspace(T_eq_K, T_jp_K, n2, endpoint=False)
    temps3 = np.full(n3, T_jp_K)

    ambient_temp_K = np.concatenate([temps1, temps2, temps3])

    # Sea states (simple random draw)
    sea_states = np.random.randint(2, 7, size=n_steps)  # Beaufort 2–6

    df = pd.DataFrame(
        {
            "Time_hr": time_hr,
            "Ambient_Temp_K": ambient_temp_K,
            "Sea_State": sea_states,
        }
    )

    df.to_excel(filepath, index=False)
    logging.info(f"Voyage template written to: {filepath}")
    return df


# -----------------------------------------------------------------------------
# EXAMPLE USAGE (can be removed in production)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate a synthetic voyage profile (16 days Australia ↔ Japan)
    voyage_df = generate_voyage_template("LH2_Voyage_Input.xlsx", days=16.0, time_step_hours=1.0)
    voyage = VoyageProfile(voyage_df)

    # 2. Define tank and simulation configuration
    tank = TankSpecs()
    config = SimulationConfig(
        initial_fill_ratio=0.98,
        initial_pressure_pa=1.10e5,
        time_step_hours=1.0,
    )

    # 3. Run the simulation
    simulator = BOGSimulator(tank, voyage, config)
    results = simulator.run()

    # 4. Quick sanity check
    print(results.head())
    print(
        "\nFinal cumulative BOR (% of initial mass): "
        f"{results['BOR_cum_pct_of_initial'].iloc[-1]:.3f}%"
    )
