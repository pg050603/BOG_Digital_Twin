"""
LH2 Carrier BOG Prediction Module (advanced physics version)
Author: (your name here)
Date: 2025-11-26

Description
-----------
Thermodynamic simulation of Liquid Hydrogen storage tanks during marine transport.

Features:
- Para-hydrogen EOS via CoolProp (Helmholtz EoS)
- Lumped-parameter energy balance on tank contents (D,U -> P,T)
- Static heat leak with optional detailed insulation model:
    * Multilayer insulation (MLI) radiation
    * Structural conduction through supports
    * Residual gas conduction depending on vacuum quality
- Sloshing Scaling Factor (SSF) as a function of:
    * Sea state (Beaufort -> significant wave height)
    * Ship speed
    * Fill ratio (mid-fill sloshing amplification)
- Self-pressurization and venting (BOG) with latent heat approximation
- Optional vacuum degradation over time (simulating loss of vacuum quality)
- Excel/CSV voyage profile ingestion and template generator

Assumptions:
- Homogeneous bulk state for LH2 (stratification captured indirectly via correlations)
- Saturated liquid at initial state (Q = 0.0)
- BOG venting happens as saturated vapor at MAWP
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

# --------------------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# --------------------------------------------------------------------------
# GLOBAL CONSTANTS
# --------------------------------------------------------------------------
FLUID_NAME = "ParaHydrogen"  # Critical for cryogenic accuracy
STEFAN_BOLTZMANN = 5.670374419e-8  # W/m²K⁴


# --------------------------------------------------------------------------
# DATA CLASSES
# --------------------------------------------------------------------------
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

    # ------------------------- Advanced insulation model -------------------
    # If True, override simple U*A*ΔT with detailed radiation + conduction
    use_detailed_insulation: bool = False

    # MLI radiation parameters (very approximate defaults – tune per design)
    mli_layers: int = 40
    emissivity_hot: float = 0.05       # outer (warm) surface effective ε
    emissivity_cold: float = 0.03      # inner (cold) tank surface effective ε
    emissivity_shield: float = 0.03    # MLI shield emissivity

    # Structural conduction (e.g. GRP/steel support struts)
    structural_k_W_mK: float = 0.3     # effective conductivity of supports
    structural_area_m2: float = 10.0   # total cross-sectional area of supports
    structural_length_m: float = 0.5   # effective conduction path length

    # Residual gas conduction in the vacuum annulus
    vacuum_gap_thickness_m: float = 0.3
    vacuum_pressure_Pa: float = 1e-3   # hard vacuum by default
    residual_gas_k_W_mK_at_1atm: float = 0.02  # approximate (air-like)


@dataclass
class SimulationConfig:
    """
    High-level simulation configuration.
    """
    initial_fill_ratio: float = 0.98         # 98% full at departure
    initial_pressure_pa: float = 1.10e5      # ~1.1 bar
    time_step_hours: float = 1.0             # Δt in hours

    # Default ship speed (used if voyage profile has no Ship_Speed column)
    design_speed_knots: float = 15.0

    # Optional vacuum degradation model (U-value growth over time)
    # e.g. tau=2000 hr (~3 months), max_u_multiplier=5 → U(t) rises from
    # U0 to 5*U0 asymptotically
    vacuum_degradation_time_constant_hr: Optional[float] = None
    max_u_multiplier: float = 5.0


@dataclass
class VoyageProfile:
    """
    Wrapper for the voyage time series (ambient conditions + sea state).

    Required columns:
    - Time_hr          : elapsed time [h]
    - Ambient_Temp_K   : ambient air temperature [K]
    - Sea_State        : Beaufort scale [0–12]

    Optional column (recommended):
    - Ship_Speed       : ship speed [knots]
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


# --------------------------------------------------------------------------
# BOG SIMULATOR WITH ENHANCED PHYSICS
# --------------------------------------------------------------------------
class BOGSimulator:
    """
    Core physics engine: evolves LH2State over a voyage profile and logs BOG.

    Workflow per time step:
    1. Compute static heat leak:
         - Either simple U*A*(T_amb - T_fluid)
         - Or detailed MLI + structural + gas conduction
       (with optional vacuum degradation)
    2. Apply Sloshing Scaling Factor (SSF) based on sea state, ship speed,
       and fill ratio.
    3. Update internal energy (isochoric): U_total += Q_total * dt
    4. Flash with (D, U) -> P, T via EOS (self-pressurization)
    5. If P > MAWP: compute BOG mass from Q/h_latent, remove energy with h_vap
    """

    def __init__(self, tank: TankSpecs, voyage: VoyageProfile, config: SimulationConfig):
        self.tank = tank
        self.voyage = voyage
        self.config = config

        self.initial_state: Optional[LH2State] = None
        self.results: Optional[pd.DataFrame] = None

    # ----------------------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # INSULATION / HEAT LEAK MODEL
    # ----------------------------------------------------------------------
    def _effective_u_value(self, time_hr: float) -> float:
        """
        Effective U-value when NOT using detailed insulation.
        Allows modelling of vacuum degradation as a gradual
        increase in U over time.
        """
        U0 = self.tank.insulation_u_value

        if self.config.vacuum_degradation_time_constant_hr is None:
            return U0

        tau = self.config.vacuum_degradation_time_constant_hr
        factor = 1.0 + (self.config.max_u_multiplier - 1.0) * (1.0 - math.exp(-time_hr / tau))
        return U0 * factor

    def _mli_radiation_W(self, T_hot_K: float, T_cold_K: float) -> float:
        """
        Radiative heat leak through multilayer insulation using a
        simplified Stefan-Boltzmann effective emissivity model.
        """
        N = max(self.tank.mli_layers, 1)
        eps_h = max(self.tank.emissivity_hot, 1e-4)
        eps_c = max(self.tank.emissivity_cold, 1e-4)
        eps_s = max(self.tank.emissivity_shield, 1e-4)

        denom = (1.0 / eps_h) + (1.0 / eps_c) - 1.0 + N * (2.0 / eps_s - 1.0)
        if denom <= 0.0:
            return 0.0

        q_rad_per_area = STEFAN_BOLTZMANN * (T_hot_K**4 - T_cold_K**4) / denom
        return q_rad_per_area * self.tank.surface_area_m2

    def _structural_conduction_W(self, T_hot_K: float, T_cold_K: float) -> float:
        """
        Conduction through support struts (e.g. GRP/steel).
        """
        k = self.tank.structural_k_W_mK
        L = max(self.tank.structural_length_m, 1e-3)
        A = self.tank.structural_area_m2

        return k * A * (T_hot_K - T_cold_K) / L

    def _residual_gas_conduction_W(self, T_hot_K: float, T_cold_K: float) -> float:
        """
        Approximate conduction through residual gas in the vacuum annulus.
        Gas conductivity scales roughly with pressure in rarefied regimes.
        """
        p = max(self.tank.vacuum_pressure_Pa, 1e-6)
        p_atm = 101325.0
        k_ref = self.tank.residual_gas_k_W_mK_at_1atm

        # Simple scaling law: k_eff ~ (p/p_atm)^alpha
        alpha = 0.6
        k_eff = k_ref * (p / p_atm) ** alpha

        L_gap = max(self.tank.vacuum_gap_thickness_m, 1e-3)
        A_gap = self.tank.surface_area_m2

        return k_eff * A_gap * (T_hot_K - T_cold_K) / L_gap

    def _static_heat_leak_W(self, T_amb_K: float, T_fluid_K: float, time_hr: float) -> float:
        """
        Compute total static heat leak (no sloshing) into the tank.

        If use_detailed_insulation is True:
            Q = Q_rad_MLI + Q_struct + Q_gas

        Else:
            Q = U_eff * A * ΔT, with U_eff optionally growing over time.
        """
        if self.tank.use_detailed_insulation:
            q_rad = self._mli_radiation_W(T_amb_K, T_fluid_K)
            q_struct = self._structural_conduction_W(T_amb_K, T_fluid_K)
            q_gas = self._residual_gas_conduction_W(T_amb_K, T_fluid_K)
            return q_rad + q_struct + q_gas

        # Simple UA model with optional vacuum degradation
        U_eff = self._effective_u_value(time_hr)
        return U_eff * self.tank.surface_area_m2 * (T_amb_K - T_fluid_K)

    # ----------------------------------------------------------------------
    # SLOSHING SCALING FACTOR
    # ----------------------------------------------------------------------
    @staticmethod
    def _beaufort_to_significant_wave_height_m(sea_state: int) -> float:
        """
        Rough mapping from Beaufort sea state to significant wave height Hs [m].
        (Midpoints of WMO ranges; used for sloshing correlations.)
        """
        mapping = {
            0: 0.05,
            1: 0.3,
            2: 0.9,
            3: 1.9,
            4: 3.3,
            5: 5.0,
            6: 7.5,
            7: 11.5,
            8: 15.0,
            9: 18.0,
            10: 20.0,
            11: 25.0,
            12: 30.0,
        }
        sea_state = max(0, min(12, sea_state))
        return mapping.get(sea_state, 0.05)

    def sloshing_factor(self, sea_state: int, ship_speed_knots: float, fill_ratio: float) -> float:
        """
        Empirical Sloshing Scaling Factor (SSF) that amplifies heat leak.

        Depends on:
        - Sea state (→ significant wave height Hs)
        - Ship speed (more excitation at high speeds)
        - Fill ratio (sloshing strongest near mid-fill)

        Calibrated so:
        - Calm seas ⇒ SSF ≈ 1.0
        - Moderate ⇒ ~1.1–1.3
        - Rough / storms + high speed + mid-fill ⇒ up to ~2.0–2.3
        """
        Hs = self._beaufort_to_significant_wave_height_m(sea_state)
        Hs_ref = 3.0  # ~Beaufort 4

        # Baseline sloshing amplification
        base = 1.0 + 0.30 * (max(Hs, 0.0) / Hs_ref) ** 1.2

        # Ship speed factor (weak power law)
        v = max(ship_speed_knots, 1.0)
        v_ref = self.config.design_speed_knots
        speed_factor = (v / v_ref) ** 0.3

        # Fill ratio factor: Gaussian, peaked at mid-fill
        f = max(0.01, min(0.99, fill_ratio))
        f_peak = 0.5
        f_spread = 0.2
        fill_factor = 1.0 + 0.2 * math.exp(-((f - f_peak) / f_spread) ** 2)

        ssf = base * speed_factor * fill_factor

        # Clamp to reasonable range
        return max(1.0, min(2.5, ssf))

    # ----------------------------------------------------------------------
    # MAIN SOLVER
    # ----------------------------------------------------------------------
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
            ship_speed_knots = getattr(row, "Ship_Speed", self.config.design_speed_knots)

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
            q_static_W = self._static_heat_leak_W(t_amb_K, T_curr, time_hr)

            # --- Fill ratio estimate for sloshing (using saturated liquid density) ---
            try:
                rho_liq_sat = CP.PropsSI("D", "P", max(P_curr, self.tank.min_pressure_pa), "Q", 0.0, FLUID_NAME)
            except ValueError:
                rho_liq_sat = state.mass_kg / self.tank.volume_m3  # crude fallback

            fill_ratio = min(1.0, max(0.0, state.mass_kg / (rho_liq_sat * self.tank.volume_m3)))

            # --- Sloshing heat load multiplier ---
            ssf = self.sloshing_factor(sea_state, ship_speed_knots, fill_ratio)
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

                        # Re-flash new state (approximately)
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
                    "Ship_Speed_kn": ship_speed_knots,
                    "SSF": ssf,
                    "Mode": mode,
                    "Heat_Leak_W": q_total_W,
                    "Tank_Temperature_K": T_trial,
                    "Tank_Pressure_bar": P_trial / 1.0e5,
                    "Fill_Ratio": fill_ratio,
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


# --------------------------------------------------------------------------
# VOYAGE TEMPLATE GENERATOR
# --------------------------------------------------------------------------
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
    - Ship_Speed: fluctuates mildly around design speed
    """
    n_steps = int(days * 24 / time_step_hours) + 1
    time_hr = np.arange(0.0, n_steps * time_step_hours, time_step_hours)

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

    # Ship speed around 15 kn with small noise
    base_speed = 15.0
    ship_speed = base_speed + np.random.normal(loc=0.0, scale=1.0, size=n_steps)
    ship_speed = np.clip(ship_speed, 10.0, 20.0)

    df = pd.DataFrame(
        {
            "Time_hr": time_hr,
            "Ambient_Temp_K": ambient_temp_K,
            "Sea_State": sea_states,
            "Ship_Speed": ship_speed,
        }
    )

    df.to_excel(filepath, index=False)
    logging.info(f"Voyage template written to: {filepath}")
    return df


# --------------------------------------------------------------------------
# EXAMPLE USAGE
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate a synthetic voyage profile (16 days Australia ↔ Japan)
    voyage_df = generate_voyage_template("LH2_Voyage_Input.xlsx", days=16.0, time_step_hours=1.0)
    voyage = VoyageProfile(voyage_df)

    # 2. Define tank and simulation configuration
    tank = TankSpecs(
        # Try detailed insulation model for more realism:
        use_detailed_insulation=True,
    )
    config = SimulationConfig(
        initial_fill_ratio=0.98,
        initial_pressure_pa=1.10e5,
        time_step_hours=1.0,
        design_speed_knots=15.0,
        # Example: simulate slow vacuum degradation over ~2000 hr
        vacuum_degradation_time_constant_hr=None,  # set e.g. 2000.0 to enable
        max_u_multiplier=5.0,
    )

    # 3. Run the simulation
    simulator = BOGSimulator(tank, voyage, config)
    results = simulator.run()

    # 4. Save results to file
    results.to_excel("LH2_BOG_Results.xlsx", index=False)
    print(results.head())
    print("Simulation complete. Results saved to LH2_BOG_Results.xlsx")
