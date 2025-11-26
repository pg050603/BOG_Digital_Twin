using DataFrames
using XLSX
using Dates
using LinearAlgebra

# --- Constants ---
const R_EARTH_KM = 6371.0

# --- Port Coordinates (Lat, Lon) ---
const PORT_HASTINGS = (-38.3, 145.2)  # Australia
const PORT_KOBE     = (34.6, 135.2)   # Japan

# --- Climatology Model for Wave Height (Hs) ---
# Approximates roughness based on latitude zones
function get_regional_hs(lat::Float64)
    # Noise factor for variability
    noise = rand() * 1.0 - 0.5 # +/- 0.5m random variation
    
    if lat < -30
        # Tasman Sea / Southern Ocean influence (Rough)
        return 3.5 + (abs(lat) - 30)*0.1 + noise
    elseif lat < -10
        # Coral Sea (Moderate)
        return 2.0 + noise
    elseif lat < 10
        # Equatorial / Doldrums (Calm)
        return 1.2 + noise
    elseif lat < 25
        # Philippine Sea (Moderate)
        return 2.5 + noise
    else
        # North Pacific approach to Japan (Rougher)
        return 3.0 + noise
    end
end

# Convert Significant Wave Height (m) to Beaufort Scale (Integer)
function hs_to_beaufort(hs::Float64)
    if hs < 0.1 return 0
    elseif hs < 0.3 return 1
    elseif hs < 0.9 return 2
    elseif hs < 1.9 return 3
    elseif hs < 3.3 return 4
    elseif hs < 5.0 return 5
    elseif hs < 7.5 return 6
    elseif hs < 11.5 return 7
    elseif hs < 15.0 return 8
    else return 9
    end
end

# --- Great Circle Interpolation ---
function generate_voyage(speed_knots::Float64, timestep_hr::Float64)
    speed_km_hr = speed_knots * 1.852
    
    # Convert to radians
    lat1, lon1 = deg2rad.(PORT_HASTINGS)
    lat2, lon2 = deg2rad.(PORT_KOBE)
    
    # Haversine distance
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2
    c = 2 * atan(sqrt(a), sqrt(1-a))
    total_dist_km = R_EARTH_KM * c
    
    total_time_hr = total_dist_km / speed_km_hr
    steps = floor(Int, total_time_hr / timestep_hr)
    
    println("Voyage Calculation:")
    println("  Distance: $(round(total_dist_km, digits=1)) km")
    println("  Duration: $(round(total_time_hr/24, digits=1)) days")
    
    # Data vectors
    times = Float64[]
    lats = Float64[]
    lons = Float64[]
    amb_temps = Float64[]
    sea_states = Int[]
    wave_heights = Float64[]
    speeds = Float64[]
    
    # Interpolation loop
    for i in 0:steps
        f = i / steps # fraction of journey
        
        # Intermediate point (Simplified Linear Interpolation for demo)
        # For strict Great Circle, vector algebra is needed, but this suffices for params
        curr_lat_rad = lat1 + f * (lat2 - lat1)
        curr_lon_rad = lon1 + f * (lon2 - lon1)
        
        curr_lat = rad2deg(curr_lat_rad)
        curr_lon = rad2deg(curr_lon_rad)
        
        t_hr = i * timestep_hr
        
        # --- Environment Physics ---
        
        # Temperature: 15C (Vic) -> 30C (Equator) -> 10C (Japan winter)
        # Modeled as a sine wave approximation over latitude
        temp_c = 20 + 10 * cos(deg2rad(curr_lat - 0)) # Hotter near equator
        temp_k = temp_c + 273.15
        
        # Sea State
        hs = get_regional_hs(curr_lat)
        bf = hs_to_beaufort(hs)
        
        # Speed (add operational fluctuation)
        curr_speed = speed_knots * (1.0 + 0.05 * (rand() - 0.5))
        
        push!(times, t_hr)
        push!(lats, curr_lat)
        push!(lons, curr_lon)
        push!(amb_temps, temp_k)
        push!(sea_states, bf)
        push!(wave_heights, hs)
        push!(speeds, curr_speed)
    end
    
    df = DataFrame(
        Time_hr = times,
        Latitude = lats,
        Longitude = lons,
        Ambient_Temp_K = amb_temps,
        Sea_State = sea_states,
        Sig_Wave_Height_m = wave_heights,
        Ship_Speed = speeds
    )
    
    return df
end

# --- Execution ---
df_voyage = generate_voyage(15.0, 1.0) # 15 knots, 1 hour steps

# Export
filename = "LH2_Voyage_Geospatial.xlsx"
XLSX.writetable(filename, df_voyage)
println("Simulated voyage data saved to: $filename")