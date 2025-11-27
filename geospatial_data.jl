using DataFrames, XLSX, Dates, Distances, LinearAlgebra
using HTTP, JSON


const R_EARTH_KM = 6371.0

# --- Port Coordinates (Lat, Lon) ---
const PORT_HASTINGS = (-38.3, 145.2)  # Australia
const PORT_KOBE     = (34.6, 135.2)   # Japan


# --- Climatology Model for Wave Height (Hs) ---
# Approximates roughness based on latitude zones


# Cache to prevent hitting the API too hard if restarting often
const WAVE_CACHE = Dict{String, Float64}()

function get_real_wave_height(lat::Float64, lon::Float64, date_time::DateTime)
    date_str = Dates.format(date_time, "yyyy-mm-dd")
    hour = Dates.hour(date_time)
    
    key = "$lat,$lon,$date_str,$hour"
    if haskey(WAVE_CACHE, key)
        return WAVE_CACHE[key]
    end

    url = "https://marine-api.open-meteo.com/v1/marine?latitude=$lat&longitude=$lon&start_date=$date_str&end_date=$date_str&hourly=wave_height&timezone=UTC"

    try
        # FIX 1: Add readtimeout so it doesn't hang forever
        r = HTTP.request("GET", url; readtimeout=5) 
        data = JSON.parse(String(r.body))
        
        raw_val = data["hourly"]["wave_height"][hour + 1]
        
        if raw_val === nothing
            hs = 0.5 
        else
            hs = Float64(raw_val)
        end
        
        WAVE_CACHE[key] = hs
        
        # FIX 2: Sleep a bit longer to be nice to the API
        sleep(0.2) 
        return hs
    catch e
        # If it times out or fails, print a small "x" and return default
        print("x") 
        return 2.5 
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


# --- The "Real" LNG Route Waypoints ---
# 1. Hastings (Start)
# 2. Tasman Sea (Offshore Sydney/Brisbane)
# 3. Jomard Entrance (The critical gap between PNG and Coral Sea)
# 4. West of Guam (Open Ocean)
# 5. Kobe (End)
const ROUTE_WAYPOINTS = [
    (-38.3, 145.2), # Hastings
    (-39.0, 147.0), # Exit Bass Strait (Turn East)
    (-25.0, 156.0), # Coral Sea (Deep water east of Great Barrier Reef)
    (-11.6, 152.0), # Approach Jomard Passage
    (10.0, 140.0),  # Philippine Sea (Open Ocean)
    (34.6, 135.2)   # Kobe
]

# --- Helper: Distance between two points ---
function get_dist_km(p1, p2)
    lat1, lon1 = deg2rad.(p1)
    lat2, lon2 = deg2rad.(p2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2
    c = 2 * atan(sqrt(a), sqrt(1-a))
    return R_EARTH_KM * c
end

# --- Helper: Interpolate between two points ---
function interpolate_pos(p1, p2, fraction)
    lat = p1[1] + fraction * (p2[1] - p1[1])
    lon = p1[2] + fraction * (p2[2] - p1[2])
    return (lat, lon)
end

function generate_voyage(speed_knots::Float64, timestep_hr::Float64, start_date::DateTime)
    speed_km_hr = speed_knots * 1.852
    
    # Calculate segments (Same as before)
    segment_dists = Float64[]
    total_dist_km = 0.0
    for i in 1:(length(ROUTE_WAYPOINTS)-1)
        d = get_dist_km(ROUTE_WAYPOINTS[i], ROUTE_WAYPOINTS[i+1])
        push!(segment_dists, d)
        total_dist_km += d
    end
    
    total_time_hr = total_dist_km / speed_km_hr
    steps = floor(Int, total_time_hr / timestep_hr)
    
    println("Voyage Calculation (Robust Mode):")
    println("  Distance: $(round(total_dist_km, digits=1)) km")
    println("  Steps to process: $steps")
    
    # Init Vectors
    times, lats, lons, amb_temps, sea_states, wave_heights, speeds = Float64[], Float64[], Float64[], Float64[], Int[], Float64[], Float64[]
    
    for i in 0:steps
        t_hr = i * timestep_hr
        dist_traveled = t_hr * speed_km_hr
        
        # Segment Logic
        current_segment_idx = 1
        dist_in_segment = dist_traveled
        for (idx, seg_len) in enumerate(segment_dists)
            if dist_in_segment <= seg_len
                current_segment_idx = idx
                break
            else
                dist_in_segment -= seg_len
                current_segment_idx = idx + 1 
            end
        end
        if current_segment_idx >= length(ROUTE_WAYPOINTS)
            current_segment_idx = length(ROUTE_WAYPOINTS) - 1
            dist_in_segment = segment_dists[end]
        end

        p1 = ROUTE_WAYPOINTS[current_segment_idx]
        p2 = ROUTE_WAYPOINTS[current_segment_idx+1]
        fraction = dist_in_segment / segment_dists[current_segment_idx]
        curr_lat, curr_lon = interpolate_pos(p1, p2, fraction)
        
        # API Call
        current_real_time = start_date + Second(round(Int, t_hr * 3600))
        hs = get_real_wave_height(curr_lat, curr_lon, current_real_time)
        bf = hs_to_beaufort(hs)
        temp_c = 20 + 10 * cos(deg2rad(curr_lat - 0)) 
        
        # Push Data
        push!(times, t_hr)
        push!(lats, curr_lat)
        push!(lons, curr_lon)
        push!(amb_temps, temp_c + 273.15)
        push!(sea_states, bf)
        push!(wave_heights, hs)
        push!(speeds, speed_knots * (1.0 + 0.05 * (rand() - 0.5)))
        
        # FIX 3: Better Progress Bar & Flush
        if i % 24 == 0
            # Print exact progress: Day X - Lat/Lon
            print("\rProcessing Day $(round(t_hr/24, digits=1)): Loc $(round(curr_lat, digits=1)), $(round(curr_lon, digits=1))   ")
            
            # FORCE the print to appear
            flush(stdout) 
        end
    end
    println("\nDone.")
    
    return DataFrame(Time_hr=times, Latitude=lats, Longitude=lons, Ambient_Temp_K=amb_temps, Sea_State=sea_states, Sig_Wave_Height_m=wave_heights, Ship_Speed=speeds)
end

# --- Execution ---

# 1. Define the output filename
filename = "LH2_Voyage_Geospatial.xlsx"

# 2. Run the simulation (Starting Jan 1st, 2024 for realistic winter waves)
#    Speed: 16 knots, Time Step: 1 hour
df_voyage = generate_voyage(16.0, 1.0, DateTime(2024, 1, 1, 8, 0, 0))

# 3. Save to Excel
#    We check if the file exists to avoid permission errors if it's open in Excel
try
    XLSX.writetable(filename, df_voyage)
    println("Success! Data saved to: $filename")
catch e
    println("ERROR: Could not save file. Is '$filename' open in Excel? Close it and try again.")
end