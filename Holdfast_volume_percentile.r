library(readxl)

# Load data
df <- read_excel("/Users/elile/Documents/MLML/Thesis/Data/Holdfast Volume Survey.xlsx")

# Compute volume from circumference and length
# Formula matches the sheet: C^2/(12*pi) * sqrt(L^2 - (C/(2*pi))^2)
# Note: L here is slant height, r = C/(2*pi), vertical height = sqrt(L^2 - r^2)
C <- df[["Circumfernce (cm)"]]   # note: column name has typo in source
L <- df[["Length (cm)"]]

df$volume_cm3 <- (C^2 / (12 * pi)) * sqrt(L^2 - (C / (2 * pi))^2)

# Drop rows where volume couldn't be computed
df_clean <- df[!is.na(df$volume_cm3), ]

# --- 50th percentile volume ---
v50 <- quantile(df_clean$volume_cm3, 0.50)
cat(sprintf("50th percentile volume: %.4f cm^3\n\n", v50))

# For reference: summary of the distribution
summary(df_clean$volume_cm3)

# --- Artificial holdfast: equal length and width ---
# "Equal length and width" means slant length L = diameter d = C/pi
# So C = pi*L, and r = C/(2*pi) = L/2
#
# Substituting into the cone volume formula:
#   V = (C^2 / (12*pi)) * sqrt(L^2 - (C/(2*pi))^2)
#   V = ((pi*L)^2 / (12*pi)) * sqrt(L^2 - (L/2)^2)
#   V = (pi*L^2 / 12) * (L*sqrt(3)/2)
#   V = pi*sqrt(3)*L^3 / 24
#
# Solve for L given V = v50:
#   L = (24 * v50 / (pi * sqrt(3)))^(1/3)

L_target <- (24 * v50 / (pi * sqrt(3)))^(1/3)
C_target <- pi * L_target
r_target <- L_target / 2
h_target <- sqrt(L_target^2 - r_target^2)   # vertical height

# Verify
v_check <- (pi * sqrt(3) * L_target^3) / 24

cat(sprintf("\nArtificial holdfast (equal slant length and diameter):\n"))
cat(sprintf("  Slant length (L) : %.2f cm\n", L_target))
cat(sprintf("  Diameter (width) : %.2f cm  [= L, as required]\n", 2 * r_target))
cat(sprintf("  Circumference (C): %.2f cm\n", C_target))
cat(sprintf("  Vertical height  : %.2f cm\n", h_target))
cat(sprintf("  Radius           : %.2f cm\n", r_target))
cat(sprintf("\nVerification — cone volume: %.4f cm^3 (target: %.4f cm^3)\n",
            v_check, v50))

# --- Percentile of a right cone with given radius and vertical height ---
# V = (1/3) * pi * r^2 * h

cone_r <- 10    # base radius (cm)  <-- set your values here
cone_h <- 10    # vertical height (cm)

cone_vol <- (1/3) * pi * cone_r^2 * cone_h

pct <- mean(df_clean$volume_cm3 <= cone_vol) * 100

cat(sprintf("\n--- Percentile lookup for a given right cone ---\n"))
cat(sprintf("  Radius           : %.2f cm\n", cone_r))
cat(sprintf("  Vertical height  : %.2f cm\n", cone_h))
cat(sprintf("  Volume           : %.4f cm^3\n", cone_vol))
cat(sprintf("  Percentile       : %.1fth\n", pct))

# --- Cone dimensions (r = h) for a given percentile ---
# With radius = vertical height (r = h):
#   V = (1/3) * pi * r^2 * h = (1/3) * pi * r^3
#   r = (3 * V / pi)^(1/3)

target_pct <- 0.25    # <-- set desired percentile (0–1)

v_pct <- quantile(df_clean$volume_cm3, target_pct)
r_eq  <- (3 * v_pct / pi)^(1/3)   # radius = vertical height

cat(sprintf("\n--- Cone dimensions at the %.0fth percentile (radius = height) ---\n",
            target_pct * 100))
cat(sprintf("  Volume           : %.4f cm^3\n", v_pct))
cat(sprintf("  Radius           : %.2f cm\n", r_eq))
cat(sprintf("  Vertical height  : %.2f cm  [= radius]\n", r_eq))