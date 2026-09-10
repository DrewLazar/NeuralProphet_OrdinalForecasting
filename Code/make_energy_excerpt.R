# =====================================================================
# make_energy_excerpt.R
#
# Produces the public excerpt of the Foundational Sciences Building load
# series for the repository.
#
#   1. subsets to September--November 2024
#   2. drops the `load` column, so no kilowatt readings are released
#
# Input : energy_data.csv   (full record, NOT for redistribution)
# Output: energy_data_excerpt.csv
# =====================================================================

IN   <- "C:/Users/dmlazar/Desktop/ordinalcluster/FinalCode/Data/energy_data.csv"
OUT  <- "C:/Users/dmlazar/Desktop/ordinalcluster/FinalCode/Data/energy_data_excerpt.csv"
FROM <- as.POSIXct("2024-09-01 00:00:00", tz = "UTC")
TO   <- as.POSIXct("2024-12-01 00:00:00", tz = "UTC")   # exclusive

d <- read.csv(IN, stringsAsFactors = FALSE)
d$datetime <- as.POSIXct(d$datetime, tz = "UTC", format = "%Y-%m-%d %H:%M:%S")

# 1. subset to the release window
d <- d[d$datetime >= FROM & d$datetime < TO, ]

# 2. drop the continuous readings
d$load <- NULL

# guard: nothing resembling kilowatts may leave
stopifnot(!("load" %in% names(d)))
stopifnot(nrow(d) > 0)

# write datetime back in the same format as the source file
out <- d
out$datetime <- format(out$datetime, "%Y-%m-%d %H:%M:%S")
write.csv(out, OUT, row.names = FALSE, quote = FALSE)

cat(sprintf("wrote %s\n  rows    : %d\n  window  : %s to %s\n  columns : %s\n",
            OUT, nrow(out),
            format(min(d$datetime), "%Y-%m-%d"),
            format(max(d$datetime), "%Y-%m-%d"),
            paste(names(out), collapse = ", ")))

cat("  category shares:",
    paste(sprintf("%d=%.1f%%", sort(unique(out$load_category)),
                  100 * as.numeric(table(out$load_category)) / nrow(out)),
          collapse = "  "), "\n")
