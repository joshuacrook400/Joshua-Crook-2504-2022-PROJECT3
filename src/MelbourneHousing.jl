using DataFrames, CSV
csv_file = CSV.File("data/Melbourne_housing_FULL.csv"; missingstring = ["NA", "", " ","#N/A"])
df = DataFrame(csv_file)




