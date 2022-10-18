import DataFrames
import CSV

println("Reading data...")
df = CSV.read("datasets/normalized/A1-top10s.txt", DataFrames.DataFrame, delim='	')
println(df)