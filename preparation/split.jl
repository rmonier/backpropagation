import DataFrames
import CSV
import Random
import Glob

function splitdf(df, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    Random.shuffle!(ids)
    sel = ids .<= DataFrames.nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

build_folder = "datasets/split/"

println("Looping over the files...")
for dataset_file in Glob.glob("*.txt", "datasets/normalized")
    println("Processing file $dataset_file...")
    folder_name = splitext(basename(dataset_file))[1]

    print("Reading data...")
    df = CSV.read(dataset_file, DataFrames.DataFrame, delim='	')
    println("done.")
    println("Full data:")
    println(df)

    print("Splitting data...")
    train_df, test_df = (if folder_name == "A1-turbine" splitdf(df, 0.85) else splitdf(df, 0.8) end)
    println("done.")

    println("Saving train data...")
    mkpath(build_folder * folder_name)
    CSV.write(build_folder * folder_name * "/train.txt", train_df, delim='	')
    println("done.")
    println("Saving test data...")
    CSV.write(build_folder * folder_name * "/test.txt", test_df, delim='	')
    println("done.")
end