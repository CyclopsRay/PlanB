# Compress the tsv files in the raw_data folder

for file in *.tsv;
do
    # split the filename into name and extension
    gzip $file;
    echo $file;
done