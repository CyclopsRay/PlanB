# Decompress the gz file in this folder

for file in *.gz;
do
    # split the filename into name and extension
    name=${file%%.gz}
    gzip -dc $file > $name;
    echo $name;
done