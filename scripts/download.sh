#!/usr/bin/bash 

## check argument length
if [[ $# -lt 1 ]]
then
	echo "Error: Invalid number of options: Please specify the data which should be downloaded."
	echo "Usage: bash scripts/download.sh <DATA_FOR_DOWNLOAD>"
	exit 0
fi

case "$1" in
"data")
    echo "Downloading data..."
    wget http://qa.mpi-inf.mpg.de/reqap/data.zip
    unzip data.zip -d .
    rm data.zip
    echo "Successfully downloaded data!"
    ;;
*)
    echo "Error: Invalid specification of the data. Data $1 could not be found."
	exit 0
    ;;
esac