#!/bin/sh

RUN_CMAKE=false
RUN_MAKE=false
RUN_MAKE_TESTS=false

usage() {
    echo "USAGE: source setup.sh [options]"
    echo "OPTIONS:"
    echo "\t--help                 -h: Display help"
    echo "\t--cmake                -c: Run CMake"
    echo "\t--make                 -m: Run make"
    echo "\t--make-tests           -t: Make tests"
    echo "\t--all                  -a: Run all the settings"
}

# Parse through the arguments and check if any relavant flag exists
while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        -c | --cmake)
            RUN_CMAKE=true
            ;;
        -m | --make)
            RUN_MAKE=true
            ;;
        -t | --make-tests)
            RUN_MAKE_TESTS=true
            ;;
        -a | --all)
            RUN_CMAKE=true
            RUN_MAKE=true
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            return 1
            ;;
    esac
    shift
done

# If build does not exist create one
mkdir -p build
cd build

if $RUN_CMAKE
then
    if $RUN_MAKE_TESTS
    then
    	echo "running cmake..."
    	cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTS=ON .. || (cd ../ && return 1)
    else
	echo "running cmake..."
	cmake -DCMAKE_EXPORT_COMPILE_COMMNADS=ON -DBUILD_TESTS=OFF .. || (cd ../ && return 1)
    fi
fi

if $RUN_MAKE
then
    echo "running make..."
    make -j4 || (cd ../ && return 1)
    echo "make complete!"
fi

cd ..
