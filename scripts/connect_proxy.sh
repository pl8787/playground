#! /bin/bash
NOT_CONNECT=$1

if [ $NOT_CONNECT ]; then
    export http_proxy=""
    export https_proxy=""
    echo "Disconnect Proxy."
else
    export http_proxy=http://10.61.3.150:8088
    export https_proxy=http://10.61.3.150:8088
    echo "Connect Proxy."
fi
