#conda env create -n varpop -f ./install/requirements.txt
fname='./install/setup.sh'
contents=`cat $fname`
if grep 'source activate varpops' $fname; then
    echo "MATCH"
fi
if [ ! "grep 'source activate varpop' $fname" ];
    then echo "source activate varpop"
fi
