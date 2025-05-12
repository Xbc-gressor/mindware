kill -9 $(ps aux | grep "kaggle" | grep -v "grep" |tr -s " "| cut -d " " -f 2)
kill -9 $(ps aux | grep "multipro" | grep -v "grep" |tr -s " "| cut -d " " -f 2)