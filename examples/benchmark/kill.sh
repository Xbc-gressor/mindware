kill -9 $(ps aux | grep "_benchmark.py" | grep -v "grep" |tr -s " "| cut -d " " -f 2)
kill -9 $(ps aux | grep "multipro" | grep -v "grep" |tr -s " "| cut -d " " -f 2)