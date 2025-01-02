#!/usr/bin/env bash
kill -9 $(ps aux | grep "python test/*" | grep -v "grep" |tr -s " "| cut -d " " -f 2)
kill -9 $(ps aux | grep "create_algorithm_meta_info_new.py" | grep -v "grep" |tr -s " "| cut -d " " -f 2)
