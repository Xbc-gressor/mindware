python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 4024 --optimizer smac --evaluation cv
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 4024 --optimizer smac --evaluation holdout
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 4024 --optimizer random --evaluation partial
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 4024 --optimizer random --evaluation partial_bohb

python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 4024 --optimizer smac --evaluation cv
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 4024 --optimizer smac --evaluation holdout
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 4024 --optimizer random --evaluation partial
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 4024 --optimizer random --evaluation partial_bohb