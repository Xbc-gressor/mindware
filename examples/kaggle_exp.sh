python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --per_time_limit 800 --optimizer smac --evaluation cv
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --per_time_limit 600 --optimizer smac --evaluation holdout
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --per_time_limit 1200 --optimizer smac --evaluation partial
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --per_time_limit 1200 --optimizer smac --evaluation partial_bohb

python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --per_time_limit 800 --optimizer mab --evaluation cv
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --per_time_limit 600 --optimizer mab --evaluation holdout
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --per_time_limit 1200 --optimizer mab --evaluation partial
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --per_time_limit 1200 --optimizer mab --evaluation partial_bohb

python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --per_time_limit 800 --optimizer smac --evaluation cv
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --per_time_limit 600 --optimizer smac --evaluation holdout
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --per_time_limit 1200 --optimizer smac --evaluation partial
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --per_time_limit 1200 --optimizer smac --evaluation partial_bohb

python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --per_time_limit 800 --optimizer mab --evaluation cv
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --per_time_limit 600 --optimizer mab --evaluation holdout
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --per_time_limit 1200 --optimizer mab --evaluation partial
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --per_time_limit 1200 --optimizer mab --evaluation partial_bohb
