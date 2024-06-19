# python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --optimizer smac --evaluation cv
# python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --optimizer smac --evaluation holdout
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --optimizer smac --evaluation partial
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --optimizer smac --evaluation partial_bohb

# python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --optimizer mab --evaluation cv
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --optimizer mab --evaluation holdout
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --optimizer mab --evaluation partial
python ./kaggle_spacetitanic_exp.py --Opt cashfe --time_limit 5024 --optimizer mab --evaluation partial_bohb

python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --optimizer smac --evaluation cv
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --optimizer smac --evaluation holdout
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --optimizer smac --evaluation partial
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --optimizer smac --evaluation partial_bohb

python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --optimizer mab --evaluation cv
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --optimizer mab --evaluation holdout
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --optimizer mab --evaluation partial
python ./kaggle_spacetitanic_exp.py --Opt cash --time_limit 5024 --optimizer mab --evaluation partial_bohb
