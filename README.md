### Interesting Links
- [XGBoost Feature Importance](https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python)

### Interesting Note
> probably on topic, will just leave this here :
> 
> ```rstudio
> table(round(train$v50/0.00146724379,2))
> table(round(train$v40/0.0007235366152,2))
> table(round(train$v10/0.0218818357511,2))
> ```
> there are many more, but it might help you to understand what data we have and why chain dependencies are present :)
> 
> This might come in handy for someone as well:
> 
> ```rstudio
> train[numerics][round(train[numerics],5)==0]=0
> ```
> 
> Cheers.
>
>
> maybe this simple script will help:
> 
> ```rstudio
> v50=sort(train$v50)
> v50_diff=v50[2:length(v50)]- v50[1:(length(v50)-1)]
> table(round(v50_diff,5))
> ```
> 
> now if you looked at the table, you will see that table values are increasing 
> by a factor of 0.00147*x, which is strong indication of v50 being a integer 
> variable
>
> - Raddar